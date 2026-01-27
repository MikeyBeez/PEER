#!/usr/bin/env python3
"""
================================================================================
Llama + PEER/Engram Integration
================================================================================

This script loads a real Llama model and injects PEER or Engram modules.

Supported models:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache 2.0, no auth required)
- meta-llama/Llama-3.2-1B (requires HF token)

Usage:
    # Install dependencies first:
    pip install torch transformers accelerate datasets

    # Run demo:
    python llama_peer_engram.py --mode engram --demo

    # Train on a dataset:
    python llama_peer_engram.py --mode peer --train --dataset wikitext

================================================================================
"""

import os
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PEERConfig:
    """Configuration for PEER layers"""
    num_experts: int = 16384          # 128x128 for testing, scale to 1M for real
    experts_per_head: int = 8
    num_heads: int = 4
    dim_key: int = 64
    dropout: float = 0.0
    non_competing: bool = True        # ReLU vs softmax for scores


@dataclass
class EngramConfig:
    """Configuration for Engram layers"""
    max_ngram: int = 3                # 2-gram and 3-gram
    embed_dim_per_ngram: int = 256
    num_heads_per_ngram: int = 8
    vocab_size_per_ngram: int = 100000
    kernel_size: int = 4


@dataclass
class IntegrationConfig:
    """Main configuration for Llama + PEER/Engram"""
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Integration mode
    mode: str = "engram"              # 'peer', 'engram', 'hybrid'

    # Which layers to modify (None = auto-select)
    peer_layers: Optional[List[int]] = None
    engram_layers: Optional[List[int]] = None

    # Training
    learning_rate: float = 1e-5       # Lower LR for stability
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0        # Gradient clipping
    num_epochs: int = 1
    batch_size: int = 4
    max_length: int = 256
    gradient_accumulation_steps: int = 4

    # Freezing strategy
    freeze_backbone: bool = False     # Don't freeze - use different LRs instead

    # PEER and Engram configs
    peer: PEERConfig = field(default_factory=PEERConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)


# =============================================================================
# PEER MODULE (for Llama)
# =============================================================================

class LlamaPEER(nn.Module):
    """
    PEER layer compatible with Llama's hidden dimensions.

    Replaces the MLP in a Llama decoder layer.
    """

    def __init__(self, hidden_size: int, config: PEERConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config

        # Product key dimensions
        self.num_keys = int(math.sqrt(config.num_experts))
        assert self.num_keys ** 2 == config.num_experts, \
            f"num_experts ({config.num_experts}) must be a perfect square"

        # Query projection
        self.to_queries = nn.Linear(
            hidden_size,
            config.dim_key * config.num_heads * 2,
            bias=False
        )

        # Product keys
        self.keys = nn.Parameter(
            torch.randn(config.num_heads, self.num_keys, 2, config.dim_key) * 0.02
        )

        # Expert embeddings
        self.expert_down = nn.Embedding(config.num_experts, hidden_size)
        self.expert_up = nn.Embedding(config.num_experts, hidden_size)

        # Initialize small for stability
        nn.init.normal_(self.expert_down.weight, std=0.01)
        nn.init.normal_(self.expert_up.weight, std=0.01)

        self.activation = nn.SiLU()  # Match Llama's activation
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        B, T, D = x.shape
        h = self.config.num_heads
        k = self.config.experts_per_head
        num_keys = self.num_keys

        # Project to queries
        queries = self.to_queries(x)  # [B, T, h * dim_key * 2]
        queries = queries.view(B, T, h, 2, self.config.dim_key)
        queries = queries.permute(3, 0, 1, 2, 4)  # [2, B, T, h, dim_key]

        # Compute similarities with both key sets
        sim1 = torch.einsum('bthd,hkd->bthk', queries[0], self.keys[:, :, 0, :])
        sim2 = torch.einsum('bthd,hkd->bthk', queries[1], self.keys[:, :, 1, :])

        # Top-k from each
        scores1, idx1 = sim1.topk(k, dim=-1)
        scores2, idx2 = sim2.topk(k, dim=-1)

        # Cartesian product
        idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, -1, -1, k)
        idx2_exp = idx2.unsqueeze(-2).expand(-1, -1, -1, k, -1)
        scores1_exp = scores1.unsqueeze(-1).expand(-1, -1, -1, -1, k)
        scores2_exp = scores2.unsqueeze(-2).expand(-1, -1, -1, k, -1)

        all_indices = (idx1_exp * num_keys + idx2_exp).view(B, T, h, k * k)
        all_scores = (scores1_exp + scores2_exp).view(B, T, h, k * k)

        # Final top-k
        final_scores, pk_idx = all_scores.topk(k, dim=-1)
        final_indices = all_indices.gather(-1, pk_idx)

        # Retrieve experts
        weights_down = self.expert_down(final_indices)  # [B, T, h, k, D]
        weights_up = self.expert_up(final_indices)

        # Forward through experts
        hidden = torch.einsum('btd,bthkd->bthk', x, weights_down)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Score activation
        if self.config.non_competing:
            hidden = hidden * F.relu(final_scores)
        else:
            hidden = hidden * F.softmax(final_scores, dim=-1)

        # Project out
        output = torch.einsum('bthk,bthkd->btd', hidden, weights_up)

        if squeeze_output:
            output = output.squeeze(0)

        return output


# =============================================================================
# ENGRAM MODULE (for Llama)
# =============================================================================

class LlamaEngram(nn.Module):
    """
    Engram layer compatible with Llama.

    Added alongside (not replacing) the MLP.
    """

    def __init__(self, hidden_size: int, config: EngramConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.layer_idx = layer_idx

        # Number of N-gram orders (e.g., 2-gram, 3-gram = 2 orders)
        num_orders = config.max_ngram - 1
        embed_per_head = config.embed_dim_per_ngram // config.num_heads_per_ngram

        # Embedding tables
        self.embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size_per_ngram, embed_per_head)
            for _ in range(num_orders * config.num_heads_per_ngram)
        ])

        # Projections
        total_embed = num_orders * config.embed_dim_per_ngram
        self.value_proj = nn.Linear(total_embed, hidden_size, bias=False)
        self.key_proj = nn.Linear(total_embed, hidden_size, bias=False)

        # Gating normalization
        self.norm_k = nn.RMSNorm(hidden_size, eps=1e-5)
        self.norm_q = nn.RMSNorm(hidden_size, eps=1e-5)

        # Output convolution
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=config.kernel_size,
            padding=config.kernel_size - 1,
            groups=hidden_size,
            bias=False
        )
        self.act = nn.SiLU()

        # Zero-init conv for identity at start
        nn.init.zeros_(self.conv.weight)

        # Small init for projections so Engram starts with minimal output
        nn.init.normal_(self.value_proj.weight, std=0.01)
        nn.init.normal_(self.key_proj.weight, std=0.01)

        # Small init for embeddings too
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        # Hash parameters (deterministic per layer)
        self._init_hash(layer_idx)

    def _init_hash(self, layer_idx: int):
        """Initialize hash multipliers"""
        torch.manual_seed(10007 * (layer_idx + 1))
        max_mult = 2**31 // 200000
        multipliers = torch.randint(1, max_mult, (self.config.max_ngram,)) * 2 + 1
        self.register_buffer('hash_mult', multipliers.long())

    def hash_ngrams(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Compute hash indices for N-grams"""
        B, T = input_ids.shape
        device = input_ids.device
        vocab_size = self.config.vocab_size_per_ngram

        indices_list = []

        for n in range(2, self.config.max_ngram + 1):
            # Gather N-gram context
            tokens = [input_ids]
            for offset in range(1, n):
                pad = torch.zeros(B, offset, dtype=input_ids.dtype, device=device)
                shifted = torch.cat([pad, input_ids[:, :-offset]], dim=1)
                tokens.append(shifted)

            # Hash
            h = tokens[0].long() * self.hash_mult[0]
            for i in range(1, n):
                h = h ^ (tokens[i].long() * self.hash_mult[i])

            # Per-head indices
            for head in range(self.config.num_heads_per_ngram):
                idx = ((h + head * 104729) % vocab_size).clamp(0, vocab_size - 1)
                indices_list.append(idx)

        return indices_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        debug: bool = False
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape

        # Stay in the same dtype as input (let autocast handle precision)
        if debug:
            print(f"  [Engram] Input: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, has_nan={hidden_states.isnan().any()}")

        # Hash and lookup
        hash_indices = self.hash_ngrams(input_ids)

        embeds = []
        for i, idx in enumerate(hash_indices):
            embeds.append(self.embeddings[i](idx))

        embeddings = torch.cat(embeds, dim=-1)  # [B, T, total_embed]

        if debug:
            print(f"  [Engram] Embeddings: min={embeddings.min():.4f}, max={embeddings.max():.4f}, has_nan={embeddings.isnan().any()}")

        # Context-aware gating
        key = self.key_proj(embeddings)
        query = hidden_states

        if debug:
            print(f"  [Engram] Key: min={key.min():.4f}, max={key.max():.4f}, has_nan={key.isnan().any()}")

        gate = (self.norm_k(key) * self.norm_q(query)).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(D)

        if debug:
            print(f"  [Engram] Gate pre-activation: min={gate.min():.4f}, max={gate.max():.4f}, has_nan={gate.isnan().any()}")

        # Engram's sqrt-sign gating (with numerical stability)
        gate_abs = gate.abs().clamp(min=1e-8, max=100.0)  # Clamp for stability
        gate = gate_abs.sqrt() * gate.sign()
        gate = torch.sigmoid(gate.clamp(-10, 10))  # Prevent extreme values

        if debug:
            print(f"  [Engram] Gate post-activation: min={gate.min():.4f}, max={gate.max():.4f}, has_nan={gate.isnan().any()}")

        # Gated value
        value = self.value_proj(embeddings)
        value = gate * value

        if debug:
            print(f"  [Engram] Value: min={value.min():.4f}, max={value.max():.4f}, has_nan={value.isnan().any()}")

        # Conv smoothing
        value_t = value.transpose(1, 2)
        value_conv = self.conv(value_t)[:, :, :T].transpose(1, 2)
        value_conv = self.act(value_conv)

        output = value + value_conv

        if debug:
            print(f"  [Engram] Output: min={output.min():.4f}, max={output.max():.4f}, has_nan={output.isnan().any()}")

        return output


# =============================================================================
# MODIFIED LLAMA DECODER LAYER
# =============================================================================

class ModifiedLlamaDecoderLayer(nn.Module):
    """
    Wrapper that adds PEER/Engram to an existing Llama decoder layer.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        layer_idx: int,
        mode: str,
        peer_config: Optional[PEERConfig] = None,
        engram_config: Optional[EngramConfig] = None,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.mode = mode
        self.hidden_size = hidden_size
        self._parent_model_ref = None  # Weak reference, set after injection

        self.peer = None
        self.engram = None

        if mode in ['peer', 'hybrid'] and peer_config is not None:
            self.peer = LlamaPEER(hidden_size, peer_config)

        if mode in ['engram', 'hybrid'] and engram_config is not None:
            self.engram = LlamaEngram(hidden_size, engram_config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Get input_ids from parent model if not passed directly
        if input_ids is None and self._parent_model_ref is not None:
            parent = self._parent_model_ref()
            if parent is not None:
                input_ids = getattr(parent, '_current_input_ids', None)

        # Engram (before attention, per paper)
        if self.engram is not None and input_ids is not None:
            # Check for debug flag
            debug = getattr(self, '_debug', False)
            engram_out = self.engram(hidden_states, input_ids, debug=debug)
            hidden_states = engram_out + hidden_states

            if debug:
                print(f"  [Layer {self.layer_idx}] After Engram: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, has_nan={hidden_states.isnan().any()}")

        # Original layer forward
        outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # PEER (replaces/augments MLP output)
        if self.peer is not None:
            # Ensure hidden_states is 3D [B, T, D]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            peer_out = self.peer(hidden_states)
            hidden_states = hidden_states + peer_out

        # Return in same format as original
        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]

        # Handle BaseModelOutputWithPast or similar
        if hasattr(outputs, 'last_hidden_state'):
            outputs.last_hidden_state = hidden_states
            return outputs

        return outputs


# =============================================================================
# MODEL MODIFICATION UTILITIES
# =============================================================================

def inject_peer_engram(
    model,
    config: IntegrationConfig,
) -> nn.Module:
    """
    Inject PEER/Engram modules into a Llama model.

    Args:
        model: HuggingFace LlamaForCausalLM or similar
        config: Integration configuration

    Returns:
        Modified model
    """
    # Get model config
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    # Auto-select layers if not specified
    if config.peer_layers is None:
        # PEER on a few later layers (reasoning) - limited for memory
        config.peer_layers = [num_layers - 3, num_layers - 1]  # Just 2 layers for demo

    if config.engram_layers is None:
        # Engram on early layers (pattern matching)
        config.engram_layers = [1, num_layers // 4]

    print(f"Model: {config.model_name}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num layers: {num_layers}")
    print(f"Mode: {config.mode}")
    print(f"PEER layers: {config.peer_layers if config.mode in ['peer', 'hybrid'] else 'N/A'}")
    print(f"Engram layers: {config.engram_layers if config.mode in ['engram', 'hybrid'] else 'N/A'}")

    # Get the layers module
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    # Modify selected layers
    for idx in range(num_layers):
        should_modify = False
        layer_mode = None

        if config.mode == 'peer' and idx in config.peer_layers:
            should_modify = True
            layer_mode = 'peer'
        elif config.mode == 'engram' and idx in config.engram_layers:
            should_modify = True
            layer_mode = 'engram'
        elif config.mode == 'hybrid':
            if idx in config.engram_layers or idx in config.peer_layers:
                should_modify = True
                layer_mode = 'hybrid'

        if should_modify:
            original_layer = layers[idx]

            # Get device and dtype from original layer
            device = next(original_layer.parameters()).device
            dtype = next(original_layer.parameters()).dtype

            modified_layer = ModifiedLlamaDecoderLayer(
                original_layer=original_layer,
                hidden_size=hidden_size,
                layer_idx=idx,
                mode=layer_mode,
                peer_config=config.peer if layer_mode in ['peer', 'hybrid'] else None,
                engram_config=config.engram if layer_mode in ['engram', 'hybrid'] else None,
            )

            # Move new modules (peer/engram) to same device and dtype as original
            if modified_layer.peer is not None:
                modified_layer.peer = modified_layer.peer.to(device=device, dtype=dtype)
            if modified_layer.engram is not None:
                modified_layer.engram = modified_layer.engram.to(device=device, dtype=dtype)

            # Set weak parent reference for accessing _current_input_ids
            if hasattr(model, 'model'):
                modified_layer._parent_model_ref = weakref.ref(model.model)
            else:
                modified_layer._parent_model_ref = weakref.ref(model)

            layers[idx] = modified_layer
            print(f"  Layer {idx}: Modified with {layer_mode} (device={device})")

    # Freeze backbone if requested
    if config.freeze_backbone:
        frozen_count = 0
        trainable_count = 0

        for name, param in model.named_parameters():
            if 'peer' in name or 'engram' in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        # NOTE: We do NOT unfreeze embed_tokens or lm_head
        # Gradients flow through frozen layers (requires_grad=False just means
        # the gradient for that parameter isn't accumulated, but the chain rule
        # still applies for downstream parameters)

        print(f"\nFrozen params: {frozen_count:,}")
        print(f"Trainable params: {trainable_count:,}")

    return model


# =============================================================================
# CUSTOM FORWARD FOR INPUT_IDS PROPAGATION
# =============================================================================

def create_forward_with_input_ids(model):
    """
    Create a custom forward that passes input_ids to all layers.
    Needed for Engram to access the original tokens.
    """
    original_forward = model.forward

    def forward_with_input_ids(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Store input_ids for layers to access
        if hasattr(model, 'model'):
            model.model._current_input_ids = input_ids
        else:
            model._current_input_ids = input_ids

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    model.forward = forward_with_input_ids
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_step(
    model,
    batch,
    optimizer,
    device,
    gradient_accumulation_steps: int = 1,
    step: int = 0,
    max_grad_norm: float = 1.0,
):
    """Single training step with bfloat16 autocast for CUDA"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = input_ids.clone()

    # Forward with bfloat16 autocast (no scaler needed for bfloat16)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / gradient_accumulation_steps

    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * gradient_accumulation_steps


def evaluate(model, dataloader, device, max_batches: int = 50):
    """Evaluate model on dataloader"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            count += 1

    model.train()
    return total_loss / max(count, 1)


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def run_demo(config: IntegrationConfig):
    """Run a demo showing the integration works"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Llama + PEER/Engram Demo")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\nLoading {config.model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU demo
        device_map="cpu",
    )

    # Count original params
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    # Inject PEER/Engram
    print(f"\nInjecting {config.mode} modules...")
    model = inject_peer_engram(model, config)

    # Count new params
    new_params = sum(p.numel() for p in model.parameters())
    print(f"New parameters: {new_params:,}")
    print(f"Added: {new_params - orig_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)

    # Store input_ids for Engram
    if hasattr(model, 'model'):
        model.model._current_input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])

    print(f"  Input: '{test_text}'")
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")

    # Test generation
    print("\nTesting generation...")
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(model, 'model'):
        model.model._current_input_ids = inputs['input_ids']

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{output_text}'")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    return model, tokenizer


def run_training(config: IntegrationConfig, dataset_name: str = "wikitext"):
    """Run training on a dataset"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("=" * 70)
    print(f"Training Llama + {config.mode.upper()}")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 on CUDA - better numerical stability than float16
    # (float16 has limited exponent range causing overflow in optimizer updates)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    # Note: Gradient checkpointing disabled - causes issues with frozen params
    # If memory is an issue, unfreeze backbone or use smaller batch size

    # Inject modules
    print(f"\nInjecting {config.mode} modules...")
    model = inject_peer_engram(model, config)

    if device.type != "cuda":
        model = model.to(device)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}...")
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    else:
        dataset = load_dataset(dataset_name, split="train")
        text_column = "text"

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Filter empty examples and tokenize
    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0)
    dataset = dataset.select(range(min(len(dataset), 5000)))  # Limit for demo
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format("torch")

    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Optimizer with different LRs for different parameter groups
    new_module_params = []
    embed_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'peer' in name or 'engram' in name:
            new_module_params.append(param)
        elif 'embed' in name or 'lm_head' in name:
            embed_params.append(param)
        else:
            backbone_params.append(param)

    # Different learning rates:
    # - New modules: full LR
    # - Backbone: 1/100th LR
    # - Embeddings: 1/1000th LR (very sensitive to updates)
    param_groups = []
    if new_module_params:
        param_groups.append({'params': new_module_params, 'lr': config.learning_rate})
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': config.learning_rate * 0.01})
    if embed_params:
        param_groups.append({'params': embed_params, 'lr': config.learning_rate * 0.001})

    if not param_groups:
        print("WARNING: No trainable parameters!")
        return model, tokenizer

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    trainable_params = new_module_params + backbone_params + embed_params

    print(f"\nTraining...")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Store input_ids for Engram
            if hasattr(model, 'model') and hasattr(model.model, '_current_input_ids'):
                model.model._current_input_ids = batch['input_ids'].to(device)

            loss = train_step(
                model, batch, optimizer, device,
                config.gradient_accumulation_steps, global_step,
                config.max_grad_norm
            )

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch}, Batch {batch_idx}: loss = {avg_loss:.4f}")

            # Check for NaN and stop early
            if math.isnan(loss):
                print(f"  NaN detected at batch {batch_idx}, stopping...")
                break

            if batch_idx >= 200:  # Limit for demo
                break

        avg_epoch_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch} complete: avg loss = {avg_epoch_loss:.4f}")

    print("\nTraining complete!")

    # Save
    save_path = f"./llama_{config.mode}_finetuned"
    os.makedirs(save_path, exist_ok=True)
    print(f"\nSaving to {save_path}...")

    # Save only the new modules
    peer_engram_state = {}
    for name, param in model.named_parameters():
        if 'peer' in name or 'engram' in name:
            peer_engram_state[name] = param.cpu()

    torch.save(peer_engram_state, f"{save_path}/peer_engram_weights.pt")
    tokenizer.save_pretrained(save_path)

    print("Done!")

    return model, tokenizer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Llama + PEER/Engram Integration")

    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name or path")
    parser.add_argument("--mode", type=str, default="engram",
                        choices=["peer", "engram", "hybrid"],
                        help="Integration mode")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo (forward pass + generation)")
    parser.add_argument("--train", action="store_true",
                        help="Run training")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset for training")
    parser.add_argument("--num-experts", type=int, default=65536,
                        help="Number of PEER experts (must be perfect square)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")

    args = parser.parse_args()

    # Build config
    config = IntegrationConfig(
        model_name=args.model,
        mode=args.mode,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    config.peer.num_experts = args.num_experts

    if args.demo:
        run_demo(config)
    elif args.train:
        run_training(config, args.dataset)
    else:
        print("Specify --demo or --train")
        parser.print_help()


if __name__ == "__main__":
    main()
