"""
================================================================================
Llama + PEER/Engram Integration Sketch
================================================================================

This file outlines two approaches to modify Llama 3 with sparse expert retrieval:
1. PEER approach: Replace FFN with product-key routed micro-experts
2. Engram approach: Add N-gram memory lookup alongside existing FFN

Target: Llama 3 8B (hidden_size=4096, intermediate_size=14336, num_layers=32)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LlamaPEERConfig:
    """Config for PEER-enhanced Llama"""
    # Original Llama params
    hidden_size: int = 4096
    intermediate_size: int = 14336  # Not used in pure PEER, kept for hybrid
    num_layers: int = 32
    rms_norm_eps: float = 1e-5

    # PEER params
    num_experts: int = 1_000_000      # 1M experts (1024 x 1024 product keys)
    num_experts_per_head: int = 16    # Top-k experts activated
    peer_heads: int = 8               # Number of routing heads
    dim_key: int = 128                # Product key dimension

    # Which layers to convert (None = all, or list of indices)
    peer_layers: Optional[list] = None

    # Hybrid mode: keep some original FFN capacity
    hybrid_ratio: float = 0.0  # 0 = pure PEER, 1 = pure FFN, 0.5 = half each


@dataclass
class LlamaEngramConfig:
    """Config for Engram-enhanced Llama"""
    # Original Llama params
    hidden_size: int = 4096
    num_layers: int = 32
    rms_norm_eps: float = 1e-5

    # Engram params
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    engram_vocab_size: list = None  # Will be set based on tokenizer

    # Which layers get Engram (paper used layers 2 and 15 for 30-layer model)
    engram_layers: list = None

    def __post_init__(self):
        if self.engram_vocab_size is None:
            # Default: ~650K entries per N-gram order
            self.engram_vocab_size = [650000, 650000]
        if self.engram_layers is None:
            # Early and mid layers (scaled from paper's [1,15] for 30 layers)
            self.engram_layers = [2, 16]


# =============================================================================
# APPROACH 1: PEER - Replace FFN with Million Experts
# =============================================================================

class RMSNorm(nn.Module):
    """Llama-style RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class LlamaPEER(nn.Module):
    """
    PEER layer to replace LlamaMLP.

    Instead of: gate_proj -> act -> * up_proj -> down_proj
    We use: product_key_lookup -> retrieve experts -> weighted sum

    Memory: 1M experts * 4096 dim * 2 (up/down) * 2 bytes = ~16GB in fp16
    """

    def __init__(self, config: LlamaPEERConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size

        # Pre-normalization (matches Llama's post_attention_layernorm placement)
        self.norm = RMSNorm(dim, eps=config.rms_norm_eps)

        # Product keys for O(sqrt(N)) lookup
        # sqrt(1M) = 1024 keys per side
        self.num_keys = int(math.sqrt(config.num_experts))
        assert self.num_keys ** 2 == config.num_experts, "num_experts must be perfect square"

        # Query projection: hidden -> (2 * heads * dim_key)
        # Split into two halves for product key matching
        self.to_queries = nn.Linear(
            dim,
            config.dim_key * config.peer_heads * 2,
            bias=False
        )

        # Product keys: [heads, num_keys, 2, dim_key]
        self.keys = nn.Parameter(
            torch.randn(config.peer_heads, self.num_keys, 2, config.dim_key) * 0.02
        )

        # Expert embeddings (the "million experts")
        # Each expert is a single neuron: one down projection, one up projection
        self.expert_down = nn.Embedding(config.num_experts, dim)
        self.expert_up = nn.Embedding(config.num_experts, dim)

        # Initialize small
        nn.init.normal_(self.expert_down.weight, std=0.02)
        nn.init.normal_(self.expert_up.weight, std=0.02)

        self.num_experts_per_head = config.num_experts_per_head
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        B, T, D = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Project to queries and split for product keys
        queries = self.to_queries(x_norm)  # [B, T, heads * dim_key * 2]
        queries = queries.view(B, T, self.config.peer_heads, 2, self.config.dim_key)
        queries = queries.permute(3, 0, 1, 2, 4)  # [2, B, T, heads, dim_key]

        # Compute similarity with both key sets
        # keys: [heads, num_keys, 2, dim_key]
        sim = torch.einsum('pbthd,hkpd->pbthk', queries, self.keys)  # [2, B, T, heads, num_keys]

        # Top-k from each key set
        k = self.num_experts_per_head
        scores_1, idx_1 = sim[0].topk(k, dim=-1)  # [B, T, heads, k]
        scores_2, idx_2 = sim[1].topk(k, dim=-1)  # [B, T, heads, k]

        # Cartesian product of indices: k*k candidates
        # idx_1 * num_keys + idx_2 gives flat expert index
        idx_1_exp = idx_1.unsqueeze(-1).expand(-1, -1, -1, -1, k)  # [B, T, h, k, k]
        idx_2_exp = idx_2.unsqueeze(-2).expand(-1, -1, -1, k, -1)  # [B, T, h, k, k]

        all_indices = idx_1_exp * self.num_keys + idx_2_exp  # [B, T, h, k, k]
        all_indices = all_indices.view(B, T, self.config.peer_heads, k * k)

        # Scores: sum (additive in log space for product)
        scores_1_exp = scores_1.unsqueeze(-1).expand(-1, -1, -1, -1, k)
        scores_2_exp = scores_2.unsqueeze(-2).expand(-1, -1, -1, k, -1)
        all_scores = (scores_1_exp + scores_2_exp).view(B, T, self.config.peer_heads, k * k)

        # Final top-k selection from k*k candidates
        final_scores, pk_idx = all_scores.topk(k, dim=-1)  # [B, T, h, k]
        final_indices = all_indices.gather(-1, pk_idx)      # [B, T, h, k]

        # Retrieve expert weights
        weights_down = self.expert_down(final_indices)  # [B, T, h, k, D]
        weights_up = self.expert_up(final_indices)      # [B, T, h, k, D]

        # Forward through experts (Algorithm 1 from paper)
        # Project in: [B, T, D] @ [B, T, h, k, D] -> [B, T, h, k]
        hidden = torch.einsum('btd,bthkd->bthk', x_norm, weights_down)

        # Activation
        hidden = self.activation(hidden)

        # Weight by routing scores (non-competing: ReLU instead of softmax)
        hidden = hidden * F.relu(final_scores)

        # Project out: [B, T, h, k] @ [B, T, h, k, D] -> [B, T, D]
        output = torch.einsum('bthk,bthkd->btd', hidden, weights_up)

        return output


# =============================================================================
# APPROACH 2: ENGRAM - Add N-gram Memory (alongside existing FFN)
# =============================================================================

class LlamaEngram(nn.Module):
    """
    Engram module to ADD to Llama (not replace FFN).

    Provides O(1) lookup of N-gram patterns via deterministic hashing.
    Added as residual: hidden_states = engram(hidden_states) + hidden_states

    Memory: ~27B params can be offloaded to CPU with ~3% throughput loss
    """

    def __init__(self, config: LlamaEngramConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Total embedding dimension across all N-gram orders and heads
        total_ngram_heads = (config.max_ngram_size - 1) * config.n_head_per_ngram
        embed_dim_per_head = config.n_embed_per_ngram // config.n_head_per_ngram

        # Embedding tables for each N-gram order and head
        # In practice, use MultiHeadEmbedding with offsets (see engram_demo)
        self.embeddings = nn.ModuleList([
            nn.Embedding(config.engram_vocab_size[n-2], embed_dim_per_head)
            for n in range(2, config.max_ngram_size + 1)
            for _ in range(config.n_head_per_ngram)
        ])

        # Projection from concatenated embeddings to hidden size
        engram_hidden = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden, config.hidden_size)
        self.key_proj = nn.Linear(engram_hidden, config.hidden_size)

        # Normalization for gating
        self.norm_key = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Short convolution for output smoothing
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size,
            kernel_size=4, padding=3, groups=config.hidden_size, bias=False
        )
        self.act = nn.SiLU()

        # Hash parameters (deterministic, seeded by layer)
        self._init_hash_params()

    def _init_hash_params(self):
        """Initialize multiplicative hash parameters per layer"""
        import numpy as np
        seed = 10007 * self.layer_idx
        rng = np.random.default_rng(seed)

        # Odd multipliers for multiplicative hashing
        max_mult = np.iinfo(np.int64).max // 200000  # Safe range
        multipliers = rng.integers(0, max_mult, size=(self.config.max_ngram_size,)) * 2 + 1
        self.register_buffer(
            'hash_multipliers',
            torch.tensor(multipliers, dtype=torch.long)
        )

    def hash_ngrams(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute hash indices for N-grams.

        Args:
            input_ids: [B, T] token IDs
        Returns:
            hash_indices: [B, T, num_ngram_orders * num_heads]
        """
        B, T = input_ids.shape
        device = input_ids.device

        all_hashes = []

        for n in range(2, self.config.max_ngram_size + 1):
            # Gather N-gram tokens with padding
            indices = []
            for offset in range(n):
                if offset == 0:
                    shifted = input_ids
                else:
                    pad = torch.zeros(B, offset, dtype=input_ids.dtype, device=device)
                    shifted = torch.cat([pad, input_ids[:, :-offset]], dim=1)
                indices.append(shifted)

            # Multiplicative-XOR hash
            hash_val = indices[0] * self.hash_multipliers[0]
            for k in range(1, n):
                hash_val = hash_val ^ (indices[k] * self.hash_multipliers[k])

            # Map to embedding table indices (one per head)
            vocab_size = self.config.engram_vocab_size[n - 2]
            for head in range(self.config.n_head_per_ngram):
                # Different prime modulus per head (simplified)
                prime = vocab_size + head * 7  # Rough approximation
                head_hash = hash_val % prime
                head_hash = head_hash.clamp(0, vocab_size - 1)
                all_hashes.append(head_hash)

        return torch.stack(all_hashes, dim=-1)  # [B, T, total_heads]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] current hidden states
            input_ids: [B, T] original token IDs for hashing
        Returns:
            output: [B, T, D] engram contribution (add as residual)
        """
        B, T, D = hidden_states.shape

        # 1. Hash input_ids to get embedding indices
        hash_indices = self.hash_ngrams(input_ids)  # [B, T, num_heads]

        # 2. Lookup embeddings
        embeds = []
        for i, embed_layer in enumerate(self.embeddings):
            idx = hash_indices[..., i]  # [B, T]
            embeds.append(embed_layer(idx))  # [B, T, embed_dim]

        embeddings = torch.cat(embeds, dim=-1)  # [B, T, total_embed_dim]

        # 3. Context-aware gating
        key = self.key_proj(embeddings)      # [B, T, D]
        query = hidden_states                 # [B, T, D]

        # Gate = sigmoid(normalized_key . normalized_query / sqrt(d))
        gate = (self.norm_key(key) * self.norm_query(query)).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(D)

        # Engram's special gating: sqrt(abs(x)) * sign(x) then sigmoid
        gate = gate.abs().clamp(min=1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)  # [B, T, 1]

        # 4. Gated value projection
        value = self.value_proj(embeddings)  # [B, T, D]
        value = gate * value

        # 5. Short convolution for smoothing
        value_conv = self.conv(value.transpose(1, 2))[:, :, :T].transpose(1, 2)
        value_conv = self.act(value_conv)

        output = value + value_conv

        return output


# =============================================================================
# MODIFIED LLAMA DECODER LAYER
# =============================================================================

class LlamaDecoderLayerPEER(nn.Module):
    """Llama decoder layer with PEER replacing FFN"""

    def __init__(self, config: LlamaPEERConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Standard Llama attention (placeholder - use real implementation)
        self.self_attn = None  # LlamaAttention(config, layer_idx)

        # Input layernorm (before attention)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Post-attention layernorm (before FFN/PEER)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # PEER instead of MLP (if this layer should use PEER)
        use_peer = (config.peer_layers is None or layer_idx in config.peer_layers)

        if use_peer:
            self.mlp = LlamaPEER(config)
            self.use_peer = True
        else:
            # Keep original FFN for some layers (hybrid approach)
            self.mlp = None  # LlamaMLP(config)
            self.use_peer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:

        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # FFN / PEER
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaDecoderLayerEngram(nn.Module):
    """Llama decoder layer with Engram added (keeps original FFN)"""

    def __init__(self, config: LlamaEngramConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Standard Llama components (placeholders)
        self.self_attn = None  # LlamaAttention(config, layer_idx)
        self.mlp = None        # LlamaMLP(config) - KEEP original FFN

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Add Engram at specific layers
        if layer_idx in config.engram_layers:
            self.engram = LlamaEngram(config, layer_idx)
        else:
            self.engram = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,  # Needed for Engram hashing
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:

        # Engram (if present) - BEFORE attention per paper
        if self.engram is not None:
            hidden_states = self.engram(hidden_states, input_ids) + hidden_states

        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # FFN (original, unchanged)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_peer_integration():
    """Example: Create PEER-enhanced Llama layer"""
    config = LlamaPEERConfig(
        hidden_size=4096,
        num_experts=1_000_000,
        num_experts_per_head=16,
        peer_heads=8,
        dim_key=128,
        peer_layers=[0, 1, 2, 3],  # Only first 4 layers use PEER
    )

    layer = LlamaDecoderLayerPEER(config, layer_idx=0)

    # Test forward pass
    x = torch.randn(2, 128, 4096)  # [batch, seq, hidden]
    # out = layer(x)

    print(f"PEER layer created")
    print(f"  Expert embeddings: {config.num_experts * config.hidden_size * 2 / 1e9:.2f}B params")
    print(f"  Memory (fp16): {config.num_experts * config.hidden_size * 2 * 2 / 1e9:.2f}GB")


def example_engram_integration():
    """Example: Create Engram-enhanced Llama layer"""
    config = LlamaEngramConfig(
        hidden_size=4096,
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        engram_vocab_size=[650000, 650000],
        engram_layers=[2, 16],  # Early and mid layers
    )

    layer = LlamaDecoderLayerEngram(config, layer_idx=2)

    print(f"Engram layer created")
    print(f"  Embedding tables: 2 orders * 8 heads * 650K * 64 dim")


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def convert_llama_to_peer(
    original_model,  # HF LlamaForCausalLM
    peer_config: LlamaPEERConfig,
    layers_to_convert: Optional[list] = None
):
    """
    Convert existing Llama model to use PEER layers.

    Strategy:
    1. Keep all attention layers unchanged
    2. Replace selected FFN layers with PEER
    3. Optionally initialize PEER from FFN weights (knowledge distillation)
    """
    # This would be the actual conversion code
    # For now, just a sketch
    pass


def convert_llama_to_engram(
    original_model,  # HF LlamaForCausalLM
    engram_config: LlamaEngramConfig,
):
    """
    Add Engram modules to existing Llama model.

    Strategy:
    1. Keep entire model unchanged
    2. Insert Engram modules at specified layers
    3. Initialize Engram embeddings (random or from corpus statistics)
    """
    pass


# =============================================================================
# TRAINING CONSIDERATIONS
# =============================================================================

"""
PEER Training Notes:
--------------------
1. Start with small expert count (e.g., 10K) and scale up
2. Use auxiliary load balancing loss to ensure experts are utilized
3. Consider freezing backbone and only training PEER initially
4. Memory: Use gradient checkpointing aggressively

Engram Training Notes:
----------------------
1. Can train Engram modules separately (sparse finetuning)
2. Embedding tables can live in CPU memory during training
3. Use Muon optimizer for backbone, Adam for embeddings (5x LR)
4. Initialize convolution weights to zero for identity at start

Hybrid Approach:
----------------
Consider combining both:
- Engram for factual/pattern retrieval (early layers)
- PEER for reasoning/computation (later layers)
- Keep some dense FFN for stability
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Llama + PEER/Engram Integration Sketch")
    print("=" * 60)

    example_peer_integration()
    print()
    example_engram_integration()

    print("\nFiles in project:")
    print("  - PEER-pytorch/    (lucidrains implementation)")
    print("  - Engram/          (DeepSeek implementation)")
    print("  - llama_integration_sketch.py (this file)")
