#!/usr/bin/env python3
"""
================================================================================
Minimal Runnable Prototype: PEER & Engram for Llama-style Transformers
================================================================================

This prototype demonstrates both approaches with a small toy model.
Run with: python prototype.py

Requirements: pip install torch numpy
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ToyConfig:
    """Small config for testing - scale up for real use"""
    # Model dimensions (small for testing)
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 4
    vocab_size: int = 1000
    max_seq_len: int = 128

    # PEER config
    num_experts: int = 10000        # 100x100 product keys (small for testing)
    experts_per_head: int = 8
    peer_heads: int = 4
    dim_key: int = 32

    # Engram config
    max_ngram: int = 3
    ngram_embed_dim: int = 64
    ngram_heads: int = 4
    ngram_vocab_size: int = 5000

    # Which layers use which approach
    peer_layers: list = None      # None = use PEER on all
    engram_layers: list = None    # None = no Engram

    def __post_init__(self):
        if self.engram_layers is None:
            self.engram_layers = [0, 2]  # Early layers


# =============================================================================
# BASIC COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Simplified RoPE"""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len: int):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# PEER LAYER (Million Experts - scaled down for testing)
# =============================================================================

class PEER(nn.Module):
    """
    Parameter Efficient Expert Retrieval

    Replaces FFN with sparse expert lookup via product keys.
    """

    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size

        # Product keys setup
        self.num_keys = int(math.sqrt(config.num_experts))
        assert self.num_keys ** 2 == config.num_experts

        # Query projection -> split for product key matching
        self.to_queries = nn.Linear(dim, config.dim_key * config.peer_heads * 2, bias=False)

        # Product keys: [heads, num_keys, 2, dim_key]
        self.keys = nn.Parameter(torch.randn(config.peer_heads, self.num_keys, 2, config.dim_key) * 0.02)

        # Expert embeddings (the "micro-experts")
        self.expert_down = nn.Embedding(config.num_experts, dim)
        self.expert_up = nn.Embedding(config.num_experts, dim)

        nn.init.normal_(self.expert_down.weight, std=0.02)
        nn.init.normal_(self.expert_up.weight, std=0.02)

        self.k = config.experts_per_head
        self.activation = nn.GELU()
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.config.peer_heads
        k = self.k
        num_keys = self.num_keys

        # Normalize input
        x_norm = self.norm(x)

        # Project to queries: [B, T, h*dim_key*2] -> [2, B, T, h, dim_key]
        queries = self.to_queries(x_norm)
        queries = queries.view(B, T, h, 2, self.config.dim_key).permute(3, 0, 1, 2, 4)

        # Compute similarity with product keys
        # queries: [2, B, T, h, dim_key], keys: [h, num_keys, 2, dim_key]
        sim1 = torch.einsum('bthd,hkd->bthk', queries[0], self.keys[:, :, 0, :])  # [B, T, h, num_keys]
        sim2 = torch.einsum('bthd,hkd->bthk', queries[1], self.keys[:, :, 1, :])  # [B, T, h, num_keys]

        # Top-k from each key set
        scores1, idx1 = sim1.topk(k, dim=-1)  # [B, T, h, k]
        scores2, idx2 = sim2.topk(k, dim=-1)  # [B, T, h, k]

        # Cartesian product: combine top-k from each side
        # This gives k*k candidates, then we take final top-k
        idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, -1, -1, k)      # [B, T, h, k, k]
        idx2_exp = idx2.unsqueeze(-2).expand(-1, -1, -1, k, -1)      # [B, T, h, k, k]
        scores1_exp = scores1.unsqueeze(-1).expand(-1, -1, -1, -1, k)
        scores2_exp = scores2.unsqueeze(-2).expand(-1, -1, -1, k, -1)

        # Flat expert indices and combined scores
        all_indices = (idx1_exp * num_keys + idx2_exp).view(B, T, h, k * k)
        all_scores = (scores1_exp + scores2_exp).view(B, T, h, k * k)

        # Final top-k selection
        final_scores, pk_idx = all_scores.topk(k, dim=-1)  # [B, T, h, k]
        final_indices = all_indices.gather(-1, pk_idx)      # [B, T, h, k]

        # Retrieve expert weights
        weights_down = self.expert_down(final_indices)  # [B, T, h, k, D]
        weights_up = self.expert_up(final_indices)      # [B, T, h, k, D]

        # Forward through experts
        # Project in: [B, T, D] x [B, T, h, k, D] -> [B, T, h, k]
        hidden = torch.einsum('btd,bthkd->bthk', x_norm, weights_down)
        hidden = self.activation(hidden)

        # Weight by scores (ReLU for non-competing)
        hidden = hidden * F.relu(final_scores)

        # Project out: [B, T, h, k] x [B, T, h, k, D] -> [B, T, D]
        output = torch.einsum('bthk,bthkd->btd', hidden, weights_up)

        return output


# =============================================================================
# ENGRAM LAYER (N-gram Memory Lookup)
# =============================================================================

class Engram(nn.Module):
    """
    Engram: O(1) N-gram memory lookup

    Added alongside (not replacing) existing FFN.
    """

    def __init__(self, config: ToyConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        dim = config.hidden_size

        # Embedding tables for each N-gram order
        # (max_ngram - 1) orders: bigram, trigram, etc.
        num_orders = config.max_ngram - 1
        embed_per_head = config.ngram_embed_dim // config.ngram_heads

        self.embeddings = nn.ModuleList([
            nn.Embedding(config.ngram_vocab_size, embed_per_head)
            for _ in range(num_orders * config.ngram_heads)
        ])

        # Projections
        total_embed_dim = num_orders * config.ngram_embed_dim
        self.value_proj = nn.Linear(total_embed_dim, dim)
        self.key_proj = nn.Linear(total_embed_dim, dim)

        # Gating norms
        self.norm_k = RMSNorm(dim)
        self.norm_q = RMSNorm(dim)

        # Output conv
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim, bias=False)
        self.act = nn.SiLU()

        # Initialize conv to near-zero for identity at start
        nn.init.zeros_(self.conv.weight)

        # Hash multipliers (deterministic per layer)
        torch.manual_seed(10007 * layer_idx)
        multipliers = torch.randint(1, 100000, (config.max_ngram,)) * 2 + 1
        self.register_buffer('hash_mult', multipliers)

    def hash_ngrams(self, input_ids: torch.Tensor) -> list:
        """Compute hash indices for each N-gram order"""
        B, T = input_ids.shape
        device = input_ids.device

        hash_indices = []

        for n in range(2, self.config.max_ngram + 1):
            # Gather shifted tokens
            tokens = [input_ids]
            for offset in range(1, n):
                pad = torch.zeros(B, offset, dtype=input_ids.dtype, device=device)
                shifted = torch.cat([pad, input_ids[:, :-offset]], dim=1)
                tokens.append(shifted)

            # Multiplicative-XOR hash
            h = tokens[0] * self.hash_mult[0]
            for i in range(1, n):
                h = h ^ (tokens[i] * self.hash_mult[i])

            # Generate indices for each head
            for head in range(self.config.ngram_heads):
                idx = (h + head * 7919) % self.config.ngram_vocab_size  # 7919 is prime
                hash_indices.append(idx)

        return hash_indices  # List of [B, T] tensors

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape

        # 1. Hash and lookup
        hash_indices = self.hash_ngrams(input_ids)

        embeds = []
        for i, idx in enumerate(hash_indices):
            embeds.append(self.embeddings[i](idx))  # [B, T, embed_per_head]

        embeddings = torch.cat(embeds, dim=-1)  # [B, T, total_embed_dim]

        # 2. Context-aware gating
        key = self.key_proj(embeddings)    # [B, T, D]
        query = hidden_states               # [B, T, D]

        # Dot product gating
        gate = (self.norm_k(key) * self.norm_q(query)).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(D)

        # Engram's sqrt-sign gating
        gate = gate.abs().clamp(min=1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)  # [B, T, 1]

        # 3. Gated value
        value = self.value_proj(embeddings)  # [B, T, D]
        value = gate * value

        # 4. Short conv
        value_t = value.transpose(1, 2)  # [B, D, T]
        value_conv = self.conv(value_t)[:, :, :T].transpose(1, 2)
        value_conv = self.act(value_conv)

        return value + value_conv


# =============================================================================
# STANDARD FFN (for comparison/hybrid)
# =============================================================================

class FFN(nn.Module):
    """Standard SwiGLU FFN (Llama-style)"""

    def __init__(self, config: ToyConfig):
        super().__init__()
        dim = config.hidden_size
        hidden = dim * 4  # Standard 4x expansion

        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# ATTENTION
# =============================================================================

class Attention(nn.Module):
    """Multi-head attention with RoPE"""

    def __init__(self, config: ToyConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with PEER/Engram/FFN options"""

    def __init__(self, config: ToyConfig, layer_idx: int, mode: str = 'ffn'):
        super().__init__()
        self.mode = mode
        self.layer_idx = layer_idx

        self.attn = Attention(config)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        # Choose FFN type based on mode
        if mode == 'peer':
            self.ffn = PEER(config)
        elif mode == 'engram':
            self.ffn = FFN(config)  # Keep FFN
            self.engram = Engram(config, layer_idx)
        else:
            self.ffn = FFN(config)
            self.engram = None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Engram (if present) - before attention
        if hasattr(self, 'engram') and self.engram is not None:
            x = self.engram(x, input_ids) + x

        # Attention
        x = self.attn(self.attn_norm(x), mask) + x

        # FFN / PEER
        x = self.ffn(self.ffn_norm(x)) + x

        return x


# =============================================================================
# FULL MODEL
# =============================================================================

class ToyTransformer(nn.Module):
    """
    Small transformer for testing PEER/Engram integration.

    Modes:
    - 'ffn': Standard dense FFN (baseline)
    - 'peer': PEER replaces FFN
    - 'engram': Engram added alongside FFN
    - 'hybrid': Mix of approaches per layer
    """

    def __init__(self, config: ToyConfig, mode: str = 'ffn'):
        super().__init__()
        self.config = config
        self.mode = mode

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build layers based on mode
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if mode == 'hybrid':
                # Engram on early layers, PEER on later
                if i in config.engram_layers:
                    layer_mode = 'engram'
                elif i >= config.num_layers // 2:
                    layer_mode = 'peer'
                else:
                    layer_mode = 'ffn'
            else:
                layer_mode = mode

            self.layers.append(TransformerBlock(config, i, layer_mode))

        self.norm = RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        B, T = input_ids.shape

        # Causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=input_ids.device), diagonal=1)

        # Forward
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x, input_ids=input_ids, mask=mask)

        x = self.norm(x)
        logits = self.head(x)

        # Loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {'logits': logits, 'loss': loss}


# =============================================================================
# TESTING & COMPARISON
# =============================================================================

def count_params(model: nn.Module) -> dict:
    """Count parameters by component"""
    counts = {}
    total = 0

    for name, param in model.named_parameters():
        component = name.split('.')[0]
        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()
        total += param.numel()

    counts['total'] = total
    return counts


def test_forward_pass(model: nn.Module, config: ToyConfig, name: str):
    """Test forward pass and measure"""
    model.eval()

    # Random input
    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = torch.randint(0, config.vocab_size, (B, T))

    # Forward
    with torch.no_grad():
        output = model(input_ids, labels)

    print(f"\n{name}:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

    # Parameter count
    params = count_params(model)
    print(f"  Parameters: {params['total']:,}")

    return output


def compare_models():
    """Compare FFN, PEER, Engram, and Hybrid approaches"""
    print("=" * 70)
    print("PEER & Engram Integration Prototype")
    print("=" * 70)

    config = ToyConfig()

    # Create models
    models = {
        'FFN (baseline)': ToyTransformer(config, mode='ffn'),
        'PEER (replace FFN)': ToyTransformer(config, mode='peer'),
        'Engram (add to FFN)': ToyTransformer(config, mode='engram'),
        'Hybrid (Engram + PEER)': ToyTransformer(config, mode='hybrid'),
    }

    for name, model in models.items():
        test_forward_pass(model, config, name)

    print("\n" + "=" * 70)
    print("All forward passes successful!")
    print("=" * 70)

    # Detailed parameter breakdown for PEER
    print("\nPEER Expert Memory Analysis:")
    peer_model = models['PEER (replace FFN)']
    for name, param in peer_model.named_parameters():
        if 'expert' in name:
            print(f"  {name}: {param.shape} = {param.numel():,} params")

    print("\nEngram Memory Analysis:")
    engram_model = models['Engram (add to FFN)']
    for name, param in engram_model.named_parameters():
        if 'embedding' in name.lower() and 'embed' not in name.split('.')[0]:
            print(f"  {name}: {param.shape} = {param.numel():,} params")


def training_demo():
    """Demo: Simple training loop"""
    print("\n" + "=" * 70)
    print("Training Demo (10 steps)")
    print("=" * 70)

    config = ToyConfig()
    model = ToyTransformer(config, mode='hybrid')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    for step in range(10):
        # Random batch
        input_ids = torch.randint(0, config.vocab_size, (4, 32))
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position

        # Forward
        output = model(input_ids, labels)
        loss = output['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    print("Training demo complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    compare_models()
    training_demo()

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Scale up config for real experiments")
    print("  2. Load pretrained Llama weights")
    print("  3. Replace/augment specific layers")
    print("  4. Fine-tune on target task")
    print("=" * 70)
