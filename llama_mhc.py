#!/usr/bin/env python3
"""
mHC (Manifold-constrained Hyper-Connections) for LLaMA

Based on DeepSeek's mHC paper (arXiv:2512.24880) and VatsaDev's implementation.

mHC replaces standard residual connections with multi-stream hyper-connections
that use Sinkhorn-Knopp projection for training stability.

Integration with PEER + Engram creates a triple-sparse architecture:
- mHC: Stable multi-stream residual connections
- PEER: Sparse expert retrieval
- Engram: N-gram pattern memory
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable


class mHC(nn.Module):
    """
    Manifold-constrained Hyper-Connections layer.

    Wraps a function (attention or MLP) with multi-stream residual connections
    where the mixing matrices are projected to doubly-stochastic via Sinkhorn-Knopp.

    Args:
        hidden_size: Model hidden dimension
        num_streams: Number of parallel streams (default 4)
        inner_fn: The function to wrap (attention or MLP block)
        sinkhorn_iters: Iterations for Sinkhorn-Knopp projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        inner_fn: Optional[Callable] = None,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.inner_fn = inner_fn
        self.sinkhorn_iters = sinkhorn_iters

        ns = num_streams

        # Learnable scaling factors (initialized small for stability)
        self.alpha_pre = nn.Parameter(torch.ones(1) * 0.01)
        self.alpha_post = nn.Parameter(torch.ones(1) * 0.01)
        self.alpha_res = nn.Parameter(torch.ones(1) * 0.01)

        # Input projections for dynamic weight generation
        # These map flattened multi-stream input to mixing weights
        self.proj_pre = nn.Linear(ns * hidden_size, ns, bias=False)
        self.proj_post = nn.Linear(ns * hidden_size, ns, bias=False)
        self.proj_res = nn.Linear(ns * hidden_size, ns * ns, bias=False)

        # Normalization before weight generation
        self.norm = nn.RMSNorm(hidden_size)

        # Initialize projections small
        nn.init.normal_(self.proj_pre.weight, std=0.01)
        nn.init.normal_(self.proj_post.weight, std=0.01)
        nn.init.normal_(self.proj_res.weight, std=0.01)

    def _sinkhorn(self, A: torch.Tensor, iters: int = 20) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm to project matrix to doubly-stochastic.

        A doubly-stochastic matrix has all rows and columns summing to 1,
        which preserves signal magnitude and stabilizes gradient flow.
        """
        eps = 1e-12
        for _ in range(iters):
            # Row normalization
            A = A / (A.sum(dim=-1, keepdim=True) + eps)
            # Column normalization
            A = A / (A.sum(dim=-2, keepdim=True) + eps)
        return A

    def _compute_weights(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the three mixing weight matrices from flattened input.

        Args:
            x_flat: [batch, seq_len, num_streams * hidden_size]

        Returns:
            w_pre: [batch, seq_len, 1, num_streams] - 1-to-N pre-mixing
            w_post: [batch, seq_len, num_streams, 1] - N-to-1 post-mixing
            w_res: [batch, seq_len, num_streams, num_streams] - N-to-N residual
        """
        ns = self.num_streams

        # Generate raw weights via learned projections + tanh
        h_pre = self.alpha_pre + torch.tanh(self.proj_pre(x_flat))   # [B, T, ns]
        h_post = self.alpha_post + torch.tanh(self.proj_post(x_flat)) # [B, T, ns]
        h_res = self.alpha_res + torch.tanh(self.proj_res(x_flat))   # [B, T, ns*ns]

        # Apply manifold constraints
        # Pre: sigmoid for [0,1] weights, 1-to-N mixing
        w_pre = torch.sigmoid(h_pre).unsqueeze(2)  # [B, T, 1, ns]

        # Post: scaled sigmoid, N-to-1 mixing
        w_post = 2 * torch.sigmoid(h_post).unsqueeze(3)  # [B, T, ns, 1]

        # Res: Sinkhorn-projected doubly-stochastic, N-to-N mixing
        h_res_mat = h_res.view(-1, x_flat.size(1), ns, ns)
        w_res = self._sinkhorn(torch.exp(h_res_mat), self.sinkhorn_iters)  # [B, T, ns, ns]

        return w_pre, w_post, w_res

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with hyper-connected residual.

        Args:
            x: Input tensor [batch, seq_len, num_streams, hidden_size]
            *args, **kwargs: Passed to inner_fn

        Returns:
            Output tensor [batch, seq_len, num_streams, hidden_size]
        """
        B, T, ns, D = x.shape

        # Normalize and flatten for weight computation
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, T, ns * D)

        # Compute dynamic mixing weights
        w_pre, w_post, w_res = self._compute_weights(x_flat)

        # Pre-mixing: weighted sum of streams -> single stream
        # [B, T, 1, ns] @ [B, T, ns, D] -> [B, T, 1, D] -> [B, T, D]
        x_in = torch.matmul(w_pre, x).squeeze(2)

        # Apply inner function (attention, MLP, etc.)
        if self.inner_fn is not None:
            result = self.inner_fn(x_in, *args, **kwargs)

            # Handle functions that return tuples (e.g., attention with cache)
            if isinstance(result, tuple):
                y, *extra = result
            else:
                y, extra = result, []
        else:
            y = x_in
            extra = []

        # Residual mixing: mix all streams with doubly-stochastic matrix
        # [B, T, ns, ns] @ [B, T, ns, D] -> [B, T, ns, D]
        x_mix = torch.matmul(w_res, x)

        # Post-mixing: expand function output back to streams
        # [B, T, ns, 1] @ [B, T, 1, D] -> [B, T, ns, D]
        y_exp = torch.matmul(w_post, y.unsqueeze(2))

        # Final hyper-connected output
        out = x_mix + y_exp

        if extra:
            return (out, *extra)
        return out


class mHCBlock(nn.Module):
    """
    A transformer block with mHC residual connections.

    Wraps both attention and MLP with mHC for stable multi-stream training.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        attention_fn: Optional[Callable] = None,
        mlp_fn: Optional[Callable] = None,
        num_layers: int = 1,  # For residual scaling
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_streams = num_streams

        # Pre-normalization
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

        # Store inner functions
        self.attention_fn = attention_fn
        self.mlp_fn = mlp_fn

        # mHC wrappers
        self.mhc_attn = mHC(hidden_size, num_streams, self._attn_wrapper)
        self.mhc_mlp = mHC(hidden_size, num_streams, self._mlp_wrapper)

        # Residual scaling (MuP-style)
        self.residual_scale = 1.0 / math.sqrt(num_layers)

    def _attn_wrapper(self, x, *args, **kwargs):
        """Wrapper to apply norm before attention"""
        if self.attention_fn is not None:
            return self.attention_fn(self.norm1(x), *args, **kwargs)
        return x

    def _mlp_wrapper(self, x, *args, **kwargs):
        """Wrapper to apply norm before MLP"""
        if self.mlp_fn is not None:
            return self.mlp_fn(self.norm2(x), *args, **kwargs)
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_streams, hidden_size]
        """
        # Attention with mHC
        attn_out = self.mhc_attn(x, *args, **kwargs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + self.residual_scale * attn_out

        # MLP with mHC
        mlp_out = self.mhc_mlp(x)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        x = x + self.residual_scale * mlp_out

        return x


class StreamExpander(nn.Module):
    """Expands single-stream input to multi-stream for mHC"""

    def __init__(self, hidden_size: int, num_streams: int):
        super().__init__()
        self.num_streams = num_streams
        # Learnable expansion (or could tile)
        self.expand = nn.Linear(hidden_size, hidden_size * num_streams, bias=False)
        nn.init.normal_(self.expand.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, ns, D]"""
        B, T, D = x.shape
        x_exp = self.expand(x)  # [B, T, ns*D]
        return x_exp.view(B, T, self.num_streams, D)


class StreamReducer(nn.Module):
    """Reduces multi-stream back to single stream"""

    def __init__(self, hidden_size: int, num_streams: int):
        super().__init__()
        self.num_streams = num_streams
        # Learnable reduction (weighted average)
        self.reduce = nn.Linear(hidden_size * num_streams, hidden_size, bias=False)
        nn.init.normal_(self.reduce.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, ns, D] -> [B, T, D]"""
        B, T, ns, D = x.shape
        x_flat = x.view(B, T, ns * D)
        return self.reduce(x_flat)


# =============================================================================
# Integration with existing LLaMA + PEER + Engram
# =============================================================================

def inject_mhc_into_llama(model, num_streams: int = 4, mhc_layers: list = None):
    """
    Inject mHC into a LLaMA model's residual connections.

    This is experimental - full integration would require modifying
    the decoder layer structure significantly.
    """
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if mhc_layers is None:
        # Default: apply to all layers
        mhc_layers = list(range(num_layers))

    print(f"mHC integration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num streams: {num_streams}")
    print(f"  mHC layers: {mhc_layers}")

    # For now, just add stream expansion/reduction at model level
    # Full integration would wrap each decoder layer's attention/MLP

    model.stream_expander = StreamExpander(hidden_size, num_streams)
    model.stream_reducer = StreamReducer(hidden_size, num_streams)

    return model


if __name__ == "__main__":
    # Quick test
    print("Testing mHC module...")

    B, T, ns, D = 2, 16, 4, 128

    # Create mHC layer
    mhc = mHC(hidden_size=D, num_streams=ns)

    # Test input
    x = torch.randn(B, T, ns, D)

    # Forward
    y = mhc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")

    # Test with inner function
    def dummy_mlp(x):
        return F.silu(x) * 0.1

    mhc_with_fn = mHC(hidden_size=D, num_streams=ns, inner_fn=dummy_mlp)
    y2 = mhc_with_fn(x)
    print(f"With inner fn - Output mean: {y2.mean():.4f}, std: {y2.std():.4f}")

    print("\nmHC test passed!")
