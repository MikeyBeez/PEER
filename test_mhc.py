#!/usr/bin/env python3
"""
Test mHC integration with TinyLlama

Tests:
1. mHC module standalone
2. mHC wrapping attention/MLP functions
3. Full forward pass with mHC-augmented model
4. Training stability check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

from llama_mhc import mHC, mHCBlock, StreamExpander, StreamReducer


def test_mhc_standalone():
    """Test basic mHC forward pass"""
    print("\n" + "="*60)
    print("TEST 1: mHC Standalone")
    print("="*60)

    B, T, ns, D = 2, 32, 4, 256

    mhc = mHC(hidden_size=D, num_streams=ns)
    x = torch.randn(B, T, ns, D)

    # Forward pass
    y = mhc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input  - mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output - mean: {y.mean():.4f}, std: {y.std():.4f}")

    # Check gradient flow
    y.sum().backward()
    grad_norm = sum(p.grad.norm().item() for p in mhc.parameters() if p.grad is not None)
    print(f"Gradient norm: {grad_norm:.4f}")

    assert y.shape == x.shape, "Shape mismatch!"
    assert not torch.isnan(y).any(), "NaN in output!"
    print("✓ PASSED")


def test_mhc_with_inner_fn():
    """Test mHC wrapping an inner function"""
    print("\n" + "="*60)
    print("TEST 2: mHC with Inner Function")
    print("="*60)

    B, T, ns, D = 2, 32, 4, 256

    # Simple MLP as inner function
    class SimpleMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4)
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.fc2(F.silu(self.fc1(x)))

    mlp = SimpleMLP(D)
    mhc = mHC(hidden_size=D, num_streams=ns, inner_fn=mlp)

    x = torch.randn(B, T, ns, D)
    y = mhc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")

    # Check gradient flow through both mHC and MLP
    y.sum().backward()
    mhc_grad = sum(p.grad.norm().item() for p in mhc.parameters() if p.grad is not None)
    mlp_grad = sum(p.grad.norm().item() for p in mlp.parameters() if p.grad is not None)
    print(f"mHC gradient norm: {mhc_grad:.4f}")
    print(f"MLP gradient norm: {mlp_grad:.4f}")

    assert mlp_grad > 0, "No gradient flow to MLP!"
    print("✓ PASSED")


def test_stream_expand_reduce():
    """Test stream expansion and reduction"""
    print("\n" + "="*60)
    print("TEST 3: Stream Expand/Reduce")
    print("="*60)

    B, T, D = 2, 32, 256
    ns = 4

    expander = StreamExpander(D, ns)
    reducer = StreamReducer(D, ns)

    x = torch.randn(B, T, D)

    # Expand
    x_exp = expander(x)
    print(f"Original: {x.shape} -> Expanded: {x_exp.shape}")

    # Reduce
    x_red = reducer(x_exp)
    print(f"Expanded: {x_exp.shape} -> Reduced: {x_red.shape}")

    assert x_exp.shape == (B, T, ns, D), "Expand shape mismatch!"
    assert x_red.shape == (B, T, D), "Reduce shape mismatch!"
    print("✓ PASSED")


def test_mhc_training_stability():
    """Test that mHC maintains stable gradients during training"""
    print("\n" + "="*60)
    print("TEST 4: Training Stability")
    print("="*60)

    B, T, ns, D = 4, 64, 4, 128
    num_steps = 100

    # Simple model: expand -> mHC layers -> reduce -> output
    class mHCModel(nn.Module):
        def __init__(self, hidden_size, num_streams, num_layers, vocab_size):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.expander = StreamExpander(hidden_size, num_streams)

            # Stack of mHC layers
            self.layers = nn.ModuleList([
                mHC(hidden_size, num_streams, inner_fn=nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                ))
                for _ in range(num_layers)
            ])

            self.reducer = StreamReducer(hidden_size, num_streams)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

            # Small init for stability
            for layer in self.layers:
                for p in layer.parameters():
                    if p.dim() > 1:
                        nn.init.normal_(p, std=0.02)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = self.expander(x)

            for layer in self.layers:
                x = layer(x)

            x = self.reducer(x)
            return self.lm_head(x)

    vocab_size = 1000
    model = mHCModel(D, ns, num_layers=4, vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    grad_norms = []

    for step in tqdm(range(num_steps), desc="Training"):
        # Random input
        input_ids = torch.randint(0, vocab_size, (B, T))
        labels = torch.randint(0, vocab_size, (B, T))

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()
        losses.append(loss.item())

    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduced: {losses[0] - losses[-1]:.4f}")
    print(f"Avg gradient norm: {sum(grad_norms)/len(grad_norms):.4f}")
    print(f"Max gradient norm: {max(grad_norms):.4f}")

    # Check for training stability
    assert not any(math.isnan(l) for l in losses), "NaN loss detected!"
    assert losses[-1] < losses[0], "Loss didn't decrease!"
    assert max(grad_norms) < 100, "Gradient explosion detected!"

    print("✓ PASSED")


def test_mhc_with_cuda():
    """Test mHC on GPU if available"""
    print("\n" + "="*60)
    print("TEST 5: CUDA Support")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return

    device = torch.device("cuda")
    B, T, ns, D = 2, 128, 4, 512

    mhc = mHC(hidden_size=D, num_streams=ns).to(device)
    x = torch.randn(B, T, ns, D, device=device)

    # Warmup
    for _ in range(3):
        _ = mhc(x)

    torch.cuda.synchronize()

    # Benchmark
    import time
    start = time.perf_counter()
    num_iters = 100

    for _ in range(num_iters):
        y = mhc(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (B * T * num_iters) / elapsed
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    print("✓ PASSED")


def test_sinkhorn_properties():
    """Test that Sinkhorn-Knopp produces valid doubly-stochastic matrices"""
    print("\n" + "="*60)
    print("TEST 6: Sinkhorn-Knopp Properties")
    print("="*60)

    ns = 4
    mhc = mHC(hidden_size=128, num_streams=ns)

    # Generate some input
    x_flat = torch.randn(2, 16, ns * 128)

    # Get the residual weight matrix
    w_pre, w_post, w_res = mhc._compute_weights(x_flat)

    # Check w_res is doubly stochastic (rows and cols sum to 1)
    row_sums = w_res.sum(dim=-1)
    col_sums = w_res.sum(dim=-2)

    print(f"w_res shape: {w_res.shape}")
    print(f"Row sums - mean: {row_sums.mean():.6f}, std: {row_sums.std():.6f}")
    print(f"Col sums - mean: {col_sums.mean():.6f}, std: {col_sums.std():.6f}")

    # Should be very close to 1.0
    assert abs(row_sums.mean() - 1.0) < 0.01, "Row sums not ~1!"
    assert abs(col_sums.mean() - 1.0) < 0.01, "Col sums not ~1!"
    assert row_sums.std() < 0.01, "Row sums not uniform!"
    assert col_sums.std() < 0.01, "Col sums not uniform!"

    # Check all entries are non-negative
    assert (w_res >= 0).all(), "Negative entries in w_res!"

    print("✓ PASSED - Matrix is doubly stochastic")


def main():
    print("="*60)
    print("mHC Integration Tests")
    print("="*60)

    test_mhc_standalone()
    test_mhc_with_inner_fn()
    test_stream_expand_reduce()
    test_sinkhorn_properties()
    test_mhc_training_stability()
    test_mhc_with_cuda()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
