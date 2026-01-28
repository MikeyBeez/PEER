# PEER + Engram + mHC: Triple-Sparse Architecture for LLaMA

This project integrates three complementary "brain upgrades" into TinyLlama:

- **PEER** (Parameter Efficient Expert Retrieval) - Routes to specialized micro-experts based on hidden state
- **Engram** - Retrieves static n-gram patterns based on literal text
- **mHC** (Manifold-constrained Hyper-Connections) - Stable multi-stream residual connections

Together they form a **Triple-Sparse Architecture**: PEER handles dynamic reasoning, Engram handles static pattern memory, and mHC stabilizes gradient flow at scale.

## Understanding PEER's Value Proposition

**PEER is fundamentally about compute efficiency, not perplexity improvement.**

Traditional scaling requires linear increases in both memory AND compute as you add parameters. PEER decouples these:

- **Massive capacity** - Store millions of micro-experts as embeddings
- **Constant compute** - Only k experts activated per token, regardless of total count
- **O(√n) retrieval** - Product-key lookup scales sub-linearly with expert count

The key insight: **throughput stays constant as you scale parameters.**

## Efficiency Benchmark

Benchmarked on RTX 5070 Ti (15.5 GB VRAM):

| Config | Parameters | VRAM | Perplexity | Throughput |
|--------|-----------|------|------------|------------|
| Baseline | 1.1B | 2.07 GB | 17.12 | 26,991 tok/s |
| PEER-16K | 1.24B (+12%) | 2.35 GB | 17.12 | 26,521 tok/s |
| PEER-65K | 1.64B (+49%) | 3.10 GB | 17.12 | 26,405 tok/s |
| PEER-262K | 3.25B (+195%) | 6.10 GB | 17.12 | 26,446 tok/s |
| **PEER-803K** | **7.68B (+598%)** | 14.35 GB | 17.12 | 26,262 tok/s |

**Key finding**: With PEER-803K we have **7x the parameters** but **throughput stays constant** (~26K tok/s). Compute doesn't scale with expert count—that's the efficiency win.

Run the benchmark yourself:
```bash
python benchmark_efficiency.py --expert-counts 16384 65536 262144 802816
```

## Quality Results (with Training)

| Model | Perplexity | Change | Notes |
|-------|-----------|--------|-------|
| Base TinyLlama | 16.54 | — | Baseline |
| + Engram (trained) | 11.30 | **-31.7%** | Pattern memory |
| + PEER-262K (trained) | 14.81 | **-10.5%** | 3x params, same throughput |
| + mHC (trained) | 12.29 | **-25.7%** | Multi-stream residuals |
| + Engram + PEER (Hybrid) | 11.06 | **-33.1%** | Best dual combination |
| + Triple-Sparse (all three) | 13.79 | **-16.6%** | PEER + Engram + mHC |

**Key findings**:
- **Engram** (-31.7%) is the strongest single component for perplexity improvement
- **mHC** (-25.7%) provides significant gains via multi-stream residual connections
- **PEER** (-10.5%) provides parameter scaling with constant throughput
- **Engram + PEER** (-33.1%) is the best dual combination
- **Triple-Sparse** (-16.6%) underperforms individual components, suggesting partial redundancy

Train each component:
```bash
python train_and_eval.py          # Engram/PEER/Hybrid
python train_peer_large.py --train --batches 3000 --lr 0.01  # PEER-262K
python train_mhc.py --train --batches 1500  # mHC
python train_triple.py --train --batches 3000  # All three together
```

## Key Insight: Initialization Matters

The critical fix that made this work: **small initialization for new modules**.

When adding components to pretrained models, default PyTorch initialization outputs high-variance noise that corrupts the carefully tuned hidden states. The solution is to initialize new modules to be near-zero at step 0:

```python
# In LlamaEngram.__init__
nn.init.normal_(self.value_proj.weight, std=0.01)
nn.init.normal_(self.key_proj.weight, std=0.01)
for emb in self.embeddings:
    nn.init.normal_(emb.weight, std=0.02)
```

Without this fix, Engram produced PPL of 450 (+2626% worse). With it: PPL 11.30 (-31.67% better).

## Architecture

### PEER (Layers 19, 21)
- 16,384 micro-experts (128x128 product keys)
- 8 experts selected per token per head
- Augments MLP output in later layers

### Engram (Layers 1, 5)
- Hashes 2-gram and 3-gram patterns
- 100K vocab size per n-gram order
- Gated addition to hidden states in early layers

### mHC (Layers 5, 11, 17)
- Expands hidden state into 4 parallel streams
- Uses Sinkhorn-Knopp projection for doubly-stochastic mixing matrices
- Gated output projection (small init for stability)
- **Result: 12.29 perplexity (-25.7%)**
- Based on [DeepSeek's mHC paper](https://arxiv.org/abs/2512.24880)

### Why This Combination Works
- **Engram in early layers**: Recognizes common phrases as units (e.g., "United States", "New York") → improves quality
- **PEER in later layers**: Provides massive parameter capacity with minimal VRAM overhead → enables scaling
- **mHC throughout**: Stabilizes training with manifold-constrained residuals → enables deeper scaling
- They serve different purposes and don't interfere because they operate at different depths

## Usage

### Training
```bash
python train_and_eval.py
```

Trains all three modes (engram, peer, hybrid) with 1500 batches each on WikiText-2, then evaluates perplexity.

### Visualization
```bash
python visualize_engram.py
```

Shows which n-gram patterns the model learned to prioritize. Generates gate activation heatmaps.

## Files

- `llama_peer_engram.py` - Core integration: PEER and Engram modules for LLaMA
- `llama_mhc.py` - mHC (Manifold-constrained Hyper-Connections) module
- `train_and_eval.py` - Training and evaluation script (Engram/PEER/Hybrid)
- `train_peer_large.py` - Train large PEER configurations (262K+ experts)
- `train_mhc.py` - Train mHC and evaluate perplexity
- `test_mhc.py` - Unit tests for mHC module
- `benchmark_efficiency.py` - VRAM/throughput/perplexity benchmark
- `visualize_engram.py` - N-gram activation visualization

## Requirements

```
torch
transformers
datasets
matplotlib
tqdm
```

## References

- PEER: [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153) - Product-key retrieval for massive sparse expert scaling
- Engram: [Byte-Level Memory-Augmented Language Modeling](https://github.com/deepseek-ai/Engram) - N-gram pattern retrieval for quality improvement
- mHC: [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) - Stable multi-stream residual connections

## License

MIT
