# Triple-Sparse Architecture: Combining PEER, Engram, and mHC for Efficient Language Model Scaling

## Abstract

We present a triple-sparse architecture that combines three complementary mechanisms for language models: PEER (Parameter Efficient Expert Retrieval) for compute-efficient parameter scaling, Engram for n-gram pattern memory, and mHC (Manifold-constrained Hyper-Connections) for stable multi-stream residual connections.

Our experiments on TinyLlama (1.1B parameters) demonstrate that PEER enables **7x parameter scaling** (to 7.68B) while maintaining constant throughput (~26K tokens/second). Individual training results show: PEER-262K achieves -10.5% perplexity, Engram achieves -31.7%, and mHC achieves -25.7%.

However, joint training of all three components hits a "Redundancy Bottleneck" (-16.6%), where PEER and Engram compete for the same information signal. We introduce **differential learning rate training**, which breaks this bottleneck by allowing modules to specialize sequentially, achieving **-33.2% perplexity**—the best result overall.

---

## Introduction

Scaling language models has traditionally required proportional increases in both memory and compute. A model with 10x more parameters requires roughly 10x more FLOPs per forward pass, creating a fundamental barrier to efficient scaling. Recent work on Mixture of Experts (MoE) partially addresses this by activating only a subset of parameters per token, but standard MoE architectures are limited to dozens or hundreds of experts due to routing overhead.

PEER (Parameter Efficient Expert Retrieval) breaks this constraint through product-key retrieval, enabling millions of micro-experts with O(√n) lookup complexity. However, the practical benefits of PEER for language model augmentation have not been thoroughly benchmarked.

We address three key questions:

1. **Compute efficiency**: Does throughput actually remain constant as experts scale?
2. **Quality improvement**: Does training additional PEER capacity improve perplexity?
3. **Complementary architectures**: How does PEER interact with other memory mechanisms?

We combine PEER with Engram (a hash-based n-gram memory module) and mHC (manifold-constrained hyper-connections), creating a *triple-sparse architecture* where:

- **Engram** (early layers): Retrieves static n-gram patterns for phrase-level recognition
- **PEER** (late layers): Routes to specialized micro-experts for contextual reasoning
- **mHC** (throughout): Stabilizes training with multi-stream residual connections

### Key Contributions

1. **Efficiency benchmarks**: We demonstrate 7x parameter scaling (1.1B → 7.68B) with <3% throughput reduction
2. **Training validation**: Trained PEER-262K achieves 14.81 perplexity vs 17.12 baseline (-10.5%)
3. **Triple-sparse architecture**: Combined PEER + Engram + mHC with differential LR achieves 11.05 perplexity (-33.2%)
4. **Redundancy Bottleneck discovery**: We identify and solve the "Sparsity Overlap" problem where joint training underperforms individual components
5. **Differential LR training**: A phased training strategy that breaks the redundancy bottleneck

---

## Method

### Architecture Overview

We augment a pretrained TinyLlama model (1.1B parameters, 22 layers) with three sparse memory modules:

- **PEER layers** at positions 19 and 21 (late layers)
- **Engram layers** at positions 1 and 5 (early layers)
- **mHC layers** at positions 7, 13, and 17 (spread throughout)

This placement follows the intuition that early layers handle pattern recognition while late layers handle reasoning.

### PEER Module

Each PEER layer contains:

- **Product keys**: Two sets of √n keys, where n is the number of experts
- **Expert embeddings**: n pairs of down-projection and up-projection vectors
- **Query projection**: Maps hidden states to query vectors for each key set

The forward pass:

1. Project hidden state to queries
2. Compute similarities to both key sets
3. Select top-k from each set
4. Form Cartesian product and select final top-k experts
5. Retrieve expert weights and compute weighted output

With k=8 experts per head and 4 heads, we retrieve 32 experts per token from pools of up to 803K experts.

### Engram Module

Each Engram layer:

1. Hashes n-grams (2-gram, 3-gram) to indices
2. Looks up embeddings from the hash table
3. Computes gated output using normalized key-query products

The gate uses sqrt-sign activation for stability.

### Critical Finding: Initialization Matters

**New modules must be initialized near-zero** to avoid corrupting pretrained hidden states:

```python
nn.init.normal_(weight, std=0.01)  # projections
nn.init.normal_(embedding, std=0.02)  # embeddings
```

Without this fix, Engram produced 450 perplexity (+2626% worse). With it: 11.30 perplexity (-31.7% better).

---

## Experiments

### Setup

- **Base model**: TinyLlama-1.1B-Chat-v1.0
- **Hardware**: NVIDIA RTX 5070 Ti (15.5 GB VRAM)
- **Dataset**: WikiText-2 (train and test splits)
- **Metrics**: Perplexity, peak VRAM, throughput (tokens/second)

### Efficiency Benchmark

We measured inference efficiency across PEER configurations without training (random initialization):

| Config | Experts | Parameters | VRAM | Throughput | PPL |
|--------|---------|------------|------|------------|-----|
| Baseline | — | 1.10B | 2.07 GB | 26,991 tok/s | 17.12 |
| PEER-16K | 16,384 | 1.24B | 2.35 GB | 26,521 tok/s | 17.12 |
| PEER-65K | 65,536 | 1.64B | 3.10 GB | 26,405 tok/s | 17.12 |
| PEER-262K | 262,144 | 3.25B | 6.10 GB | 26,446 tok/s | 17.12 |
| PEER-590K | 589,824 | 5.93B | 11.10 GB | 26,401 tok/s | 17.12 |
| **PEER-803K** | **802,816** | **7.68B** | **14.35 GB** | **26,262 tok/s** | **17.12** |

**Key finding**: Throughput remains nearly constant (26,262-26,991 tok/s, <3% variance) despite **7x parameter increase**. Memory scales linearly, but compute does not—validating PEER's core efficiency claim.

### Training Results

| Model | Perplexity | Change | Parameters |
|-------|-----------|--------|------------|
| Baseline TinyLlama | 16.54 | — | 1.10B |
| + Engram (trained) | 11.30 | -31.7% | 1.15B |
| + PEER-262K (trained) | 14.81 | -10.5% | 3.25B |
| + mHC (trained) | 12.29 | -25.7% | 1.21B |
| + Engram + PEER (Hybrid) | 11.06 | -33.1% | 3.30B |

---

## The Redundancy Bottleneck

When training PEER + Engram + mHC jointly, we observed a surprising result: the combined model (-16.6%) **underperformed** individual components like Engram (-31.7%) or mHC (-25.7%).

| Configuration | Perplexity | Change |
|--------------|-----------|--------|
| Baseline | 16.54 | — |
| Engram alone | 11.30 | -31.7% |
| mHC alone | 12.29 | -25.7% |
| PEER alone | 14.81 | -10.5% |
| **Triple-Sparse (joint)** | **13.79** | **-16.6%** |

### Root Cause: Sparsity Overlap

Analysis of output magnitudes revealed that Engram dominates (2.62:1 ratio vs PEER), effectively "crowding out" PEER's contribution. Both modules compete for the same information signal—literal pattern matching.

When trained together with equal learning rates:
- Engram learns faster (simpler hash-based mechanism)
- Engram captures patterns that PEER would otherwise specialize in
- PEER becomes redundant, contributing little to the final output

---

## Solution: Differential Learning Rate Training

Instead of freezing modules (which causes distribution shift), we modulate learning rates across phases:

**Phase 1 (Engram-Lead)**: 1,000 batches
- Engram LR = 1e-3 (high)
- PEER LR = 1e-5 (minimal)
- mHC LR = 1e-3 (high)

**Phase 2 (PEER-Catchup)**: 1,000 batches
- PEER LR = 1e-3 (high)
- Engram LR = 1e-5 (minimal)
- mHC LR = 1e-4 (medium)

**Phase 3 (Coordinate)**: 1,000 batches
- All LR = 5e-4 (medium)

### Results

| Strategy | Perplexity | Change | Notes |
|----------|-----------|--------|-------|
| Baseline | 16.54 | — | — |
| Joint training | 13.79 | -16.6% | Redundancy bottleneck |
| Staggered (freeze) | 99.77 | +503% | Distribution shift |
| **Differential LR** | **11.05** | **-33.2%** | **Best result** |

**Key insight**: Never fully freeze modules. Maintaining a minimal learning rate (1e-5) allows modules to adapt to each other while one "leads" and the other "follows."

The staggered approach (completely freezing modules) failed catastrophically because frozen modules can't adapt to the changing output distributions of modules that are still training.

---

## Analysis

### Why Throughput Stays Constant

PEER's constant throughput despite 7x parameter increase stems from:

1. **Sparse retrieval**: Only k=8 experts activated per head regardless of pool size
2. **Product-key efficiency**: O(√n) lookup instead of O(n)
3. **Embedding-based experts**: Simple lookup + matmul, no routing networks

The dominant cost is the base model's attention and MLP computation, which PEER augments rather than replaces.

### Why Training Improves Quality

Untrained PEER maintains baseline perplexity because small initialization (std=0.01) means PEER output ≈ 0 initially—the model effectively ignores untrained contributions.

After training:
- Experts specialize to different input patterns
- Product-key retrieval routes tokens to relevant experts
- The 262K experts provide fine-grained specialization impossible with standard MoE

### Functional Overlap Between Modules

Engram and PEER were designed for different purposes:
- **Engram**: Deterministic n-gram lookup, improves pattern recognition
- **PEER**: Learned expert routing, improves contextual reasoning
- **mHC**: Multi-stream residuals, stabilizes training

However, both PEER and Engram attempt to "retrieve" information based on input—Engram via literal n-gram hashing, PEER via learned query-key matching. When trained jointly, Engram learns faster and captures patterns that PEER would otherwise specialize in.

Differential LR training forces PEER to specialize in patterns Engram *cannot* capture (context-dependent, non-literal patterns), achieving 11.05 PPL—better than either alone.

---

## Complete Results Summary

| Model | Perplexity | Change | Params | Throughput |
|-------|-----------|--------|--------|------------|
| Baseline TinyLlama | 16.54 | — | 1.10B | 26,991 tok/s |
| **Single Components** |||||
| + Engram | 11.30 | -31.7% | 1.15B | ~26K tok/s |
| + PEER-262K | 14.81 | -10.5% | 3.25B | 26,446 tok/s |
| + mHC | 12.29 | -25.7% | 1.21B | ~25K tok/s |
| **Dual Combinations** |||||
| + Engram + PEER | 11.06 | -33.1% | 3.30B | ~26K tok/s |
| **Triple-Sparse** |||||
| Joint training | 13.79 | -16.6% | 3.41B | ~25K tok/s |
| Staggered (freeze) | 99.77 | +503% | 3.41B | — |
| **Differential LR** | **11.05** | **-33.2%** | **3.41B** | ~25K tok/s |

---

## Implications for Scaling

Our results suggest a new scaling paradigm:

1. Train a moderate base model (1-7B parameters)
2. Add PEER layers with millions of experts
3. Train only PEER parameters (faster, less memory)
4. Result: Quality of larger model, inference cost of smaller model

---

## Limitations

1. **Single evaluation dataset**: WikiText-2 is n-gram heavy, which may favor Engram
2. **Small base model**: TinyLlama (1.1B) may not represent larger model behavior
3. **No downstream task evaluation**: Only perplexity measured, not task performance
4. **Differential LR sensitivity**: Optimal phase durations and LR ratios may vary across architectures
5. **Limited training budget**: Longer training may yield further improvements

---

## Conclusion

We present a triple-sparse architecture combining PEER, Engram, and mHC for efficient language model scaling. Our benchmarks demonstrate that PEER enables **7x parameter scaling with constant throughput**, and that training these parameters yields meaningful quality improvements.

A key discovery is the **Redundancy Bottleneck**: joint training of multiple sparse modules can underperform individual components due to functional overlap. We solve this with **differential learning rate training**, which allows modules to specialize sequentially while maintaining coordination. This achieves **11.05 perplexity (-33.2%)**—the best result across all configurations.

### Key Takeaways

1. **Sparse modules compete**: Different sparse retrieval mechanisms can interfere when trained jointly
2. **Training strategy matters**: The same architecture yields vastly different results (99.77 vs 11.05 PPL) depending on training approach
3. **Differential LR enables coordination**: Modulating learning rates forces specialization while maintaining adaptability

These results validate both PEER's efficiency claims and highlight the importance of training strategies for multi-component architectures.

---

## Code Availability

Code and trained weights are available at: https://github.com/MikeyBeez/PEER

---

## References

- PEER: [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)
- Engram: [Byte-Level Memory-Augmented Language Modeling](https://github.com/deepseek-ai/Engram)
- mHC: [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
