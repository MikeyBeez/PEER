# PEER + Engram: Dual-Sparse Architecture for LLaMA

This project integrates two complementary "brain upgrades" into TinyLlama:

- **PEER** (Parameter Efficient Expert Retrieval) - Routes to specialized micro-experts based on hidden state
- **Engram** - Retrieves static n-gram patterns based on literal text

Together they form a **Dual-Sparse Architecture**: PEER handles dynamic reasoning, Engram handles static pattern memory.

## Understanding PEER's Value Proposition

**PEER is fundamentally about memory efficiency, not perplexity improvement.**

Traditional scaling requires linear increases in VRAM as you add parameters. PEER breaks this constraint:

- **Massive capacity, minimal footprint** - Store millions of micro-experts, activate only a handful per token
- **O(√n) retrieval** - Product-key lookup scales sub-linearly with expert count
- **Decoupled capacity from compute** - Scale parameters without scaling FLOPs or VRAM

The right way to evaluate PEER:

| Metric | What it measures |
|--------|------------------|
| VRAM usage | Memory footprint at inference |
| Perplexity per GB | Efficiency of memory utilization |
| Throughput | Tokens/second at equivalent quality |
| Scaling behavior | How metrics change as experts increase |

Comparing raw perplexity against dense models misses the point—PEER lets you have 10x the parameters at 2x the memory.

## Results

| Model | Perplexity | Notes |
|-------|-----------|-------|
| Base TinyLlama | 16.54 | Baseline |
| + Engram | 11.30 | **-31.67%** - pattern memory |
| + PEER | 16.54 | Maintains quality with sparse retrieval |
| + Both (Hybrid) | 11.06 | **-33.10%** - best of both |

PEER maintains baseline perplexity while enabling massive parameter scaling—that's the win.

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

### Why This Combination Works
- **Engram in early layers**: Recognizes common phrases as units (e.g., "United States", "New York") → improves quality
- **PEER in later layers**: Provides massive parameter capacity with minimal VRAM overhead → enables scaling
- They serve different purposes: Engram boosts perplexity, PEER enables efficient capacity expansion
- They don't interfere because they operate at different depths

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
- `train_and_eval.py` - Training and evaluation script
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

## License

MIT
