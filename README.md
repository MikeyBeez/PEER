# PEER + Engram: Dual-Sparse Architecture for LLaMA

This project integrates two complementary "brain upgrades" into TinyLlama:

- **PEER** (Parameter Efficient Expert Retrieval) - Routes to specialized micro-experts based on hidden state
- **Engram** - Retrieves static n-gram patterns based on literal text

Together they form a **Dual-Sparse Architecture**: PEER handles dynamic reasoning, Engram handles static pattern memory.

## Results

| Model | Perplexity | Change |
|-------|-----------|--------|
| Base TinyLlama | 16.54 | — |
| + Engram | 11.30 | **-31.67%** |
| + PEER | 16.54 | 0.00% |
| + Both (Hybrid) | 11.06 | **-33.10%** |

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
- **Engram in early layers**: Recognizes common phrases as units (e.g., "United States", "New York")
- **PEER in later layers**: Routes to specialized reasoning based on context
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

- PEER: [Parameter Efficient Expert Retrieval](https://arxiv.org/abs/2304.01665)
- Engram: Memory-augmented language modeling via n-gram hashing

## License

MIT
