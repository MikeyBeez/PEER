# Dual-Sparse Architecture: How We Got 7x More Parameters Without Slowing Down

## The Problem With Scaling Language Models

Here's the fundamental problem with making language models bigger: if you want 10x more parameters, you need roughly 10x more compute per forward pass. Memory and speed scale together, creating an expensive barrier to improvement.

Mixture of Experts (MoE) partially solved this by only activating some parameters per token. But traditional MoE is limited to dozens or hundreds of experts because routing becomes expensive. What if you could have *millions* of experts?

That's exactly what PEER (Parameter Efficient Expert Retrieval) promises. And we decided to test whether it actually delivers.

## What We Built

We integrated two complementary "brain upgrades" into TinyLlama, a 1.1 billion parameter model:

**PEER** uses product-key retrieval to select from hundreds of thousands of micro-experts. Instead of routing through all experts (which would be slow), it uses a clever trick: split each expert's "address" into two parts, search each part separately, then combine. This gives you O(√n) lookup instead of O(n). With 800,000 experts, that's checking 1,800 keys instead of 800,000.

**Engram** is simpler—it hashes n-grams (2-word and 3-word sequences) and looks up learned embeddings. Think of it as pattern memory for common phrases like "United States" or "New York."

We put Engram in early layers (for pattern recognition) and PEER in late layers (for reasoning). They don't interfere because they operate at different depths.

## The Efficiency Results

We ran PEER configurations from 16,000 experts up to 803,000 experts on an RTX 5070 Ti with 15.5 GB of VRAM. The results were striking.

The baseline TinyLlama uses 2.07 GB of VRAM and processes about 27,000 tokens per second.

With PEER-16K (16,384 experts), we added 12% more parameters. VRAM went up slightly to 2.35 GB. Throughput: 26,521 tokens per second. Essentially unchanged.

With PEER-65K, we had 49% more parameters. VRAM: 3.10 GB. Throughput: 26,405 tokens per second. Still basically the same.

With PEER-262K, we hit 3x the parameters (3.25 billion total). VRAM: 6.10 GB. Throughput: 26,446 tokens per second.

With PEER-803K, we reached 7x the parameters (7.68 billion total). VRAM: 14.35 GB (nearly maxing out the GPU). Throughput: 26,262 tokens per second.

**The key finding: 7x more parameters with less than 3% throughput reduction.**

Memory scales linearly—that's expected, since you're storing more expert embeddings. But compute stays flat. That's the whole point of PEER, and it actually works.

## But Does It Actually Help?

Here's the thing about those efficiency numbers: the perplexity was identical across all configurations (17.12). That's because we tested with randomly initialized PEER modules. The model was essentially ignoring them.

The real question is: if you train those experts, does the extra capacity actually improve the model?

We trained PEER-262K for 3,000 batches on WikiText-2. The result: perplexity dropped from 17.12 to 14.81. That's a 13.5% improvement.

This is significant. The additional expert capacity isn't just sitting there—it's being utilized. The model learns to route different inputs to different specialized experts, and that specialization helps.

## The Full Picture

Here's how all the pieces compare after training:

Baseline TinyLlama achieves 17.12 perplexity.

Adding trained Engram drops it to 11.30—a 34% improvement. Engram is remarkably effective for pattern-heavy text like Wikipedia.

Adding trained PEER-262K (without Engram) drops it to 14.81—a 13.5% improvement. Not as dramatic as Engram, but remember: this is with 3x the parameters at the same inference speed.

Combining both gives 11.06 perplexity—a 35.4% improvement. The hybrid is slightly better than Engram alone, suggesting the approaches are complementary.

## Why This Matters

The implications for scaling are interesting. Traditional wisdom says: want a better model? Make it bigger. But bigger means slower and more expensive to run.

PEER suggests an alternative: keep your base model moderate (say, 1-7 billion parameters), then add millions of micro-experts. Train only the expert parameters (which is faster and uses less memory than full model training). The result is quality approaching a larger model with inference costs of the smaller one.

This is particularly relevant for deployment. Inference cost is often the bottleneck—you're running the model millions of times. If you can get 7B-model quality at 1B-model speeds, that's a significant win.

## The Critical Implementation Detail

One thing nearly derailed our experiments: initialization.

When you add new modules to a pretrained model, the default PyTorch initialization produces high-variance random outputs. These random outputs corrupt the carefully tuned hidden states that the pretrained model expects.

The fix is simple but crucial: initialize new modules to output near-zero values at the start. We used standard deviation of 0.01 for projection weights and 0.02 for embeddings.

Without this fix, Engram produced 450 perplexity—catastrophically worse than baseline. With the fix: 11.30 perplexity, dramatically better. Same architecture, same training, just different initialization.

If you're adding modules to pretrained models, remember: start small. Let the model learn to use them gradually rather than forcing it to immediately handle random noise.

## Limitations

We should be honest about what we haven't proven:

**Memory still scales linearly.** PEER saves compute, not memory. With 803K experts, we used 14.35 GB just for weights, leaving little room for training. Practical deployment of million-expert models will need techniques like CPU offloading or quantization.

**We only tested on WikiText-2.** This is a standard benchmark, but it's n-gram heavy, which favors Engram. Broader evaluation on diverse tasks (reasoning, coding, math) would strengthen the conclusions.

**TinyLlama is small.** Behavior at 1.1B parameters might not generalize to 70B or 400B. Though the efficiency principles should hold, the quality improvements might differ.

**Limited training.** 3,000 batches might not fully utilize 262,000 expert capacity. Longer training or larger expert counts might show more dramatic improvements.

## What's Next

Several directions seem promising:

Scaling to 1M+ experts with memory optimization techniques. Our GPU maxed out at 803K, but with CPU offloading or 8-bit quantization, millions should be feasible.

Evaluating on diverse benchmarks. Does PEER help with reasoning tasks? Code generation? Following instructions?

Comparing training efficiency. How does "train only PEER parameters" compare to full model training in terms of quality per FLOP?

Optimizing layer placement. We put PEER in layers 19 and 21 based on intuition. Systematic ablation might find better configurations.

## Conclusion

We set out to test whether PEER's efficiency claims hold in practice. They do. Seven times more parameters, essentially the same throughput.

More importantly, we showed that training those additional parameters improves quality. PEER-262K achieved 13.5% better perplexity than baseline. Combined with Engram, we hit 35.4% improvement.

The core insight is simple: **decouple parameter capacity from compute cost.** Store millions of tiny experts. Retrieve only a few per token. Let the routing learn which experts matter for which inputs.

As models scale and inference costs dominate, this decoupling becomes increasingly valuable. You don't have to choose between quality and speed—with the right architecture, you can have both.

---

*Code and trained weights available at: github.com/MikeyBeez/PEER*

*This work builds on PEER (arXiv:2407.04153) and Engram (github.com/deepseek-ai/Engram).*
