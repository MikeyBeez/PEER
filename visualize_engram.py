#!/usr/bin/env python3
"""
Visualize which n-grams the Engram module is prioritizing.

This script:
1. Loads the trained Engram model
2. Runs text through it while capturing gate activations
3. Maps high-activation positions back to their n-gram text
4. Displays which patterns the model learned to use
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_peer_engram import IntegrationConfig, inject_peer_engram, LlamaEngram
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def get_ngram_text(tokens: list, position: int, n: int) -> str:
    """Extract the n-gram ending at position."""
    start = max(0, position - n + 1)
    return " ".join(tokens[start:position + 1])


class EngramProbe:
    """Hooks into Engram layers to capture gate activations."""

    def __init__(self, model):
        self.model = model
        self.gate_values = {}
        self.hooks = []
        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on Engram layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, LlamaEngram):
                layer_idx = module.layer_idx
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)
                print(f"Installed probe on Engram layer {layer_idx}")

    def _make_hook(self, layer_idx):
        """Create a hook that captures gate values."""
        def hook(module, input, output):
            # We need to capture the gate values during forward pass
            # Re-compute them here (slight overhead but cleaner)
            hidden_states = input[0]
            input_ids = input[1] if len(input) > 1 else None

            if input_ids is None:
                # Try to get from parent model
                if module._parent_model_ref is not None:
                    parent = module._parent_model_ref()
                    if parent is not None:
                        input_ids = getattr(parent, '_current_input_ids', None)

            if input_ids is not None:
                B, T, D = hidden_states.shape

                # Compute gate values (same as in forward)
                hash_indices = module.hash_ngrams(input_ids)
                embeds = []
                for i, idx in enumerate(hash_indices):
                    embeds.append(module.embeddings[i](idx))
                embeddings = torch.cat(embeds, dim=-1)

                key = module.key_proj(embeddings)
                query = hidden_states

                import math
                gate = (module.norm_k(key) * module.norm_q(query)).sum(dim=-1, keepdim=True)
                gate = gate / math.sqrt(D)
                gate_abs = gate.abs().clamp(min=1e-8, max=100.0)
                gate = gate_abs.sqrt() * gate.sign()
                gate = torch.sigmoid(gate.clamp(-10, 10))

                # Store gate values [B, T, 1] -> [B, T]
                self.gate_values[layer_idx] = gate.squeeze(-1).detach().cpu()

        return hook

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear captured values."""
        self.gate_values = {}


def visualize_gates(text: str, model, tokenizer, probe, save_path: str = None):
    """Visualize gate activations for a piece of text."""

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(next(model.parameters()).device)

    # Store input_ids for Engram access
    model.model._current_input_ids = input_ids

    # Clear previous captures
    probe.clear()

    # Forward pass
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(input_ids=input_ids)

    # Get tokens as strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Create visualization
    num_layers = len(probe.gate_values)
    if num_layers == 0:
        print("No gate values captured!")
        return

    fig, axes = plt.subplots(num_layers + 1, 1, figsize=(14, 3 * (num_layers + 1)))
    if num_layers == 1:
        axes = [axes, plt.subplot(num_layers + 1, 1, 2)]

    layer_indices = sorted(probe.gate_values.keys())

    # Plot each layer's gates
    for i, layer_idx in enumerate(layer_indices):
        gates = probe.gate_values[layer_idx][0].numpy()  # [T]

        ax = axes[i]
        bars = ax.bar(range(len(gates)), gates, color='steelblue', alpha=0.7)

        # Highlight high activations
        threshold = np.percentile(gates, 80)
        for j, (bar, g) in enumerate(zip(bars, gates)):
            if g > threshold:
                bar.set_color('coral')

        ax.set_ylabel(f'Layer {layer_idx}\nGate Value')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Engram Gate Activations - Layer {layer_idx}')

    # Combined view (average across layers)
    all_gates = np.stack([probe.gate_values[idx][0].numpy() for idx in layer_indices])
    avg_gates = all_gates.mean(axis=0)

    ax = axes[-1]
    bars = ax.bar(range(len(avg_gates)), avg_gates, color='green', alpha=0.7)
    threshold = np.percentile(avg_gates, 80)
    for j, (bar, g) in enumerate(zip(bars, avg_gates)):
        if g > threshold:
            bar.set_color('darkgreen')

    ax.set_ylabel('Average\nGate Value')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_title('Average Engram Activation Across Layers')
    ax.set_xlabel('Token Position')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    return tokens, probe.gate_values


def analyze_ngram_patterns(texts: list, model, tokenizer, probe, top_k: int = 20):
    """Analyze which n-gram patterns consistently get high activation."""

    ngram_activations = defaultdict(list)  # ngram_text -> list of gate values

    device = next(model.parameters()).device

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)

        model.model._current_input_ids = input_ids
        probe.clear()

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(input_ids=input_ids)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Average gates across layers
        if not probe.gate_values:
            continue

        layer_indices = sorted(probe.gate_values.keys())
        all_gates = np.stack([probe.gate_values[idx][0].numpy() for idx in layer_indices])
        avg_gates = all_gates.mean(axis=0)

        # Record n-gram activations
        for pos in range(len(tokens)):
            for n in [2, 3]:  # 2-grams and 3-grams
                if pos >= n - 1:
                    ngram = get_ngram_text(tokens, pos, n)
                    ngram_activations[ngram].append(avg_gates[pos])

    # Compute statistics
    ngram_stats = []
    for ngram, activations in ngram_activations.items():
        if len(activations) >= 2:  # Only patterns seen multiple times
            ngram_stats.append({
                'ngram': ngram,
                'mean_activation': np.mean(activations),
                'std_activation': np.std(activations),
                'count': len(activations)
            })

    # Sort by mean activation
    ngram_stats.sort(key=lambda x: x['mean_activation'], reverse=True)

    print("\n" + "=" * 60)
    print("TOP N-GRAM PATTERNS BY ENGRAM ACTIVATION")
    print("=" * 60)
    print(f"{'N-gram':<30} {'Mean Gate':>10} {'Std':>8} {'Count':>6}")
    print("-" * 60)

    for stat in ngram_stats[:top_k]:
        print(f"{stat['ngram']:<30} {stat['mean_activation']:>10.4f} {stat['std_activation']:>8.4f} {stat['count']:>6}")

    print("\n" + "=" * 60)
    print("LOWEST N-GRAM PATTERNS (Engram ignores these)")
    print("=" * 60)

    for stat in ngram_stats[-top_k:]:
        print(f"{stat['ngram']:<30} {stat['mean_activation']:>10.4f} {stat['std_activation']:>8.4f} {stat['count']:>6}")

    return ngram_stats


def main():
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = IntegrationConfig(mode="engram")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Inject Engram
    model = inject_peer_engram(model, config)

    # Load trained weights
    print("\nLoading trained Engram weights...")
    saved = torch.load("./llama_engram_v2/weights.pt", map_location=device, weights_only=True)
    state = model.state_dict()
    loaded = 0
    for n, p in saved.items():
        if n in state and state[n].shape == p.shape:
            state[n] = p.to(state[n].device)
            loaded += 1
    model.load_state_dict(state)
    print(f"Loaded {loaded} tensors")

    model.eval()

    # Install probes
    probe = EngramProbe(model)

    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and chaos.",
        "New York City is the largest city in the United States.",
        "The president announced a new policy on climate change.",
        "Machine learning models require large amounts of training data.",
        "Once upon a time, in a land far away, there lived a princess.",
        "The stock market crashed dramatically on Black Tuesday.",
        "Scientists have discovered a new species of deep-sea fish.",
    ]

    print("\n" + "=" * 60)
    print("VISUALIZING GATE ACTIVATIONS")
    print("=" * 60)

    # Visualize first text
    print(f"\nText: {test_texts[0]}")
    visualize_gates(test_texts[0], model, tokenizer, probe, save_path="engram_gates_1.png")

    print(f"\nText: {test_texts[2]}")
    visualize_gates(test_texts[2], model, tokenizer, probe, save_path="engram_gates_2.png")

    # Analyze patterns across multiple texts
    print("\nAnalyzing n-gram patterns across test texts...")

    # Load more diverse text for analysis
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    analysis_texts = [ex['text'] for ex in dataset if len(ex['text'].strip()) > 50][:100]

    ngram_stats = analyze_ngram_patterns(analysis_texts, model, tokenizer, probe)

    # Cleanup
    probe.remove_hooks()

    print("\nDone! Check engram_gates_1.png and engram_gates_2.png for visualizations.")


if __name__ == "__main__":
    main()
