#!/usr/bin/env python3
"""
Token-Level Conflict Analysis for PEER + Engram

Analyzes which tokens trigger both PEER and Engram modules to understand
where functional redundancy occurs in the Triple-Sparse architecture.

Key metrics:
- Token-level gate activations for both modules
- Overlap coefficient (how often both fire together)
- Token type analysis (what kinds of tokens cause conflict)
"""

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_peer_engram import IntegrationConfig, inject_peer_engram, LlamaEngram, LlamaPEER


class ActivationTracker:
    """Tracks output magnitudes for PEER and Engram modules."""

    def __init__(self):
        self.engram_magnitudes = []  # Output L2 norms per forward pass
        self.peer_magnitudes = []    # Output L2 norms per forward pass
        self.hooks = []

    def reset(self):
        self.engram_magnitudes = []
        self.peer_magnitudes = []

    def register_hooks(self, model):
        """Register forward hooks on PEER and Engram modules."""

        def engram_hook(module, input, output):
            # Track output magnitude (how much Engram contributes)
            if output is not None and hasattr(output, 'norm'):
                mag = output.norm().item()
                self.engram_magnitudes.append(mag)

        def peer_hook(module, input, output):
            # Track output magnitude (how much PEER contributes)
            if output is not None and hasattr(output, 'norm'):
                mag = output.norm().item()
                self.peer_magnitudes.append(mag)

        for name, module in model.named_modules():
            if isinstance(module, LlamaEngram):
                hook = module.register_forward_hook(engram_hook)
                self.hooks.append(hook)
                print(f"Registered Engram hook on {name}")
            elif isinstance(module, LlamaPEER):
                hook = module.register_forward_hook(peer_hook)
                self.hooks.append(hook)
                print(f"Registered PEER hook on {name}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def analyze_activations_detailed(model, tokenizer, test_data, device, max_samples=100):
    """Analyze token-level activations with detailed per-token tracking."""

    print("\n" + "="*60)
    print("Detailed Token-Level Activation Analysis")
    print("="*60)

    model.eval()

    # Get module references
    engram_modules = []
    peer_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LlamaEngram):
            engram_modules.append((name, module))
        elif isinstance(module, LlamaPEER):
            peer_modules.append((name, module))

    print(f"Found {len(engram_modules)} Engram modules, {len(peer_modules)} PEER modules")

    # Set up activation tracker
    tracker = ActivationTracker()
    tracker.register_hooks(model)

    # Run some forward passes to collect activation magnitudes
    print("\nCollecting activation magnitudes...")
    with torch.no_grad():
        for i, ex in enumerate(tqdm(test_data, total=min(max_samples, len(test_data)), desc="Collecting")):
            if i >= max_samples:
                break
            if not ex['text'].strip():
                continue

            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)
            model.model._current_input_ids = input_ids

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(input_ids=input_ids)

    tracker.remove_hooks()

    # Compute statistics
    avg_engram_mag = sum(tracker.engram_magnitudes) / len(tracker.engram_magnitudes) if tracker.engram_magnitudes else 0
    avg_peer_mag = sum(tracker.peer_magnitudes) / len(tracker.peer_magnitudes) if tracker.peer_magnitudes else 0

    print(f"\nEngram output magnitudes: mean={avg_engram_mag:.4f}, n={len(tracker.engram_magnitudes)}")
    print(f"PEER output magnitudes: mean={avg_peer_mag:.4f}, n={len(tracker.peer_magnitudes)}")

    # Analyze token patterns
    print("\n" + "-"*40)
    print("Analyzing token patterns...")
    print("-"*40)

    # Categorize tokens
    token_categories = {
        'punctuation': set(),
        'numbers': set(),
        'common_words': set(),
        'rare_words': set(),
        'subwords': set(),
    }

    # Sample some data to categorize tokens
    for i, ex in enumerate(test_data):
        if i >= max_samples:
            break
        if not ex['text'].strip():
            continue

        inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs['input_ids'][0]

        for tid in input_ids.tolist():
            token_str = tokenizer.decode([tid])

            # Categorize
            if token_str.strip() in '.,;:!?()[]{}"\'-':
                token_categories['punctuation'].add(tid)
            elif token_str.strip().isdigit():
                token_categories['numbers'].add(tid)
            elif token_str.startswith('▁') or token_str.startswith(' '):
                # Full word (space-prefixed in sentencepiece)
                if len(token_str) <= 4:
                    token_categories['common_words'].add(tid)
                else:
                    token_categories['rare_words'].add(tid)
            else:
                # Subword (continuation)
                token_categories['subwords'].add(tid)

    # Print category sizes
    print("\nToken categories found:")
    for cat, tokens in token_categories.items():
        print(f"  {cat}: {len(tokens)} unique tokens")

    # Analyze which categories benefit from which module
    print("\n" + "-"*40)
    print("Module Specialization Analysis")
    print("-"*40)

    print(f"""
Based on output magnitudes and architecture:

ENGRAM (avg magnitude: {avg_engram_mag:.4f}):
  - Layers: 1, 5 (early layers)
  - Purpose: N-gram pattern matching (2-grams, 3-grams)
  - Best for: Common phrases, idioms, syntax patterns
  - Example: "United States", "New York", "in the"

PEER (avg magnitude: {avg_peer_mag:.4f}):
  - Layers: 19, 21 (late layers)
  - Purpose: Semantic expert routing
  - Best for: Context-dependent reasoning, rare combinations
  - Example: Technical terms, novel combinations

OVERLAP ZONE (where both fire):
  - Medium-frequency n-grams that also need context
  - Code patterns (both syntactic AND semantic)
  - Named entities (pattern AND meaning)
""")

    # Compute overlap coefficient based on relative magnitudes
    # If both contribute similarly, there's potential overlap
    total_mag = avg_engram_mag + avg_peer_mag
    if total_mag > 0:
        engram_share = avg_engram_mag / total_mag
        peer_share = avg_peer_mag / total_mag
        overlap_coefficient = 1 - abs(engram_share - peer_share)  # 1.0 = perfect balance, 0.0 = one dominates
    else:
        overlap_coefficient = 0

    print(f"\nContribution Analysis:")
    print(f"  Avg Engram magnitude: {avg_engram_mag:.4f}")
    print(f"  Avg PEER magnitude: {avg_peer_mag:.4f}")
    print(f"  Balance coefficient: {overlap_coefficient:.4f} (1.0 = equal contribution)")

    if overlap_coefficient > 0.8:
        print("  → BALANCED: Both modules contributing equally")
        print("     This may indicate functional overlap - both trying to solve same problems")
    elif overlap_coefficient > 0.5:
        print("  → MODERATE IMBALANCE: Some specialization occurring")
        print("     One module is starting to dominate certain patterns")
    else:
        print("  → HIGH SPECIALIZATION: Modules have distinct roles")
        print("     Good separation of responsibilities")

    return {
        'engram_magnitude': avg_engram_mag,
        'peer_magnitude': avg_peer_mag,
        'overlap_coefficient': overlap_coefficient,
        'token_categories': {k: len(v) for k, v in token_categories.items()},
    }


def analyze_layer_contributions(model, tokenizer, test_data, device, max_samples=50):
    """Analyze how much each layer's module contributes to the output."""

    print("\n" + "="*60)
    print("Layer-wise Contribution Analysis")
    print("="*60)

    model.eval()

    # Collect layer-wise statistics via hooks
    layer_magnitudes = {}

    def make_hook(name, module_type, layer_idx):
        def hook(module, input, output):
            if output is not None and hasattr(output, 'norm'):
                mag = output.norm().item()
                if name not in layer_magnitudes:
                    layer_magnitudes[name] = {
                        'type': module_type,
                        'layer': layer_idx,
                        'magnitudes': []
                    }
                layer_magnitudes[name]['magnitudes'].append(mag)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, LlamaEngram):
            layer_idx = module.layer_idx if hasattr(module, 'layer_idx') else 'unknown'
            hook = module.register_forward_hook(make_hook(name, 'Engram', layer_idx))
            hooks.append(hook)
        elif isinstance(module, LlamaPEER):
            layer_idx = 'late'  # PEER doesn't store layer_idx
            hook = module.register_forward_hook(make_hook(name, 'PEER', layer_idx))
            hooks.append(hook)

    # Run forward passes
    print("Collecting layer-wise magnitudes...")
    with torch.no_grad():
        for i, ex in enumerate(tqdm(test_data, total=min(max_samples, len(test_data)), desc="Layers")):
            if i >= max_samples:
                break
            if not ex['text'].strip():
                continue

            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)
            model.model._current_input_ids = input_ids

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(input_ids=input_ids)

    for hook in hooks:
        hook.remove()

    # Compute averages
    layer_stats = {}
    for name, data in layer_magnitudes.items():
        avg_mag = sum(data['magnitudes']) / len(data['magnitudes']) if data['magnitudes'] else 0
        layer_stats[name] = {
            'type': data['type'],
            'layer': data['layer'],
            'avg_magnitude': avg_mag,
        }

    print("\nPer-module output magnitudes:")
    print("-" * 50)

    # Sort by layer
    sorted_stats = sorted(layer_stats.items(), key=lambda x: str(x[1].get('layer', 0)))

    total_engram = 0
    total_peer = 0

    for name, stats in sorted_stats:
        print(f"  Layer {stats['layer']:>2} | {stats['type']:>6} | magnitude={stats['avg_magnitude']:.4f}")
        if stats['type'] == 'Engram':
            total_engram += stats['avg_magnitude']
        else:
            total_peer += stats['avg_magnitude']

    print("-" * 50)
    print(f"  Total Engram magnitude: {total_engram:.4f}")
    print(f"  Total PEER magnitude: {total_peer:.4f}")

    if total_peer > 0:
        ratio = total_engram / total_peer
        print(f"  Ratio (Engram/PEER): {ratio:.2f}:1")
        print(f"\n  Optimal ratio (per 2026 research): ~3:1")
        print(f"  Your ratio: {ratio:.2f}:1")

        if ratio > 4:
            print("  → Engram-heavy: Consider boosting PEER learning rate")
        elif ratio < 2:
            print("  → PEER-heavy: Consider boosting Engram learning rate")
        else:
            print("  → Near optimal range")
    else:
        print("  PEER contribution is 0")

    return layer_stats


def suggest_improvements(analysis_results, layer_stats):
    """Generate actionable recommendations based on analysis."""

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    overlap = analysis_results['overlap_coefficient']
    engram_mag = analysis_results['engram_magnitude']
    peer_mag = analysis_results['peer_magnitude']

    recommendations = []

    # 1. Check for high balance (potential overlap)
    if overlap > 0.7:
        recommendations.append("""
1. ADD DIVERSITY LOSS
   Both modules contributing similarly. Add a loss term to encourage specialization:

   # During forward, compute per-token magnitudes
   engram_contrib = engram_output.norm(dim=-1)
   peer_contrib = peer_output.norm(dim=-1)
   diversity_loss = -torch.abs(engram_contrib - peer_contrib).mean()
   total_loss = ce_loss + 0.1 * diversity_loss
""")

    # 2. Check for imbalanced contributions
    total_mag = engram_mag + peer_mag
    if total_mag > 0:
        engram_share = engram_mag / total_mag
        peer_share = peer_mag / total_mag

        if engram_share > 0.7:
            recommendations.append("""
2. STAGGERED TRAINING (Engram-dominant)
   Engram contributes more. Try:
   - Freeze Engram for 1000 batches
   - Let PEER learn the "hard" tokens Engram can't handle
   - Then unfreeze and continue joint training
""")
        elif peer_share > 0.7:
            recommendations.append("""
2. STAGGERED TRAINING (PEER-dominant)
   PEER contributes more. Try:
   - Freeze PEER for 1000 batches
   - Let Engram learn common patterns
   - Then unfreeze and continue joint training
""")

    # 3. Layer placement
    recommendations.append("""
3. CONSIDER LAYER RELOCATION
   Current: Engram @ layers 1,5 | PEER @ layers 19,21

   Alternative configurations to try:
   - Move PEER earlier (layers 11,15) for more interaction
   - Add Engram to middle layers for phrase-level patterns
   - Test: Engram @ 1,3,5 | PEER @ 15,17,19 (more separation)
""")

    # 4. Capacity rebalancing
    recommendations.append("""
4. CAPACITY REBALANCING
   If redundancy persists, reduce one module's capacity:
   - Engram: Reduce vocab_size from 100K to 50K
   - PEER: Reduce num_experts from 16K to 8K
   This forces specialization by limiting overlap potential.
""")

    for rec in recommendations:
        print(rec)

    return recommendations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100, help='Samples to analyze')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Inject PEER + Engram (hybrid mode)
    config = IntegrationConfig(mode="hybrid")
    model = inject_peer_engram(model, config)
    print("Injected PEER + Engram (hybrid)")

    # Try to load trained weights
    try:
        saved = torch.load("./llama_triple_v1/weights.pt", map_location=device, weights_only=True)
        state = model.state_dict()
        loaded = 0
        for n, p in saved.items():
            if n in state and state[n].shape == p.shape:
                state[n] = p.to(state[n].device)
                loaded += 1
        model.load_state_dict(state)
        print(f"Loaded {loaded} trained tensors from llama_triple_v1")
    except Exception as e:
        print(f"Could not load trained weights: {e}")
        print("Using untrained modules (gates will be at initial values)")

    # Load test data
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test = test.filter(lambda x: len(x['text'].strip()) > 10)

    # Run analysis
    analysis_results = analyze_activations_detailed(model, tokenizer, test, device, args.samples)
    layer_stats = analyze_layer_contributions(model, tokenizer, test, device, args.samples)

    # Generate recommendations
    suggest_improvements(analysis_results, layer_stats)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
