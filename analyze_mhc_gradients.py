#!/usr/bin/env python3
"""
Analyze mHC's mechanism: Does it act as gradient bandwidth for sparse modules?

Hypothesis: mHC's multi-stream residuals provide richer gradient paths,
allowing sparse modules (PEER, Engram) to receive stronger training signals.

Tests:
1. Compare gradient magnitudes at sparse modules with/without mHC
2. Track gradient flow through mHC streams
3. Measure effective gradient bandwidth
"""

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_peer_engram import IntegrationConfig, inject_peer_engram, LlamaEngram, LlamaPEER
from llama_mhc import mHC, StreamExpander, StreamReducer


class mHCLayer(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int = 4, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.expander = StreamExpander(hidden_size, num_streams)
        self.reducer = StreamReducer(hidden_size, num_streams)
        self.mhc = mHC(hidden_size, num_streams)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.001)
        self.gate_scale = nn.Parameter(torch.tensor(-4.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        squeeze_output = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        x_expanded = self.expander(hidden_states)
        x_mhc = self.mhc(x_expanded)
        x_reduced = self.reducer(x_mhc)
        mhc_out = self.out_proj(x_reduced)
        gate = torch.sigmoid(self.gate_scale)
        output = hidden_states + gate * mhc_out
        if squeeze_output:
            output = output.squeeze(0)
        return output


class mHCWrapper(nn.Module):
    def __init__(self, original_layer, hidden_size: int, num_streams: int = 4, layer_idx: int = 0):
        super().__init__()
        self.original_layer = original_layer
        self.mhc_layer = mHCLayer(hidden_size, num_streams, layer_idx)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        original_outputs = self.original_layer(hidden_states, **kwargs)
        if isinstance(original_outputs, tuple):
            layer_output = original_outputs[0]
            extra = original_outputs[1:]
        else:
            layer_output = original_outputs
            extra = None
        enhanced_output = self.mhc_layer(layer_output)
        if extra is not None:
            return (enhanced_output,) + extra
        return enhanced_output


def inject_mhc(model, mhc_layers=None, num_streams=4):
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if mhc_layers is None:
        mhc_layers = [7, 13, 17]
    layers = model.model.layers
    for idx in mhc_layers:
        if idx < num_layers:
            original_layer = layers[idx]
            device = next(original_layer.parameters()).device
            dtype = next(original_layer.parameters()).dtype
            wrapped = mHCWrapper(original_layer, hidden_size, num_streams, idx)
            wrapped.mhc_layer = wrapped.mhc_layer.to(device=device, dtype=dtype)
            layers[idx] = wrapped
    return model


class GradientTracker:
    """Track gradient magnitudes for specific module types."""
    
    def __init__(self):
        self.gradients = defaultdict(list)
        self.hooks = []
    
    def register_hooks(self, model):
        """Register backward hooks on target modules."""
        
        def make_hook(name, module_type):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].norm().item()
                    self.gradients[f"{module_type}_{name}"].append(grad_norm)
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, LlamaEngram):
                h = module.register_full_backward_hook(make_hook(name, "Engram"))
                self.hooks.append(h)
            elif isinstance(module, LlamaPEER):
                h = module.register_full_backward_hook(make_hook(name, "PEER"))
                self.hooks.append(h)
            elif isinstance(module, mHCLayer):
                h = module.register_full_backward_hook(make_hook(name, "mHC"))
                self.hooks.append(h)
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_stats(self):
        stats = {}
        for name, grads in self.gradients.items():
            if grads:
                stats[name] = {
                    'mean': sum(grads) / len(grads),
                    'max': max(grads),
                    'min': min(grads),
                    'count': len(grads),
                }
        return stats


def run_gradient_analysis(model, tokenizer, dataset, device, num_batches=50, desc=""):
    """Run forward/backward passes and collect gradient statistics."""
    
    tracker = GradientTracker()
    tracker.register_hooks(model)
    
    model.train()
    
    for i, ex in enumerate(tqdm(dataset, total=num_batches, desc=desc)):
        if i >= num_batches:
            break
        
        text = ex.get('text', '')
        if not text or len(text.strip()) < 20:
            continue
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        if inputs['input_ids'].shape[1] < 2:
            continue
        
        input_ids = inputs['input_ids'].to(device)
        
        if hasattr(model.model, '_current_input_ids'):
            model.model._current_input_ids = input_ids
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
        
        loss.backward()
        model.zero_grad()
    
    tracker.remove_hooks()
    return tracker.get_stats()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=50, help='Batches to analyze')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\n{'='*60}")
    print("mHC GRADIENT BANDWIDTH ANALYSIS")
    print("Testing: Does mHC amplify gradients to sparse modules?")
    print('='*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)
    
    # Test 1: PEER + Engram WITHOUT mHC
    print(f"\n{'-'*40}")
    print("Configuration 1: PEER + Engram (no mHC)")
    print('-'*40)
    
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    config = IntegrationConfig(mode="hybrid")
    model1 = inject_peer_engram(model1, config)
    
    stats_no_mhc = run_gradient_analysis(
        model1, tokenizer, dataset, device, args.batches, "Without mHC"
    )
    
    del model1
    torch.cuda.empty_cache()
    
    # Test 2: PEER + Engram WITH mHC
    print(f"\n{'-'*40}")
    print("Configuration 2: PEER + Engram + mHC")
    print('-'*40)
    
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    config = IntegrationConfig(mode="hybrid")
    model2 = inject_peer_engram(model2, config)
    model2 = inject_mhc(model2, [7, 13, 17], num_streams=4)
    
    stats_with_mhc = run_gradient_analysis(
        model2, tokenizer, dataset, device, args.batches, "With mHC"
    )
    
    del model2
    torch.cuda.empty_cache()
    
    # Compare results
    print(f"\n{'='*60}")
    print("GRADIENT COMPARISON")
    print('='*60)
    
    # Aggregate by module type
    def aggregate_by_type(stats):
        aggregated = defaultdict(list)
        for name, data in stats.items():
            module_type = name.split('_')[0]
            aggregated[module_type].append(data['mean'])
        return {k: sum(v)/len(v) if v else 0 for k, v in aggregated.items()}
    
    agg_no_mhc = aggregate_by_type(stats_no_mhc)
    agg_with_mhc = aggregate_by_type(stats_with_mhc)
    
    print(f"\n{'Module':<15} {'Without mHC':>15} {'With mHC':>15} {'Change':>15}")
    print("-" * 62)
    
    for module_type in ['Engram', 'PEER']:
        no_mhc = agg_no_mhc.get(module_type, 0)
        with_mhc = agg_with_mhc.get(module_type, 0)
        if no_mhc > 0:
            change = ((with_mhc - no_mhc) / no_mhc) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        print(f"{module_type:<15} {no_mhc:>15.6f} {with_mhc:>15.6f} {change_str:>15}")
    
    # Detailed per-module stats
    print(f"\n{'-'*40}")
    print("Detailed Per-Module Statistics")
    print('-'*40)
    
    print("\nWithout mHC:")
    for name, data in sorted(stats_no_mhc.items()):
        print(f"  {name}: mean={data['mean']:.6f}, max={data['max']:.6f}")
    
    print("\nWith mHC:")
    for name, data in sorted(stats_with_mhc.items()):
        print(f"  {name}: mean={data['mean']:.6f}, max={data['max']:.6f}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print('='*60)
    
    engram_boost = 0
    peer_boost = 0
    
    if agg_no_mhc.get('Engram', 0) > 0:
        engram_boost = ((agg_with_mhc.get('Engram', 0) - agg_no_mhc['Engram']) / agg_no_mhc['Engram']) * 100
    if agg_no_mhc.get('PEER', 0) > 0:
        peer_boost = ((agg_with_mhc.get('PEER', 0) - agg_no_mhc['PEER']) / agg_no_mhc['PEER']) * 100
    
    print(f"\nGradient magnitude changes with mHC:")
    print(f"  Engram: {engram_boost:+.1f}%")
    print(f"  PEER: {peer_boost:+.1f}%")
    
    if engram_boost > 10 or peer_boost > 10:
        print(f"\n→ HYPOTHESIS SUPPORTED: mHC increases gradient flow to sparse modules")
        print(f"  mHC acts as a 'gradient bandwidth multiplier'")
        print(f"  Multi-stream residuals provide richer backprop paths")
    elif engram_boost < -10 or peer_boost < -10:
        print(f"\n→ UNEXPECTED: mHC decreases gradients to sparse modules")
        print(f"  mHC may be absorbing gradient signal")
    else:
        print(f"\n→ NEUTRAL: mHC has minimal effect on gradient magnitudes")
        print(f"  mHC's benefit may be in representation quality, not gradient flow")
    
    # Additional analysis: mHC stream statistics
    if 'mHC' in agg_with_mhc:
        print(f"\nmHC internal gradient magnitude: {agg_with_mhc['mHC']:.6f}")
        print("  This represents gradient flow through the multi-stream mechanism")


if __name__ == "__main__":
    main()
