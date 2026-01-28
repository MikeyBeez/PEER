#!/usr/bin/env python3
"""
Evaluate models on diverse datasets to address WikiText-2 n-gram bias.

Tests on:
- C4 (cleaned web crawl) - more diverse than WikiText
- Code (The Stack) - very different n-gram structure
"""

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_peer_engram import IntegrationConfig, inject_peer_engram
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


def evaluate_perplexity(model, tokenizer, dataset, device, max_samples=500, text_key='text', desc="Evaluating"):
    """Evaluate perplexity on a dataset."""
    model.eval()
    total_loss, total_tokens = 0, 0
    
    with torch.no_grad():
        for i, ex in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc=desc)):
            if i >= max_samples:
                break
            
            text = ex.get(text_key, ex.get('content', ''))
            if not text or len(text.strip()) < 20:
                continue
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            
            # Set current input IDs for Engram
            if hasattr(model, 'model'):
                model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=input_ids)
            
            n = inputs['attention_mask'].sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n
    
    if total_tokens == 0:
        return float('inf')
    return math.exp(total_loss / total_tokens)


def load_model_with_weights(model_name, config_mode, weights_path, device, mhc_layers=None):
    """Load model with specified configuration and trained weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if config_mode != "baseline":
        config = IntegrationConfig(mode=config_mode)
        model = inject_peer_engram(model, config)
    
    if mhc_layers:
        model = inject_mhc(model, mhc_layers, num_streams=4)
    
    # Load trained weights if available
    if weights_path and os.path.exists(weights_path):
        try:
            saved = torch.load(weights_path, map_location=device, weights_only=True)
            state = model.state_dict()
            loaded = 0
            for n, p in saved.items():
                if n in state and state[n].shape == p.shape:
                    state[n] = p.to(state[n].device)
                    loaded += 1
            model.load_state_dict(state)
            print(f"  Loaded {loaded} tensors from {weights_path}")
        except Exception as e:
            print(f"  Warning: Could not load weights: {e}")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=300, help='Samples to evaluate')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\n{'='*60}")
    print("CROSS-DATASET EVALUATION")
    print("Testing if gains generalize beyond WikiText-2's n-gram structure")
    print('='*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load C4 validation set (diverse web content)
    print(f"\nLoading C4 validation set...")
    try:
        c4_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
        c4_samples = list(c4_dataset.take(args.samples + 50))
        print(f"C4 loaded: {len(c4_samples)} samples")
    except Exception as e:
        print(f"Could not load C4: {e}")
        print("Falling back to WikiText-103 (larger, more diverse than WikiText-2)")
        c4_samples = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        c4_samples = c4_samples.filter(lambda x: len(x['text'].strip()) > 50)
    
    # Define configurations to test
    configs = [
        ("Baseline", "baseline", None, None),
        ("Engram only", "engram", "./llama_engram_v1/weights.pt", None),
        ("PEER only", "peer", "./llama_peer_large_v1/weights.pt", None),
        ("Triple-Sparse (Diff LR)", "hybrid", "./llama_differential_v1/weights.pt", [7, 13, 17]),
    ]
    
    results = {}
    
    for name, mode, weights, mhc_layers in configs:
        print(f"\n{'-'*40}")
        print(f"Evaluating: {name}")
        
        model = load_model_with_weights(model_name, mode, weights, device, mhc_layers)
        ppl = evaluate_perplexity(model, tokenizer, c4_samples, device, args.samples, desc=name)
        results[name] = ppl
        
        print(f"  Perplexity: {ppl:.2f}")
        
        del model
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS ON C4 / WIKITEXT-103")
    print('='*60)
    
    baseline_ppl = results.get("Baseline", 0)
    
    print(f"\nConfiguration                          PPL      Change")
    print("-" * 55)
    
    for name, ppl in results.items():
        if baseline_ppl > 0:
            change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "—"
        print(f"{name:40} {ppl:7.2f}  {change_str}")
    
    # Compare to WikiText-2 results
    print(f"\n{'='*60}")
    print("COMPARISON: WikiText-2 vs New Dataset")
    print('='*60)
    
    wikitext_results = {
        "Baseline": 16.54,
        "Engram only": 11.30,
        "PEER only": 14.81,
        "Triple-Sparse (Diff LR)": 11.05,
    }
    
    print(f"\n{'Configuration':<30} {'WikiText-2':>15} {'New Dataset':>15}")
    print("-" * 62)
    
    for name in results:
        wt2 = wikitext_results.get(name, 0)
        new = results[name]
        if wt2 > 0 and baseline_ppl > 0:
            wt2_change = ((wt2 - wikitext_results["Baseline"]) / wikitext_results["Baseline"]) * 100
            new_change = ((new - baseline_ppl) / baseline_ppl) * 100
            print(f"{name:<30} {wt2:>7.2f} ({wt2_change:+.1f}%) {new:>7.2f} ({new_change:+.1f}%)")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS: N-gram Bias Test")
    print('='*60)
    
    if "Engram only" in results:
        engram_wt2 = ((wikitext_results["Engram only"] - wikitext_results["Baseline"]) / wikitext_results["Baseline"]) * 100
        engram_new = ((results["Engram only"] - baseline_ppl) / baseline_ppl) * 100
        
        print(f"\nEngram improvement:")
        print(f"  WikiText-2: {engram_wt2:+.1f}%")
        print(f"  New dataset: {engram_new:+.1f}%")
        print(f"  Delta: {engram_new - engram_wt2:+.1f} percentage points")
        
        if engram_new > engram_wt2 + 5:
            print("\n→ Engram gains DECREASED on diverse data")
            print("  Supports hypothesis: WikiText-2 favors n-gram memorization")
        elif engram_new < engram_wt2 - 5:
            print("\n→ Engram gains INCREASED on diverse data")
            print("  Engram generalizes well across domains")
        else:
            print("\n→ Engram gains are CONSISTENT across datasets")
            print("  N-gram patterns provide stable value")
    
    if "PEER only" in results:
        peer_wt2 = ((wikitext_results["PEER only"] - wikitext_results["Baseline"]) / wikitext_results["Baseline"]) * 100
        peer_new = ((results["PEER only"] - baseline_ppl) / baseline_ppl) * 100
        
        print(f"\nPEER improvement:")
        print(f"  WikiText-2: {peer_wt2:+.1f}%")
        print(f"  New dataset: {peer_new:+.1f}%")
        print(f"  Delta: {peer_new - peer_wt2:+.1f} percentage points")


if __name__ == "__main__":
    main()
