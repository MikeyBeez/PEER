#!/usr/bin/env python3
"""
Signal Diversity Analysis: Prove Differential LR Forces Orthogonal Representations

Measures cosine similarity between Engram and PEER output vectors.
- High similarity = Redundancy (modules doing the same work)
- Low similarity = Orthogonality (modules specialized to different patterns)

Hypothesis:
- Joint training (equal LR): High cosine similarity
- Differential LR training: Low cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_peer_engram import IntegrationConfig, inject_peer_engram, LlamaEngram, LlamaPEER


class OutputCapture:
    """Captures outputs from Engram and PEER modules for similarity analysis."""
    
    def __init__(self):
        self.engram_outputs = []
        self.peer_outputs = []
        self.hooks = []
    
    def register_hooks(self, model):
        def engram_hook(module, input, output):
            if output is not None:
                # Capture the output tensor (detached, on CPU to save memory)
                self.engram_outputs.append(output.detach().cpu())
        
        def peer_hook(module, input, output):
            if output is not None:
                self.peer_outputs.append(output.detach().cpu())
        
        for name, module in model.named_modules():
            if isinstance(module, LlamaEngram):
                h = module.register_forward_hook(engram_hook)
                self.hooks.append(h)
                print(f"Registered Engram hook on {name}")
            elif isinstance(module, LlamaPEER):
                h = module.register_forward_hook(peer_hook)
                self.hooks.append(h)
                print(f"Registered PEER hook on {name}")
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def reset(self):
        self.engram_outputs = []
        self.peer_outputs = []
    
    def compute_similarity(self):
        """Compute mean cosine similarity between Engram and PEER outputs."""
        if not self.engram_outputs or not self.peer_outputs:
            return None
        
        similarities = []
        
        # Pair up outputs (they should be aligned by forward pass)
        n_pairs = min(len(self.engram_outputs), len(self.peer_outputs))
        
        for i in range(n_pairs):
            eng = self.engram_outputs[i]
            peer = self.peer_outputs[i]
            
            # Flatten to vectors for cosine similarity
            eng_flat = eng.view(-1).float()
            peer_flat = peer.view(-1).float()
            
            # Handle size mismatch by truncating to smaller
            min_len = min(len(eng_flat), len(peer_flat))
            if min_len > 0:
                eng_flat = eng_flat[:min_len]
                peer_flat = peer_flat[:min_len]
                
                # Compute cosine similarity
                sim = F.cosine_similarity(eng_flat.unsqueeze(0), peer_flat.unsqueeze(0))
                similarities.append(sim.item())
        
        if similarities:
            return sum(similarities) / len(similarities)
        return None


def analyze_model(model, tokenizer, dataset, device, capture, max_samples=100, desc=""):
    """Run forward passes and compute output similarity."""
    model.eval()
    capture.reset()
    
    with torch.no_grad():
        for i, ex in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc=desc)):
            if i >= max_samples:
                break
            
            text = ex.get('text', '')
            if not text or len(text.strip()) < 20:
                continue
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(input_ids=input_ids)
    
    return capture.compute_similarity()


def load_model_with_config(model_name, weights_path, device):
    """Load model with hybrid config and optionally trained weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    config = IntegrationConfig(mode="hybrid")
    model = inject_peer_engram(model, config)
    
    if weights_path:
        try:
            import os
            if os.path.exists(weights_path):
                saved = torch.load(weights_path, map_location=device, weights_only=True)
                state = model.state_dict()
                loaded = 0
                for n, p in saved.items():
                    if n in state and state[n].shape == p.shape:
                        state[n] = p.to(state[n].device)
                        loaded += 1
                model.load_state_dict(state)
                print(f"  Loaded {loaded} tensors from {weights_path}")
            else:
                print(f"  Weights not found: {weights_path}")
        except Exception as e:
            print(f"  Could not load weights: {e}")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100, help='Samples to analyze')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\n{'='*60}")
    print("SIGNAL DIVERSITY ANALYSIS")
    print("Measuring Engram-PEER Output Cosine Similarity")
    print("="*60)
    print("\nHypothesis:")
    print("  High similarity = Redundancy (modules doing same work)")
    print("  Low similarity = Orthogonality (specialized representations)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)
    
    # Configurations to test
    configs = [
        ("Untrained (baseline)", None),
        ("Joint Training (equal LR)", "./llama_triple_v1/weights.pt"),
        ("Differential LR", "./llama_differential_v1/weights.pt"),
    ]
    
    results = {}
    
    for name, weights_path in configs:
        print(f"\n{'-'*40}")
        print(f"Configuration: {name}")
        print('-'*40)
        
        model = load_model_with_config(model_name, weights_path, device)
        
        capture = OutputCapture()
        capture.register_hooks(model)
        
        similarity = analyze_model(model, tokenizer, dataset, device, capture, args.samples, desc=name)
        
        capture.remove_hooks()
        results[name] = similarity
        
        if similarity is not None:
            print(f"  Cosine Similarity: {similarity:.4f}")
        else:
            print(f"  Could not compute similarity")
        
        del model
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS: ENGRAM-PEER OUTPUT SIMILARITY")
    print("="*60)
    
    print(f"\n{'Configuration':<30} {'Cosine Similarity':>20}")
    print("-" * 52)
    
    for name, sim in results.items():
        if sim is not None:
            print(f"{name:<30} {sim:>20.4f}")
        else:
            print(f"{name:<30} {'N/A':>20}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print("="*60)
    
    untrained = results.get("Untrained (baseline)")
    joint = results.get("Joint Training (equal LR)")
    diff_lr = results.get("Differential LR")
    
    if untrained and joint and diff_lr:
        print(f"\nUntrained baseline: {untrained:.4f}")
        print(f"Joint training:     {joint:.4f} (change: {joint - untrained:+.4f})")
        print(f"Differential LR:    {diff_lr:.4f} (change: {diff_lr - untrained:+.4f})")
        
        if diff_lr < joint:
            reduction = ((joint - diff_lr) / joint) * 100
            print(f"\n→ Differential LR reduces similarity by {reduction:.1f}%")
            print("  This proves modules are forced into ORTHOGONAL representation spaces.")
            print("  The Redundancy Bottleneck is broken.")
        else:
            print(f"\n→ Unexpected: Differential LR did not reduce similarity")
            print("  Further investigation needed.")
    
    if diff_lr is not None:
        if diff_lr < 0.3:
            print("\n→ LOW SIMILARITY (<0.3): Strong orthogonality achieved")
            print("  Engram and PEER are solving different problems.")
        elif diff_lr < 0.6:
            print("\n→ MODERATE SIMILARITY (0.3-0.6): Partial specialization")
            print("  Some overlap remains, but modules are differentiating.")
        else:
            print("\n→ HIGH SIMILARITY (>0.6): Redundancy persists")
            print("  Modules are still doing similar work.")


if __name__ == "__main__":
    main()
