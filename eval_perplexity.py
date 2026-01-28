#!/usr/bin/env python3
"""Evaluate perplexity on test set for all models"""

import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llama_peer_engram import IntegrationConfig, inject_peer_engram

def evaluate_perplexity(model, tokenizer, dataset, device, max_samples=200, max_length=256):
    """Calculate perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc="Evaluating")):
            if i >= max_samples:
                break
            
            text = example['text']
            if not text.strip():
                continue
            
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=False
            )
            
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Store for Engram
            if hasattr(model, 'model') and hasattr(model.model, '_current_input_ids'):
                model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
            
            # Loss is already averaged over tokens
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss

def load_model_with_weights(mode, device):
    """Load model and inject trained weights"""
    config = IntegrationConfig(mode=mode)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    if mode != "base":
        model = inject_peer_engram(model, config)
        
        # Load trained weights
        weights_path = f"./llama_{mode}_finetuned/peer_engram_weights.pt"
        try:
            saved_weights = torch.load(weights_path, map_location=device, weights_only=True)
            model_state = model.state_dict()
            loaded = 0
            for name, param in saved_weights.items():
                if name in model_state and model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded += 1
            print(f"  Loaded {loaded} weight tensors")
        except FileNotFoundError:
            print(f"  No trained weights found, using initialized")
    
    return model

def main():
    print("="*70)
    print("Perplexity Evaluation on WikiText-2 Test Set")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test dataset
    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)
    print(f"Test samples: {len(dataset)}\n")
    
    results = {}
    
    for mode in ["base", "engram", "peer", "hybrid"]:
        print(f"\n{'='*70}")
        print(f"Evaluating: {mode.upper()}")
        print("="*70)
        
        model = load_model_with_weights(mode, device)
        
        perplexity, avg_loss = evaluate_perplexity(
            model, tokenizer, dataset, device,
            max_samples=200,
            max_length=256
        )
        
        results[mode] = {"perplexity": perplexity, "loss": avg_loss}
        print(f"\n  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<12} {'Loss':>10} {'Perplexity':>12} {'vs Base':>10}")
    print("-"*46)
    
    base_ppl = results["base"]["perplexity"]
    for mode in ["base", "engram", "peer", "hybrid"]:
        r = results[mode]
        diff = ((r["perplexity"] - base_ppl) / base_ppl) * 100
        diff_str = f"{diff:+.1f}%" if mode != "base" else "-"
        print(f"{mode:<12} {r['loss']:>10.4f} {r['perplexity']:>12.2f} {diff_str:>10}")

if __name__ == "__main__":
    main()
