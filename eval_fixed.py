#!/usr/bin/env python3
"""Fixed evaluation - properly load weights into model"""

import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llama_peer_engram import IntegrationConfig, inject_peer_engram

def evaluate_model(mode: str, tokenizer, dataset, device):
    """Evaluate perplexity with proper weight loading"""
    
    config = IntegrationConfig(mode=mode)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if mode != "base":
        model = inject_peer_engram(model, config)
        
        weights_path = f"./llama_{mode}_trained/weights.pt"
        try:
            saved = torch.load(weights_path, map_location=device, weights_only=True)
            
            # Properly load weights using load_state_dict with strict=False
            model_state = model.state_dict()
            
            # Update state dict with saved weights
            updated = 0
            for name, param in saved.items():
                if name in model_state and model_state[name].shape == param.shape:
                    model_state[name] = param.to(model_state[name].device)
                    updated += 1
            
            # Actually load the updated state dict back into model
            model.load_state_dict(model_state)
            print(f"  Loaded {updated} tensors for {mode}")
            
            # Verify weights changed
            for name, param in model.named_parameters():
                if 'engram' in name or 'peer' in name:
                    print(f"    {name.split('.')[-2]}.{name.split('.')[-1]}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
                    break
                    
        except Exception as e:
            print(f"  Error loading weights: {e}")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=min(300, len(dataset)), desc=f"Eval {mode}")):
            if i >= 300:
                break
            
            text = example['text']
            if not text.strip():
                continue
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            
            # Set input_ids for Engram
            if hasattr(model, 'model'):
                model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids)
            
            num_tokens = inputs['attention_mask'].sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    del model
    torch.cuda.empty_cache()
    
    return ppl, avg_loss

def main():
    device = torch.device("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test set
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset = test_dataset.filter(lambda x: len(x['text'].strip()) > 10)
    print(f"Test samples: {len(test_dataset)}\n")
    
    results = {}
    
    for mode in ['base', 'engram', 'peer', 'hybrid']:
        print(f"\n{'='*50}")
        print(f"Evaluating: {mode.upper()}")
        print('='*50)
        
        ppl, loss = evaluate_model(mode, tokenizer, test_dataset, device)
        results[mode] = {'ppl': ppl, 'loss': loss}
        print(f"\n  Loss: {loss:.4f}, Perplexity: {ppl:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<10} {'Loss':>8} {'PPL':>10} {'vs Base':>12}")
    print("-"*42)
    
    base_ppl = results['base']['ppl']
    for mode in ['base', 'engram', 'peer', 'hybrid']:
        r = results[mode]
        diff = ((r['ppl'] - base_ppl) / base_ppl) * 100
        diff_str = f"{diff:+.2f}%" if mode != 'base' else '-'
        print(f"{mode:<10} {r['loss']:>8.4f} {r['ppl']:>10.2f} {diff_str:>12}")

if __name__ == "__main__":
    main()
