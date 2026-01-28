#!/usr/bin/env python3
"""Longer training for PEER/Engram models"""

import os
import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llama_peer_engram import IntegrationConfig, inject_peer_engram

def train_model(mode: str, num_batches: int = 1000, batch_size: int = 4):
    print(f"\n{'='*70}")
    print(f"Training {mode.upper()} - {num_batches} batches")
    print('='*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config with higher LR for new modules
    config = IntegrationConfig(mode=mode)
    config.learning_rate = 5e-4  # Higher LR for new modules
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Inject modules
    model = inject_peer_engram(model, config)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)
    
    # Optimizer - higher LR for new modules, very low for backbone
    new_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'peer' in name or 'engram' in name:
            new_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': new_params, 'lr': 5e-4},           # High LR for new modules
        {'params': backbone_params, 'lr': 1e-6},      # Very low for backbone
    ], weight_decay=0.01)
    
    print(f"New module params: {sum(p.numel() for p in new_params):,}")
    print(f"Backbone params: {sum(p.numel() for p in backbone_params):,}")
    
    # Training loop
    model.train()
    total_loss = 0
    batch_count = 0
    
    data_iter = iter(dataloader)
    
    for step in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Store for Engram
        if hasattr(model, 'model'):
            model.model._current_input_ids = input_ids
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        batch_count += 1
        
        if step % 100 == 0:
            avg_loss = total_loss / batch_count
            print(f"  Step {step}: loss = {avg_loss:.4f}")
        
        if math.isnan(loss.item()):
            print("NaN detected, stopping")
            break
    
    final_loss = total_loss / batch_count
    print(f"\nFinal avg loss: {final_loss:.4f}")
    
    # Save
    save_path = f"./llama_{mode}_trained"
    os.makedirs(save_path, exist_ok=True)
    
    state_dict = {}
    for name, param in model.named_parameters():
        if 'peer' in name or 'engram' in name:
            state_dict[name] = param.cpu()
    
    torch.save(state_dict, f"{save_path}/weights.pt")
    print(f"Saved to {save_path}/weights.pt")
    
    del model
    torch.cuda.empty_cache()
    
    return final_loss

def evaluate_model(mode: str, tokenizer, dataset, device):
    """Evaluate perplexity"""
    from tqdm import tqdm
    
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
            state = model.state_dict()
            loaded = 0
            for name, param in saved.items():
                if name in state and state[name].shape == param.shape:
                    state[name].copy_(param)
                    loaded += 1
            print(f"  Loaded {loaded} tensors for {mode}")
        except:
            print(f"  No weights for {mode}")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=min(300, len(dataset)))):
            if i >= 300:
                break
            
            text = example['text']
            if not text.strip():
                continue
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            
            if hasattr(model, 'model'):
                model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids)
            
            num_tokens = inputs['attention_mask'].sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    ppl = math.exp(total_loss / total_tokens)
    
    del model
    torch.cuda.empty_cache()
    
    return ppl

def main():
    device = torch.device("cuda")
    
    # Train each mode
    for mode in ['engram', 'peer', 'hybrid']:
        train_model(mode, num_batches=1000, batch_size=4)
    
    # Evaluate all
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset = test_dataset.filter(lambda x: len(x['text'].strip()) > 10)
    
    results = {}
    for mode in ['base', 'engram', 'peer', 'hybrid']:
        print(f"\nEvaluating {mode}...")
        ppl = evaluate_model(mode, tokenizer, test_dataset, device)
        results[mode] = ppl
        print(f"  {mode}: PPL = {ppl:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Model':<10} {'Perplexity':>12} {'vs Base':>10}")
    print("-"*34)
    base_ppl = results['base']
    for mode in ['base', 'engram', 'peer', 'hybrid']:
        diff = ((results[mode] - base_ppl) / base_ppl) * 100
        diff_str = f"{diff:+.2f}%" if mode != 'base' else '-'
        print(f"{mode:<10} {results[mode]:>12.2f} {diff_str:>10}")

if __name__ == "__main__":
    main()
