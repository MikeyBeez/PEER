#!/usr/bin/env python3
"""Train and evaluate with fixed input_ids propagation"""

import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llama_peer_engram import IntegrationConfig, inject_peer_engram

def train_model(mode: str, num_batches: int = 1500):
    print(f"\n{'='*60}")
    print(f"Training {mode.upper()} - {num_batches} batches")
    print('='*60)
    
    device = torch.device("cuda")
    config = IntegrationConfig(mode=mode)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    model = inject_peer_engram(model, config)
    
    # Verify parent model is set
    for layer in model.model.layers:
        if hasattr(layer, '_parent_model'):
            print(f"  Parent model set: {layer._parent_model is not None}")
            break
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256, padding="max_length")
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)
    
    # Separate params
    new_params = [p for n, p in model.named_parameters() if ('peer' in n or 'engram' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if ('peer' not in n and 'engram' not in n) and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {'params': new_params, 'lr': 1e-3},       # Higher LR for new modules
        {'params': backbone_params, 'lr': 1e-6},  # Very low for backbone
    ])
    
    print(f"New params: {sum(p.numel() for p in new_params):,}")
    
    model.train()
    total_loss = 0
    data_iter = iter(dataloader)
    
    for step in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Store input_ids for Engram layers to access
        model.model._current_input_ids = input_ids
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if step % 250 == 0:
            print(f"  Step {step}: loss = {total_loss/(step+1):.4f}")
    
    final_loss = total_loss / num_batches
    print(f"Final loss: {final_loss:.4f}")
    
    # Save
    save_path = f"./llama_{mode}_v2"
    os.makedirs(save_path, exist_ok=True)
    state = {n: p.cpu() for n, p in model.named_parameters() if 'peer' in n or 'engram' in n}
    torch.save(state, f"{save_path}/weights.pt")
    
    del model
    torch.cuda.empty_cache()
    return final_loss

def evaluate_model(mode: str, tokenizer, test_data, device):
    config = IntegrationConfig(mode=mode)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    if mode != "base":
        model = inject_peer_engram(model, config)
        
        try:
            saved = torch.load(f"./llama_{mode}_v2/weights.pt", map_location=device, weights_only=True)
            state = model.state_dict()
            loaded = 0
            for n, p in saved.items():
                if n in state and state[n].shape == p.shape:
                    state[n] = p.to(state[n].device)
                    loaded += 1
            model.load_state_dict(state)
            print(f"  Loaded {loaded} tensors")
        except Exception as e:
            print(f"  Load error: {e}")
    
    model.eval()
    total_loss, total_tokens = 0, 0
    
    with torch.no_grad():
        for i, ex in enumerate(tqdm(test_data, total=min(400, len(test_data)), desc=mode)):
            if i >= 400:
                break
            if not ex['text'].strip():
                continue
            
            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue
            
            input_ids = inputs['input_ids'].to(device)
            
            # Set input_ids for Engram
            if hasattr(model, 'model'):
                model.model._current_input_ids = input_ids
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=input_ids)
            
            n = inputs['attention_mask'].sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n
    
    ppl = math.exp(total_loss / total_tokens)
    del model
    torch.cuda.empty_cache()
    return ppl

def main():
    device = torch.device("cuda")
    
    # Train
    for mode in ['engram', 'peer', 'hybrid']:
        train_model(mode, num_batches=1500)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test = test.filter(lambda x: len(x['text'].strip()) > 10)
    
    results = {}
    for mode in ['base', 'engram', 'peer', 'hybrid']:
        print(f"\n{mode}:")
        results[mode] = evaluate_model(mode, tokenizer, test, device)
        print(f"  PPL = {results[mode]:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    base = results['base']
    for m in ['base', 'engram', 'peer', 'hybrid']:
        diff = ((results[m] - base) / base) * 100
        print(f"{m:<10} PPL={results[m]:>7.2f}  {diff:+.2f}%" if m != 'base' else f"{m:<10} PPL={results[m]:>7.2f}")

if __name__ == "__main__":
    main()
