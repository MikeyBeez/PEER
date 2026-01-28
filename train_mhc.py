#!/usr/bin/env python3
"""
Train and evaluate with mHC (Manifold-constrained Hyper-Connections)

Tests mHC integration with TinyLlama to measure perplexity impact.
"""

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_mhc import mHC, StreamExpander, StreamReducer


class mHCLayer(nn.Module):
    """
    Standalone mHC layer added after decoder layer (like Engram).

    Adds gated mHC-processed hidden states to original output.
    """

    def __init__(self, hidden_size: int, num_streams: int = 4, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_streams = num_streams

        # mHC components
        self.expander = StreamExpander(hidden_size, num_streams)
        self.reducer = StreamReducer(hidden_size, num_streams)
        self.mhc = mHC(hidden_size, num_streams)

        # Output projection - very small init
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.001)

        # Learnable gate (start very small)
        self.gate_scale = nn.Parameter(torch.tensor(-4.0))  # sigmoid(-4) ≈ 0.018

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D input
        squeeze_output = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True

        # mHC path: expand -> mHC mix -> reduce
        x_expanded = self.expander(hidden_states)  # [B, T, ns, D]
        x_mhc = self.mhc(x_expanded)  # [B, T, ns, D]
        x_reduced = self.reducer(x_mhc)  # [B, T, D]

        # Project with learned gate
        mhc_out = self.out_proj(x_reduced)
        gate = torch.sigmoid(self.gate_scale)
        output = hidden_states + gate * mhc_out

        if squeeze_output:
            output = output.squeeze(0)
        return output


class mHCWrapper(nn.Module):
    """
    Wraps a LLaMA decoder layer to add mHC processing after it.
    """

    def __init__(self, original_layer, hidden_size: int, num_streams: int = 4, layer_idx: int = 0):
        super().__init__()
        self.original_layer = original_layer
        self.mhc_layer = mHCLayer(hidden_size, num_streams, layer_idx)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Original forward pass
        original_outputs = self.original_layer(hidden_states, **kwargs)

        # Handle both Tensor and tuple outputs
        if isinstance(original_outputs, tuple):
            layer_output = original_outputs[0]
            extra = original_outputs[1:]
        else:
            layer_output = original_outputs
            extra = None

        # Add mHC processing
        enhanced_output = self.mhc_layer(layer_output)

        if extra is not None:
            return (enhanced_output,) + extra
        return enhanced_output


def inject_mhc(model, mhc_layers=None, num_streams=4):
    """Inject mHC wrappers into specified layers"""
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if mhc_layers is None:
        # Apply to middle layers (where mHC can help stabilize)
        mhc_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    print(f"Injecting mHC into layers: {mhc_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num streams: {num_streams}")

    layers = model.model.layers

    for idx in mhc_layers:
        if idx < num_layers:
            original_layer = layers[idx]
            # Get device from original layer
            device = next(original_layer.parameters()).device
            dtype = next(original_layer.parameters()).dtype
            wrapped = mHCWrapper(original_layer, hidden_size, num_streams, idx)
            # Move mHC layer to same device/dtype
            wrapped.mhc_layer = wrapped.mhc_layer.to(device=device, dtype=dtype)
            layers[idx] = wrapped
            print(f"  Layer {idx}: wrapped with mHC (device={device})")

    return model


def train_mhc(num_batches=1500, num_streams=4, mhc_layers=None):
    """Train model with mHC"""
    print(f"\n{'='*60}")
    print(f"Training with mHC - {num_batches} batches")
    print('='*60)

    device = torch.device("cuda")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Inject mHC
    model = inject_mhc(model, mhc_layers, num_streams)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    mhc_params = sum(p.numel() for n, p in model.named_parameters()
                     if 'mhc_layer' in n)
    print(f"Total params: {total_params:,}")
    print(f"mHC params: {mhc_params:,}")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

    # Separate params: mHC gets higher LR
    mhc_param_list = [p for n, p in model.named_parameters()
                      if 'mhc_layer' in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if 'mhc_layer' not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': mhc_param_list, 'lr': 1e-3},
        {'params': backbone_params, 'lr': 1e-6},
    ])

    print(f"mHC trainable params: {sum(p.numel() for p in mhc_param_list):,}")

    model.train()
    total_loss = 0
    data_iter = iter(dataloader)

    for step in tqdm(range(num_batches), desc="Training mHC"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 250 == 0:
            # Check gate values
            gate_vals = []
            for name, module in model.named_modules():
                if isinstance(module, mHCLayer):
                    gate_vals.append(f"{torch.sigmoid(module.gate_scale).item():.4f}")
            print(f"  Step {step}: loss = {total_loss/(step+1):.4f}, gates = {gate_vals}")

    final_loss = total_loss / num_batches
    print(f"Final loss: {final_loss:.4f}")

    # Save mHC weights
    save_path = "./llama_mhc_v1"
    os.makedirs(save_path, exist_ok=True)
    state = {n: p.cpu() for n, p in model.named_parameters() if 'mhc_layer' in n}
    torch.save(state, f"{save_path}/weights.pt")
    print(f"Saved mHC weights to {save_path}")

    del model
    torch.cuda.empty_cache()
    return final_loss


def evaluate_mhc(tokenizer, test_data, device, num_streams=4, mhc_layers=None):
    """Evaluate model with mHC"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = inject_mhc(model, mhc_layers, num_streams)

    # Load saved weights
    try:
        saved = torch.load("./llama_mhc_v1/weights.pt", map_location=device, weights_only=True)
        state = model.state_dict()
        loaded = 0
        for n, p in saved.items():
            if n in state and state[n].shape == p.shape:
                state[n] = p.to(state[n].device)
                loaded += 1
        model.load_state_dict(state)
        print(f"  Loaded {loaded} mHC tensors")
    except Exception as e:
        print(f"  Load error: {e}")

    model.eval()
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for i, ex in enumerate(tqdm(test_data, total=min(400, len(test_data)), desc="Evaluating mHC")):
            if i >= 400:
                break
            if not ex['text'].strip():
                continue

            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=input_ids)

            n = inputs['attention_mask'].sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n

    ppl = math.exp(total_loss / total_tokens)
    del model
    torch.cuda.empty_cache()
    return ppl


def evaluate_baseline(tokenizer, test_data, device):
    """Evaluate baseline model without mHC"""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for i, ex in enumerate(tqdm(test_data, total=min(400, len(test_data)), desc="Evaluating baseline")):
            if i >= 400:
                break
            if not ex['text'].strip():
                continue

            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train mHC')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--batches', type=int, default=1500, help='Training batches')
    parser.add_argument('--streams', type=int, default=4, help='Number of mHC streams')
    args = parser.parse_args()

    device = torch.device("cuda")

    # Use layers 5, 11, 17 (spread across 22-layer model)
    mhc_layers = [5, 11, 17]

    if not args.eval_only:
        train_mhc(num_batches=args.batches, num_streams=args.streams, mhc_layers=mhc_layers)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test = test.filter(lambda x: len(x['text'].strip()) > 10)

    print("\nBaseline (no mHC):")
    baseline_ppl = evaluate_baseline(tokenizer, test, device)
    print(f"  PPL = {baseline_ppl:.2f}")

    print("\nmHC (trained):")
    mhc_ppl = evaluate_mhc(tokenizer, test, device, num_streams=args.streams, mhc_layers=mhc_layers)
    print(f"  PPL = {mhc_ppl:.2f}")

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    diff = ((mhc_ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"Baseline:  PPL = {baseline_ppl:.2f}")
    print(f"+ mHC:     PPL = {mhc_ppl:.2f}  ({diff:+.2f}%)")

    if diff < 0:
        print(f"\nmHC improved perplexity by {-diff:.2f}%")
    else:
        print(f"\nmHC increased perplexity by {diff:.2f}%")


if __name__ == "__main__":
    main()
