#!/usr/bin/env python3
"""
Staggered Training for Triple-Sparse Architecture

Trains PEER and Engram in phases to prevent functional redundancy:
1. Phase 1: Train Engram alone (learn common patterns)
2. Phase 2: Freeze Engram, train PEER (learn what Engram can't)
3. Phase 3: Fine-tune both together (coordination)

This breaks the "Sparsity Overlap" bottleneck observed when training together.
"""

import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llama_peer_engram import IntegrationConfig, inject_peer_engram, LlamaEngram, LlamaPEER
from llama_mhc import mHC, StreamExpander, StreamReducer


class mHCLayer(nn.Module):
    """Standalone mHC layer added after decoder layer."""

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
    """Wraps a decoder layer to add mHC processing after it."""

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
    """Inject mHC wrappers into specified layers."""
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if mhc_layers is None:
        mhc_layers = [7, 13, 17]

    print(f"Injecting mHC into layers: {mhc_layers}")

    layers = model.model.layers

    for idx in mhc_layers:
        if idx < num_layers:
            original_layer = layers[idx]
            device = next(original_layer.parameters()).device
            dtype = next(original_layer.parameters()).dtype
            wrapped = mHCWrapper(original_layer, hidden_size, num_streams, idx)
            wrapped.mhc_layer = wrapped.mhc_layer.to(device=device, dtype=dtype)
            layers[idx] = wrapped
            print(f"  Layer {idx}: wrapped with mHC")

    return model


def freeze_modules(model, freeze_engram=False, freeze_peer=False, freeze_mhc=False):
    """Freeze/unfreeze specific module types."""
    for name, param in model.named_parameters():
        if 'engram' in name and freeze_engram:
            param.requires_grad = False
        elif 'peer' in name and freeze_peer:
            param.requires_grad = False
        elif 'mhc_layer' in name and freeze_mhc:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model):
    """Count trainable parameters by module type."""
    engram = sum(p.numel() for n, p in model.named_parameters() if 'engram' in n and p.requires_grad)
    peer = sum(p.numel() for n, p in model.named_parameters() if 'peer' in n and p.requires_grad)
    mhc = sum(p.numel() for n, p in model.named_parameters() if 'mhc_layer' in n and p.requires_grad)
    backbone = sum(p.numel() for n, p in model.named_parameters()
                   if 'engram' not in n and 'peer' not in n and 'mhc_layer' not in n and p.requires_grad)
    return {'engram': engram, 'peer': peer, 'mhc': mhc, 'backbone': backbone}


def train_phase(model, dataloader, device, num_batches, phase_name, lr_new=1e-3, lr_backbone=1e-6):
    """Run one training phase."""
    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name} ({num_batches} batches)")
    print('='*60)

    counts = count_trainable(model)
    print(f"Trainable: Engram={counts['engram']:,}, PEER={counts['peer']:,}, mHC={counts['mhc']:,}")

    # Separate params
    new_params = [p for n, p in model.named_parameters()
                  if ('peer' in n or 'engram' in n or 'mhc_layer' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if ('peer' not in n and 'engram' not in n and 'mhc_layer' not in n) and p.requires_grad]

    if not new_params and not backbone_params:
        print("No trainable parameters!")
        return 0

    param_groups = []
    if new_params:
        param_groups.append({'params': new_params, 'lr': lr_new})
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})

    optimizer = torch.optim.AdamW(param_groups)

    model.train()
    total_loss = 0
    data_iter = iter(dataloader)

    for step in tqdm(range(num_batches), desc=phase_name):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        model.model._current_input_ids = input_ids

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 500 == 0 and step > 0:
            print(f"  Step {step}: loss = {total_loss/(step+1):.4f}")

    final_loss = total_loss / num_batches
    print(f"Phase complete. Avg loss: {final_loss:.4f}")
    return final_loss


def train_staggered(phase1_batches=1000, phase2_batches=1500, phase3_batches=500):
    """Train with staggered freezing strategy."""
    print(f"\n{'='*60}")
    print("STAGGERED TRAINING FOR TRIPLE-SPARSE")
    print("Strategy: Engram first → PEER second → Fine-tune together")
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

    # Inject all modules
    config = IntegrationConfig(mode="hybrid")
    model = inject_peer_engram(model, config)
    print("Injected PEER + Engram (hybrid)")

    mhc_layers = [7, 13, 17]
    model = inject_mhc(model, mhc_layers, num_streams=4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

    # ==========================================
    # Phase 1: Train Engram + mHC (freeze PEER)
    # ==========================================
    unfreeze_all(model)
    freeze_modules(model, freeze_peer=True)
    train_phase(model, dataloader, device, phase1_batches,
                "Phase 1: Engram + mHC (PEER frozen)", lr_new=1e-3)

    # ==========================================
    # Phase 2: Train PEER (freeze Engram)
    # ==========================================
    unfreeze_all(model)
    freeze_modules(model, freeze_engram=True)
    train_phase(model, dataloader, device, phase2_batches,
                "Phase 2: PEER + mHC (Engram frozen)", lr_new=1e-3)

    # ==========================================
    # Phase 3: Fine-tune all together
    # ==========================================
    unfreeze_all(model)
    train_phase(model, dataloader, device, phase3_batches,
                "Phase 3: All together (fine-tuning)", lr_new=5e-4)

    # Save weights
    save_path = "./llama_staggered_v1"
    os.makedirs(save_path, exist_ok=True)
    state = {n: p.cpu() for n, p in model.named_parameters()
             if 'peer' in n or 'engram' in n or 'mhc_layer' in n}
    torch.save(state, f"{save_path}/weights.pt")
    print(f"\nSaved weights to {save_path}")

    del model
    torch.cuda.empty_cache()
    return save_path


def evaluate(weights_path, mhc_layers=None):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    device = torch.device("cuda")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and setup model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    config = IntegrationConfig(mode="hybrid")
    model = inject_peer_engram(model, config)

    if mhc_layers is None:
        mhc_layers = [7, 13, 17]
    model = inject_mhc(model, mhc_layers, num_streams=4)

    # Load trained weights
    try:
        saved = torch.load(f"{weights_path}/weights.pt", map_location=device, weights_only=True)
        state = model.state_dict()
        loaded = 0
        for n, p in saved.items():
            if n in state and state[n].shape == p.shape:
                state[n] = p.to(state[n].device)
                loaded += 1
        model.load_state_dict(state)
        print(f"Loaded {loaded} tensors from {weights_path}")
    except Exception as e:
        print(f"Load error: {e}")

    # Load test data
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test = test.filter(lambda x: len(x['text'].strip()) > 10)

    # Evaluate
    model.eval()
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for i, ex in enumerate(tqdm(test, total=min(400, len(test)), desc="Evaluating")):
            if i >= 400:
                break
            if not ex['text'].strip():
                continue

            inputs = tokenizer(ex['text'], return_tensors="pt", truncation=True, max_length=256)
            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)
            model.model._current_input_ids = input_ids

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=input_ids)

            n = inputs['attention_mask'].sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n

    ppl = math.exp(total_loss / total_tokens)
    print(f"\nStaggered Triple-Sparse PPL: {ppl:.2f}")

    # Compare to baseline
    baseline_ppl = 16.54  # Known baseline
    diff = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"Baseline: {baseline_ppl:.2f}")
    print(f"Change: {diff:+.2f}%")

    del model
    torch.cuda.empty_cache()
    return ppl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run staggered training')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--phase1', type=int, default=1000, help='Phase 1 batches (Engram)')
    parser.add_argument('--phase2', type=int, default=1500, help='Phase 2 batches (PEER)')
    parser.add_argument('--phase3', type=int, default=500, help='Phase 3 batches (Fine-tune)')
    args = parser.parse_args()

    if not args.eval_only:
        weights_path = train_staggered(
            phase1_batches=args.phase1,
            phase2_batches=args.phase2,
            phase3_batches=args.phase3
        )
    else:
        weights_path = "./llama_staggered_v1"

    ppl = evaluate(weights_path)

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Baseline TinyLlama:     16.54")
    print(f"Joint Triple-Sparse:   13.79 (-16.6%)")
    print(f"Staggered Training:    {ppl:.2f} ({((ppl-16.54)/16.54)*100:+.1f}%)")


if __name__ == "__main__":
    main()
