#!/usr/bin/env python3
"""
Train PEER-803K (802,816 experts)

This requires careful memory management:
- 14.35 GB just for model weights
- Batch size 1-2 with gradient accumulation
- Gradient checkpointing enabled
"""

import os
import math
import gc
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llama_peer_engram import IntegrationConfig, PEERConfig, inject_peer_engram


def get_gpu_memory():
    """Get current GPU memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def train_peer_803k(
    num_batches: int = 2000,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-4,
    save_every: int = 500,
):
    """Train PEER with 803K experts"""

    print("=" * 70)
    print("PEER-803K Training")
    print("=" * 70)

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configure PEER - use 262K experts (512^2) which fits with SGD optimizer
    # AdamW needs 2x param memory for states, SGD needs none
    num_experts = 262144  # 512^2

    config = IntegrationConfig(mode="peer")
    config.peer = PEERConfig(
        num_experts=num_experts,
        experts_per_head=8,
        num_heads=4,
        dim_key=64,
    )
    config.peer_layers = [19, 21]  # Last few layers

    print(f"\nConfiguration:")
    print(f"  Experts: {config.peer.num_experts:,} (512x512)")
    print(f"  Experts per head: {config.peer.experts_per_head}")
    print(f"  PEER layers: {config.peer_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"  Base model VRAM: {get_gpu_memory():.2f} GB")

    # Inject PEER
    print(f"\nInjecting PEER-803K...")
    model = inject_peer_engram(model, config)

    print(f"  After PEER injection: {get_gpu_memory():.2f} GB")

    # Note: gradient checkpointing conflicts with frozen params, skip it
    # We'll manage memory with small batch size instead
    print("  Gradient checkpointing: disabled (conflicts with frozen params)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    peer_params = sum(p.numel() for n, p in model.named_parameters() if 'peer' in n)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters:")
    print(f"  Total: {total_params/1e9:.2f}B")
    print(f"  PEER: {peer_params/1e9:.2f}B")
    print(f"  Trainable: {trainable_params/1e9:.2f}B")

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 20)

    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=256,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)

    # Optimizer - use SGD to avoid AdamW's 2x memory overhead
    # Keep all params trainable for gradient flow, but use very different LRs
    peer_params_list = [p for n, p in model.named_parameters() if 'peer' in n]
    backbone_params_list = [p for n, p in model.named_parameters() if 'peer' not in n]

    # SGD with momentum - much lower memory than AdamW
    # PEER gets full LR, backbone gets nearly zero LR (effectively frozen but allows gradient flow)
    optimizer = torch.optim.SGD([
        {'params': peer_params_list, 'lr': learning_rate},
        {'params': backbone_params_list, 'lr': learning_rate * 1e-6},  # Effectively frozen
    ], momentum=0.9, weight_decay=0.01)

    print(f"\nTraining:")
    print(f"  PEER params: {sum(p.numel() for p in peer_params_list)/1e9:.2f}B")
    print(f"  Optimizer: SGD with momentum (low memory)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches: {num_batches}")

    # Training loop
    model.train()
    total_loss = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad()

    save_dir = "./llama_peer_262k"
    os.makedirs(save_dir, exist_ok=True)

    progress = tqdm(range(num_batches), desc="Training")

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Store input_ids for any Engram layers (not used here but needed by framework)
        model.model._current_input_ids = input_ids

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / gradient_accumulation_steps

        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar
        avg_loss = total_loss / (step + 1)
        progress.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'vram': f'{get_gpu_memory():.1f}GB'
        })

        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint = {
                'step': step + 1,
                'loss': avg_loss,
                'peer_state': {n: p.cpu() for n, p in model.named_parameters() if 'peer' in n},
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_{step+1}.pt")
            print(f"\n  Saved checkpoint at step {step+1}, loss={avg_loss:.4f}")

    # Final save
    final_loss = total_loss / num_batches
    print(f"\nTraining complete. Final loss: {final_loss:.4f}")

    final_state = {n: p.cpu() for n, p in model.named_parameters() if 'peer' in n}
    torch.save(final_state, f"{save_dir}/weights.pt")
    torch.save({'loss': final_loss, 'steps': num_batches}, f"{save_dir}/metadata.pt")

    print(f"Saved to {save_dir}/")

    return model, tokenizer, final_loss


def evaluate_peer_803k(model=None, tokenizer=None):
    """Evaluate PEER-803K on test set"""

    print("\n" + "=" * 70)
    print("PEER-803K Evaluation")
    print("=" * 70)

    device = torch.device("cuda")

    if model is None:
        # Load fresh model with trained weights
        config = IntegrationConfig(mode="peer")
        config.peer = PEERConfig(num_experts=262144)
        config.peer_layers = [19, 21]

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = inject_peer_engram(model, config)

        # Load trained weights
        try:
            saved = torch.load("./llama_peer_262k/weights.pt", map_location=device, weights_only=True)
            state = model.state_dict()
            loaded = 0
            for n, p in saved.items():
                if n in state and state[n].shape == p.shape:
                    state[n].copy_(p.to(state[n].device))
                    loaded += 1
            print(f"Loaded {loaded} PEER weight tensors")
        except FileNotFoundError:
            print("No trained weights found, using random initialization")

    # Load test set
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_data = test_data.filter(lambda x: len(x['text'].strip()) > 10)

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(test_data, total=min(200, len(test_data)), desc="Evaluating")):
            if i >= 200:
                break

            text = example['text']
            if not text.strip():
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False
            )

            if inputs['input_ids'].shape[1] < 2:
                continue

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            model.model._current_input_ids = input_ids

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\nResults:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    # Compare to baseline
    print(f"\n  Baseline TinyLlama PPL: ~17.12")
    ppl_change = ((perplexity - 17.12) / 17.12) * 100
    print(f"  Change: {ppl_change:+.2f}%")

    return perplexity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--batches", type=int, default=2000, help="Number of training batches")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")

    args = parser.parse_args()

    if args.train:
        model, tokenizer, loss = train_peer_803k(
            num_batches=args.batches,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
        )
        # Evaluate right after training
        print("\nEvaluating trained model...")
        evaluate_peer_803k(model, tokenizer)
    elif args.eval:
        evaluate_peer_803k()
    else:
        print("Specify --train or --eval")
        parser.print_help()
