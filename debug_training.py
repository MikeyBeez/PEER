#!/usr/bin/env python3
"""Debug script to find where NaN occurs in Llama + Engram training"""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_peer_engram import (
    IntegrationConfig, inject_peer_engram, ModifiedLlamaDecoderLayer
)

def debug_training():
    print("=" * 70)
    print("Debugging Llama + Engram Training NaN")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    config = IntegrationConfig(mode='engram')
    print(f"\nLoading {config.model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,  # Better numerical range than fp16
        device_map="auto",
    )

    # Inject Engram
    print("\nInjecting Engram modules...")
    model = inject_peer_engram(model, config)

    # Enable debug on modified layers
    for module in model.modules():
        if isinstance(module, ModifiedLlamaDecoderLayer):
            module._debug = True

    # Setup optimizer with different LRs
    new_module_params = []
    embed_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'peer' in name or 'engram' in name:
            new_module_params.append(param)
        elif 'embed' in name or 'lm_head' in name:
            embed_params.append(param)
        else:
            backbone_params.append(param)

    lr = 1e-5  # Lower LR
    param_groups = [
        {'params': new_module_params, 'lr': lr},
        {'params': backbone_params, 'lr': lr * 0.01},
        {'params': embed_params, 'lr': lr * 0.001},
    ]
    optimizer = torch.optim.AdamW([g for g in param_groups if g['params']])
    trainable_params = new_module_params + backbone_params + embed_params

    # No GradScaler needed for bfloat16
    scaler = None

    print(f"\nTrainable params: {sum(p.numel() for p in trainable_params):,}")

    # Test data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test of the Engram module.",
        "Machine learning is transforming artificial intelligence.",
        "The weather today is sunny with a chance of rain.",
    ]

    model.train()

    for batch_idx in range(10):
        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx}")
        print("=" * 70)

        # Tokenize
        text = texts[batch_idx % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=64, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = input_ids.clone()

        print(f"Input: '{text[:50]}...'")
        print(f"Input IDs shape: {input_ids.shape}")

        # Check weights before forward
        print("\nChecking Engram weights before forward:")
        for name, param in model.named_parameters():
            if 'engram' in name and 'embedding' in name:
                has_nan = param.isnan().any().item()
                has_inf = param.isinf().any().item()
                pmin, pmax = param.min().item(), param.max().item()
                print(f"  {name.split('.')[-2]}.{name.split('.')[-1]}: NaN={has_nan}, Inf={has_inf}, range=[{pmin:.4f}, {pmax:.4f}]")
                if has_nan or has_inf:
                    break

        # Add hooks to trace NaN
        nan_locations = []
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                if out is not None and isinstance(out, torch.Tensor):
                    if out.isnan().any():
                        nan_locations.append(f"{name}: NaN in output")
                    if out.isinf().any():
                        nan_locations.append(f"{name}: Inf in output")
            return hook

        # Register hooks on key layers
        if hasattr(model, 'model'):
            hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
            for i, layer in enumerate(model.model.layers):
                hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

        # Forward with autocast (bfloat16)
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            # Remove hooks
            for h in hooks:
                h.remove()

            print(f"\nLoss: {loss.item():.4f}")
            print(f"Loss has NaN: {loss.isnan().any()}")

            if nan_locations:
                print(f"NaN detected at: {nan_locations}")

            if loss.isnan().any():
                print("NaN detected in loss!")
                break

            # Backward with scaler
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Check gradients
            nan_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.isnan().any():
                    nan_grads.append(name)

            if nan_grads:
                print(f"\nNaN gradients in: {nan_grads[:5]}...")
                break

            # Check gradient norms
            total_norm = 0
            for param in trainable_params:
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm:.4f}")

            if math.isnan(total_norm) or math.isinf(total_norm):
                print("NaN/Inf in gradient norm!")
                break

            # Clip gradients and step
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            # Check weights after step
            print("\nChecking key weights after optimizer step:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    has_nan = param.isnan().any().item()
                    has_inf = param.isinf().any().item()
                    if has_nan or has_inf:
                        print(f"  {name}: NaN={has_nan}, Inf={has_inf}")

            # Specifically check embed_tokens
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_w = model.model.embed_tokens.weight
                print(f"  embed_tokens: NaN={embed_w.isnan().any().item()}, Inf={embed_w.isinf().any().item()}, range=[{embed_w.min():.4f}, {embed_w.max():.4f}]")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Disable debug
    for module in model.modules():
        if isinstance(module, ModifiedLlamaDecoderLayer):
            module._debug = False

    print("\n" + "=" * 70)
    print("Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    debug_training()
