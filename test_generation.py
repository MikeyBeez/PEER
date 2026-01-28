#!/usr/bin/env python3
"""Test generation with trained PEER/Engram models"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_peer_engram import IntegrationConfig, inject_peer_engram

def test_generation(mode: str, prompts: list):
    print(f"\n{'='*70}")
    print(f"Testing {mode.upper()} mode")
    print('='*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model
    config = IntegrationConfig(mode=mode)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    # Inject modules
    model = inject_peer_engram(model, config)
    
    # Load trained weights
    weights_path = f"./llama_{mode}_finetuned/peer_engram_weights.pt"
    try:
        saved_weights = torch.load(weights_path, map_location=device, weights_only=True)

        # Load weights into model (only matching shapes)
        model_state = model.state_dict()
        loaded = 0
        skipped = 0
        for name, param in saved_weights.items():
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded += 1
                else:
                    skipped += 1
        print(f"Loaded {loaded} weight tensors from {weights_path}")
        if skipped:
            print(f"Skipped {skipped} tensors (shape mismatch)")
    except FileNotFoundError:
        print(f"No trained weights found at {weights_path}, using initialized weights")
    
    model.eval()
    
    # Store input_ids for Engram
    if hasattr(model, 'model'):
        model.model._current_input_ids = None
    
    # Generate for each prompt
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        if hasattr(model, 'model'):
            model.model._current_input_ids = inputs['input_ids']
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.1,
                )
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Output: {output}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

def main():
    prompts = [
        "The history of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The best way to learn programming is",
        "In the year 2050, humanity discovered",
    ]
    
    # Test base model first for comparison
    print("\n" + "="*70)
    print("Testing BASE model (no modifications)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    base_model.eval()
    
    for prompt in prompts[:2]:  # Just 2 for base
        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = base_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Output: {output}")
    
    del base_model
    torch.cuda.empty_cache()
    
    # Test each mode
    for mode in ['engram', 'peer', 'hybrid']:
        test_generation(mode, prompts[:2])  # 2 prompts each

if __name__ == "__main__":
    main()
