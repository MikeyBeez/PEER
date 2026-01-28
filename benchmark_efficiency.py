#!/usr/bin/env python3
"""
PEER Efficiency Benchmark
=========================

Measures the key value proposition of PEER: memory efficiency.

Metrics:
- Peak VRAM at various expert counts
- Perplexity at each scale
- Perplexity per GB of VRAM
- Parameter count vs memory footprint
- Throughput (tokens/second)

Usage:
    python benchmark_efficiency.py
    python benchmark_efficiency.py --expert-counts 16384 65536 262144
    python benchmark_efficiency.py --quick  # Fast mode with fewer samples
"""

import argparse
import gc
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results for a single configuration"""
    name: str
    num_experts: int
    num_peer_layers: int

    # Memory metrics
    peak_vram_gb: float
    model_size_gb: float
    peer_params: int
    total_params: int

    # Quality metrics
    perplexity: Optional[float] = None
    avg_loss: Optional[float] = None

    # Throughput metrics
    tokens_per_second: Optional[float] = None

    # Derived metrics
    @property
    def perplexity_per_gb(self) -> Optional[float]:
        if self.perplexity and self.peak_vram_gb > 0:
            return self.perplexity / self.peak_vram_gb
        return None

    @property
    def params_per_gb(self) -> float:
        if self.peak_vram_gb > 0:
            return self.total_params / self.peak_vram_gb / 1e9
        return 0


def get_gpu_memory_gb() -> float:
    """Get current GPU memory allocated in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_peak_gpu_memory_gb() -> float:
    """Get peak GPU memory allocated in GB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0


def reset_peak_memory():
    """Reset peak memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def count_parameters(model) -> Dict[str, int]:
    """Count parameters by module type"""
    total = 0
    peer_params = 0
    engram_params = 0
    backbone_params = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if 'peer' in name.lower():
            peer_params += count
        elif 'engram' in name.lower():
            engram_params += count
        else:
            backbone_params += count

    return {
        'total': total,
        'peer': peer_params,
        'engram': engram_params,
        'backbone': backbone_params,
    }


def evaluate_perplexity(model, tokenizer, dataset, device, max_samples=100, max_length=256):
    """Calculate perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)),
                                         desc="  Perplexity", leave=False)):
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
            if hasattr(model, 'model'):
                model.model._current_input_ids = input_ids

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf'), float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def measure_throughput(model, tokenizer, device, num_tokens=1000, batch_size=4, seq_len=128):
    """Measure inference throughput in tokens/second"""
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    if hasattr(model, 'model'):
        model.model._current_input_ids = input_ids

    # Warmup
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    num_iterations = max(1, num_tokens // (batch_size * seq_len))
    start_time = time.perf_counter()

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            for _ in range(num_iterations):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start_time

    total_tokens = num_iterations * batch_size * seq_len
    return total_tokens / elapsed


def benchmark_configuration(
    num_experts: int,
    peer_layers: List[int],
    tokenizer,
    dataset,
    device,
    max_perplexity_samples: int = 100,
    measure_throughput_flag: bool = True,
) -> BenchmarkResult:
    """Benchmark a single PEER configuration"""
    from transformers import AutoModelForCausalLM
    from llama_peer_engram import IntegrationConfig, PEERConfig, inject_peer_engram

    name = f"PEER-{num_experts//1000}K" if num_experts >= 1000 else f"PEER-{num_experts}"
    print(f"\n  Benchmarking {name} ({len(peer_layers)} layers)...")

    reset_peak_memory()

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    # Configure PEER
    config = IntegrationConfig(mode="peer")
    config.peer = PEERConfig(
        num_experts=num_experts,
        experts_per_head=8,
        num_heads=4,
        dim_key=64,
    )
    config.peer_layers = peer_layers

    # Inject PEER
    model = inject_peer_engram(model, config)

    # Measure memory after model load
    model_size_gb = get_gpu_memory_gb()

    # Count parameters
    param_counts = count_parameters(model)

    # Run a forward pass to get peak memory
    dummy_input = torch.randint(0, 32000, (1, 128), device=device)
    if hasattr(model, 'model'):
        model.model._current_input_ids = dummy_input

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _ = model(input_ids=dummy_input, attention_mask=torch.ones_like(dummy_input))

    peak_vram_gb = get_peak_gpu_memory_gb()

    # Evaluate perplexity
    perplexity, avg_loss = evaluate_perplexity(
        model, tokenizer, dataset, device,
        max_samples=max_perplexity_samples
    )

    # Measure throughput
    tokens_per_second = None
    if measure_throughput_flag:
        tokens_per_second = measure_throughput(model, tokenizer, device)

    result = BenchmarkResult(
        name=name,
        num_experts=num_experts,
        num_peer_layers=len(peer_layers),
        peak_vram_gb=peak_vram_gb,
        model_size_gb=model_size_gb,
        peer_params=param_counts['peer'],
        total_params=param_counts['total'],
        perplexity=perplexity,
        avg_loss=avg_loss,
        tokens_per_second=tokens_per_second,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def benchmark_baseline(tokenizer, dataset, device, max_perplexity_samples: int = 100) -> BenchmarkResult:
    """Benchmark baseline TinyLlama without PEER"""
    from transformers import AutoModelForCausalLM

    print("\n  Benchmarking Baseline TinyLlama...")

    reset_peak_memory()

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    model_size_gb = get_gpu_memory_gb()
    param_counts = count_parameters(model)

    # Forward pass for peak memory
    dummy_input = torch.randint(0, 32000, (1, 128), device=device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _ = model(input_ids=dummy_input, attention_mask=torch.ones_like(dummy_input))

    peak_vram_gb = get_peak_gpu_memory_gb()

    perplexity, avg_loss = evaluate_perplexity(
        model, tokenizer, dataset, device,
        max_samples=max_perplexity_samples
    )

    tokens_per_second = measure_throughput(model, tokenizer, device)

    result = BenchmarkResult(
        name="Baseline",
        num_experts=0,
        num_peer_layers=0,
        peak_vram_gb=peak_vram_gb,
        model_size_gb=model_size_gb,
        peer_params=0,
        total_params=param_counts['total'],
        perplexity=perplexity,
        avg_loss=avg_loss,
        tokens_per_second=tokens_per_second,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def print_results(results: List[BenchmarkResult], baseline: BenchmarkResult):
    """Print benchmark results in a nice table"""
    print("\n" + "=" * 90)
    print("PEER EFFICIENCY BENCHMARK RESULTS")
    print("=" * 90)

    # Memory table
    print("\n📊 MEMORY EFFICIENCY")
    print("-" * 90)
    print(f"{'Config':<15} {'Experts':>10} {'PEER Params':>12} {'Total Params':>14} {'Peak VRAM':>12} {'Params/GB':>12}")
    print("-" * 90)

    print(f"{'Baseline':<15} {'-':>10} {'-':>12} {baseline.total_params/1e6:>11.1f}M {baseline.peak_vram_gb:>10.2f} GB {baseline.params_per_gb:>10.1f}M")

    for r in results:
        experts_str = f"{r.num_experts//1000}K" if r.num_experts >= 1000 else str(r.num_experts)
        peer_params_str = f"{r.peer_params/1e6:.1f}M"
        total_params_str = f"{r.total_params/1e6:.1f}M"
        print(f"{r.name:<15} {experts_str:>10} {peer_params_str:>12} {total_params_str:>14} {r.peak_vram_gb:>10.2f} GB {r.params_per_gb:>10.1f}M")

    # Quality table
    print("\n📈 QUALITY vs EFFICIENCY")
    print("-" * 90)
    print(f"{'Config':<15} {'Perplexity':>12} {'vs Base':>10} {'VRAM':>10} {'PPL/GB':>10} {'Tok/sec':>12}")
    print("-" * 90)

    print(f"{'Baseline':<15} {baseline.perplexity:>12.2f} {'-':>10} {baseline.peak_vram_gb:>8.2f}GB {'-':>10} {baseline.tokens_per_second:>10.0f}")

    for r in results:
        ppl_diff = ((r.perplexity - baseline.perplexity) / baseline.perplexity) * 100
        ppl_diff_str = f"{ppl_diff:+.1f}%"
        ppl_per_gb = r.perplexity / r.peak_vram_gb if r.peak_vram_gb > 0 else 0
        tps = r.tokens_per_second if r.tokens_per_second else 0
        print(f"{r.name:<15} {r.perplexity:>12.2f} {ppl_diff_str:>10} {r.peak_vram_gb:>8.2f}GB {ppl_per_gb:>10.2f} {tps:>10.0f}")

    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 90)

    if results:
        max_experts = max(r.num_experts for r in results)
        max_experts_result = [r for r in results if r.num_experts == max_experts][0]

        param_ratio = max_experts_result.total_params / baseline.total_params
        vram_ratio = max_experts_result.peak_vram_gb / baseline.peak_vram_gb

        print(f"• Largest config ({max_experts_result.name}): {param_ratio:.1f}x parameters, {vram_ratio:.1f}x VRAM")
        print(f"• Parameter efficiency: {param_ratio/vram_ratio:.2f}x more params per GB of VRAM")

        if max_experts_result.perplexity and baseline.perplexity:
            ppl_change = ((max_experts_result.perplexity - baseline.perplexity) / baseline.perplexity) * 100
            print(f"• Perplexity change: {ppl_change:+.1f}% (goal: maintain quality while scaling)")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="PEER Efficiency Benchmark")
    parser.add_argument("--expert-counts", type=int, nargs="+",
                        default=[16384, 65536, 262144],
                        help="Expert counts to benchmark (must be perfect squares)")
    parser.add_argument("--peer-layers", type=int, nargs="+", default=[19, 21],
                        help="Which layers to add PEER to")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer perplexity samples")
    parser.add_argument("--no-throughput", action="store_true",
                        help="Skip throughput measurement")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples for perplexity evaluation")

    args = parser.parse_args()

    # Validate expert counts are perfect squares
    for count in args.expert_counts:
        sqrt = int(math.sqrt(count))
        if sqrt * sqrt != count:
            print(f"Error: {count} is not a perfect square")
            return

    print("=" * 90)
    print("PEER EFFICIENCY BENCHMARK")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer and dataset
    print("\nLoading tokenizer and dataset...")
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 10)

    max_samples = 50 if args.quick else args.max_samples
    print(f"Perplexity samples: {max_samples}")

    # Benchmark baseline
    print("\n" + "=" * 70)
    print("BASELINE BENCHMARK")
    print("=" * 70)

    baseline = benchmark_baseline(tokenizer, dataset, device, max_samples)
    print(f"  Peak VRAM: {baseline.peak_vram_gb:.2f} GB")
    print(f"  Perplexity: {baseline.perplexity:.2f}")
    print(f"  Throughput: {baseline.tokens_per_second:.0f} tok/s")

    # Benchmark PEER configurations
    print("\n" + "=" * 70)
    print("PEER CONFIGURATIONS")
    print("=" * 70)

    results = []
    for num_experts in args.expert_counts:
        try:
            result = benchmark_configuration(
                num_experts=num_experts,
                peer_layers=args.peer_layers,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                max_perplexity_samples=max_samples,
                measure_throughput_flag=not args.no_throughput,
            )
            results.append(result)
            print(f"  Peak VRAM: {result.peak_vram_gb:.2f} GB")
            print(f"  Perplexity: {result.perplexity:.2f}")
            if result.tokens_per_second:
                print(f"  Throughput: {result.tokens_per_second:.0f} tok/s")
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️ OOM with {num_experts} experts - skipping")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  ⚠️ Error: {e}")

    # Print summary
    print_results(results, baseline)


if __name__ == "__main__":
    main()
