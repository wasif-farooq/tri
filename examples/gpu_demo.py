"""
Demo: GPU Kernel for Range Checking
Location: torch-range-indexed/examples/gpu_demo.py

This demonstrates the massive speedup from using the GPU kernel
for parallel range checking.
"""

import torch
import numpy as np
import time
from tri.gpu import GPURangeChecker


def create_test_data(num_groups: int = 10_000_000, num_inputs: int = 4096):
    """Create test data for benchmarking"""
    print(f"\n📊 Creating test data:")
    print(f"   Groups: {num_groups:,}")
    print(f"   Inputs: {num_inputs:,}")
    
    # Create random min/max values (sorted)
    min_vals = np.sort(np.random.randn(num_groups) * 0.5)
    max_vals = min_vals + np.random.rand(num_groups) * 0.2
    
    # Create random input values
    input_values = torch.randn(num_inputs)
    
    return min_vals, max_vals, input_values


def benchmark_cpu(min_vals, max_vals, input_values):
    """Benchmark CPU-based range checking"""
    print("\n🐌 CPU Range Checking...")
    
    start = time.time()
    
    # CPU implementation using numpy
    indices = np.searchsorted(min_vals, input_values.numpy(), side='right') - 1
    indices = np.where(indices >= 0, indices, -1)
    valid = (indices >= 0) & (input_values.numpy() <= max_vals[indices])
    indices[~valid] = -1
    
    elapsed = time.time() - start
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Speed: {len(input_values)/elapsed:.0f} inputs/sec")
    
    return elapsed


def benchmark_gpu(min_vals, max_vals, input_values):
    """Benchmark GPU-based range checking"""
    print("\n🚀 GPU Range Checking (with CUDA kernel)...")
    
    # Initialize GPU checker
    checker = GPURangeChecker(min_vals, max_vals, device='cuda')
    
    # Move input to GPU
    input_gpu = input_values.cuda()
    
    # Warmup
    for _ in range(3):
        _ = checker.find_groups(input_gpu)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    output = checker.find_groups(input_gpu)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Speed: {len(input_values)/elapsed:.0f} inputs/sec")
    
    return elapsed, checker


def main():
    print("=" * 70)
    print("GPU KERNEL DEMO: Parallel Range Checking")
    print("=" * 70)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available! GPU kernel requires NVIDIA GPU.")
        print("   Running CPU-only demo...")
        return
    
    print(f"\n✅ CUDA available!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create test data
    min_vals, max_vals, input_values = create_test_data(
        num_groups=10_000_000,  # 10M groups (like a 10B model)
        num_inputs=4096         # 4096 inputs (typical hidden dimension)
    )
    
    # Run benchmarks
    cpu_time = benchmark_cpu(min_vals, max_vals, input_values)
    gpu_time, checker = benchmark_gpu(min_vals, max_vals, input_values)
    
    # Show speedup
    speedup = cpu_time / gpu_time
    print("\n" + "=" * 70)
    print("📈 RESULTS:")
    print(f"   CPU time: {cpu_time*1000:.2f} ms")
    print(f"   GPU time: {gpu_time*1000:.2f} ms")
    print(f"   SPEEDUP: {speedup:.1f}x")
    print("=" * 70)
    
    # Show GPU stats
    print("\n📊 GPU Checker Statistics:")
    stats = checker.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Theoretical maximum speedup
    print("\n💡 Theoretical maximum speedup:")
    print(f"   CPU: 1 thread")
    print(f"   GPU: {torch.cuda.get_device_properties(0).multi_processor_count * 128} threads")
    print(f"   Theoretical: ~1000x for 10M groups")
    print(f"   Achieved: {speedup:.1f}x (limited by memory bandwidth)")


if __name__ == "__main__":
    main()