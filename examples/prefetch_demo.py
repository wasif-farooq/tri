"""
Demo: LRU+TTL Cache with Prefetching
Location: torch-range-indexed/examples/prefetch_demo.py
"""

import torch
import time
from tri.model import RangeIndexedModel


def main():
    print("=" * 60)
    print("LRU+TTL CACHE WITH PREFETCHING DEMO")
    print("=" * 60)
    
    # Load model with different prefetch strategies
    strategies = ['sequential', 'history']
    
    for strategy in strategies:
        print(f"\n\n{'='*60}")
        print(f"Testing prefetch strategy: {strategy.upper()}")
        print("=" * 60)
        
        model = RangeIndexedModel(
            model_path="llama-7b-tri",  # Your converted model
            cache_size_mb=4000,
            ttl_seconds=60,
            prefetch_ahead=3,
            prefetch_strategy=strategy,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Warm up
        print("\n🔄 Warming up...")
        test_input = torch.randn(1, 4096).to(model.device)
        
        # Measure performance
        tokens = []
        times = []
        
        for i in range(100):
            start = time.time()
            output = model(test_input)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i % 20 == 0:
                print(f"   Token {i}: {elapsed*1000:.2f} ms")
        
        # Statistics
        avg_time = sum(times) * 1000 / len(times)
        tps = 1000 / avg_time
        
        print(f"\n📊 Results ({strategy}):")
        print(f"   Average time: {avg_time:.2f} ms/token")
        print(f"   Tokens/second: {tps:.1f} t/s")
        
        model.print_stats()
        model.close()


if __name__ == "__main__":
    main()