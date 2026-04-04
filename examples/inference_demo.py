"""Example: Run inference with TRI model"""

import torch
from tri import RangeIndexedModel, get_device_info


def main():
    print("🚀 Running TRI inference demo...")
    
    # Check device
    device_info = get_device_info()
    print(f"Device: {device_info['devices'][0]['name']}")
    print(f"VRAM: {device_info['devices'][0].get('vram_mb', 0):.0f} MB")
    
    # Load model (only metadata in VRAM!)
    print("\n📦 Loading model (metadata only)...")
    model = RangeIndexedModel(
        "llama-7b-tri",
        cache_size_mb=2000,  # 2GB cache
        use_prefetch=True
    )
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    print(f"\n💬 Prompt: {prompt}")
    
    print("\n🔄 Generating...")
    output = model.generate(prompt, max_length=100, temperature=0.8)
    
    print(f"\n✨ Output: {output}")
    
    # Show statistics
    stats = model.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"   Hits: {stats['cache_hits']:,}")
    print(f"   Misses: {stats['cache_misses']:,}")
    
    model.close()


if __name__ == "__main__":
    main()