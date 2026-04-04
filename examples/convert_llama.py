"""Example: Convert Llama model to TRI format"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tri import RangeIndexedConverter


def main():
    print("🦙 Converting Llama 2 7B to TRI format...")
    
    # Load the model
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert to TRI format
    converter = RangeIndexedConverter(group_size=1000, dtype=torch.float16)
    stats = converter.convert_model(model, "llama-7b-tri")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Model can now be run with {stats['metadata_size_mb']:.0f}MB VRAM")
    print(f"   Full weights stored in {stats['weights_size_gb']:.1f}GB file")
    
    # Save tokenizer too
    tokenizer.save_pretrained("llama-7b-tri-tokenizer")
    
    print("\n📝 Next steps:")
    print("   1. Run inference: python examples/inference_demo.py")
    print("   2. Benchmark: python examples/benchmark.py")


if __name__ == "__main__":
    main()