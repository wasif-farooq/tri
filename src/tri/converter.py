"""Model converter for range-indexed format"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import struct


class RangeIndexedConverter:
    """
    Converts PyTorch models to range-indexed format.
    
    This is a one-time conversion that prepares your model for
    efficient inference with the TRI system.
    
    Args:
        group_size: Number of weights per index group (default: 1000)
        dtype: Data type for weights (default: torch.float16)
    
    Example:
        >>> converter = RangeIndexedConverter(group_size=1000)
        >>> converter.convert_model(model, "llama-7b-tri")
        Converting model...
        Total weights: 7,000,000,000
        Total groups: 7,000,000
        Metadata size: 140.00 MB
        Weights size: 14.00 GB
    """
    
    def __init__(self, group_size: int = 1000, dtype: torch.dtype = torch.float16):
        self.group_size = group_size
        self.dtype = dtype
        self.bytes_per_weight = 2 if dtype == torch.float16 else 4
        
    def convert_model(self, model: nn.Module, output_path: str, 
                      device: str = 'cpu') -> Dict:
        """
        Convert a PyTorch model to range-indexed format.
        
        Args:
            model: PyTorch model to convert
            output_path: Base path for output files (no extension)
            device: Device to use for conversion ('cpu' or 'cuda')
            
        Returns:
            Dictionary with conversion statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Converting model to range-indexed format...")
        print(f"   Group size: {self.group_size} weights/group")
        print(f"   Data type: {self.dtype}")
        
        model = model.to(device)
        model.eval()
        
        metadata = []
        weight_file = open(f"{output_path}.weights.bin", "wb")
        
        total_weights = 0
        total_groups = 0
        
        # Process each layer
        for name, module in tqdm(model.named_modules(), desc="Converting layers"):
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
                weights = self._extract_weights(module, device)
                total_weights += len(weights)
                
                num_groups = (len(weights) + self.group_size - 1) // self.group_size
                total_groups += num_groups
                
                for group_idx in range(num_groups):
                    start = group_idx * self.group_size
                    end = min(start + self.group_size, len(weights))
                    group_weights = weights[start:end]
                    
                    # Calculate statistics
                    min_val = float(np.min(group_weights))
                    max_val = float(np.max(group_weights))
                    mean_val = float(np.mean(group_weights))
                    std_val = float(np.std(group_weights))
                    
                    # Write to NVMe
                    offset = weight_file.tell()
                    weight_file.write(group_weights.astype(np.float32).tobytes())
                    
                    # Store metadata
                    metadata.append({
                        'layer_name': name,
                        'layer_type': module.__class__.__name__,
                        'group_idx': group_idx,
                        'start_pos': start,
                        'end_pos': end,
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val,
                        'offset': offset,
                        'size': len(group_weights),
                        'shape': self._get_shape(module),
                    })
        
        weight_file.close()
        
        # Save metadata
        with open(f"{output_path}.metadata.pkl", "wb") as f:
            pickle.dump({
                'metadata': metadata,
                'group_size': self.group_size,
                'dtype': str(self.dtype),
                'total_weights': total_weights,
                'total_groups': total_groups,
                'version': '1.0.0'
            }, f)
        
        # Save index for fast loading
        self._build_fast_index(output_path, metadata)
        
        stats = {
            'total_weights': total_weights,
            'total_groups': total_groups,
            'metadata_size_mb': total_groups * 48 / 1024 / 1024,
            'weights_size_gb': total_weights * self.bytes_per_weight / 1024 / 1024 / 1024,
        }
        
        print(f"\n✅ Conversion complete!")
        print(f"   Total weights: {total_weights:,}")
        print(f"   Total groups: {total_groups:,}")
        print(f"   Metadata size: {stats['metadata_size_mb']:.2f} MB")
        print(f"   Weights size: {stats['weights_size_gb']:.2f} GB")
        
        return stats
    
    def _extract_weights(self, module: nn.Module, device: str) -> np.ndarray:
        """Extract and flatten weights from a module"""
        if hasattr(module, 'weight'):
            weights = module.weight.data.cpu().numpy().flatten()
        elif hasattr(module, 'embeddings'):
            weights = module.embeddings.data.cpu().numpy().flatten()
        else:
            weights = np.array([])
        return weights
    
    def _get_shape(self, module: nn.Module) -> Tuple[int, ...]:
        """Get shape of module's weights"""
        if hasattr(module, 'weight'):
            return module.weight.shape
        elif hasattr(module, 'embeddings'):
            return module.embeddings.shape
        return ()
    
    def _build_fast_index(self, output_path: Path, metadata: List[Dict]):
        """Build fast binary index for range lookups"""
        index_path = f"{output_path}.index.bin"
        with open(index_path, "wb") as f:
            for meta in metadata:
                # Pack metadata into binary for fast loading
                packed = struct.pack(
                    'ffffqq',
                    meta['min'],
                    meta['max'],
                    meta['mean'],
                    meta['std'],
                    meta['offset'],
                    meta['size']
                )
                f.write(packed)