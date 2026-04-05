"""
Range-indexed model with LRU+TTL cache and prefetching
Location: torch-range-indexed/src/model.py
"""

import torch
import torch.nn as nn
import pickle
from typing import Dict, List, Optional, Any

from .cache import LRUTTLCacheWithPrefetch
from .layers import RangeIndexedLinear
from .utils import get_device_info


class RangeIndexedModel(nn.Module):
    """
    Complete model with LRU+TTL cache and async prefetching.
    """
    
    def __init__(
        self,
        model_path: str,
        cache_size_mb: int = 8000,
        ttl_seconds: int = 60,
        prefetch_ahead: int = 3,
        prefetch_strategy: str = 'history',
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Load metadata
        with open(f"{model_path}.metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        # Initialize cache with LRU+TTL and prefetch
        self.cache = LRUTTLCacheWithPrefetch(
            max_size_mb=cache_size_mb,
            ttl_seconds=ttl_seconds,
            prefetch_ahead=prefetch_ahead,
            prefetch_strategy=prefetch_strategy,
            nvme_loader=None  # Will be set per layer
        )
        
        self.weight_file_path = f"{model_path}.weights.bin"
        
        # Group metadata by layer
        self.layer_metadata = {}
        for meta in self.metadata['metadata']:
            layer_name = meta['layer_name']
            if layer_name not in self.layer_metadata:
                self.layer_metadata[layer_name] = []
            self.layer_metadata[layer_name].append(meta)
        
        # Build layers
        self.layers = nn.ModuleList()
        for layer_name, layer_meta in self.layer_metadata.items():
            shape = layer_meta[0]['shape']
            if len(shape) == 2:
                linear_layer = RangeIndexedLinear(
                    in_features=shape[1],
                    out_features=shape[0],
                    metadata=layer_meta,
                    weight_file_path=self.weight_file_path,
                    cache=self.cache,
                    device=self.device,
                    use_prefetch=True
                )
                self.layers.append(linear_layer)
        
        print(f"✅ Model loaded with LRU+TTL cache + Prefetch")
        print(f"   Cache size: {cache_size_mb} MB")
        print(f"   TTL: {ttl_seconds} seconds")
        print(f"   Prefetch strategy: {prefetch_strategy}")
        print(f"   Prefetch ahead: {prefetch_ahead}")
        print(f"   Metadata: {self._get_metadata_size_mb():.2f} MB")
    
    def _get_metadata_size_mb(self) -> float:
        return len(self.metadata['metadata']) * 48 / 1024 / 1024
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_stats(self) -> Dict:
        """Get complete cache and prefetch statistics"""
        total_hits = 0
        total_misses = 0
        all_stats = {}
        
        for i, layer in enumerate(self.layers):
            stats = layer.get_stats()
            total_hits += stats['cache_hits']
            total_misses += stats['cache_misses']
            all_stats[f'layer_{i}'] = stats
        
        total = total_hits + total_misses
        all_stats['overall'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / total if total > 0 else 0,
        }
        
        # Add cache-wide stats
        all_stats['cache'] = self.cache.get_full_stats()
        
        return all_stats
    
    def print_stats(self):
        """Pretty print statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("CACHE & PREFETCH STATISTICS")
        print("=" * 60)
        
        overall = stats.get('overall', {})
        print(f"\n📊 Overall:")
        print(f"   Hit rate: {overall.get('overall_hit_rate', 0)*100:.1f}%")
        print(f"   Hits: {overall.get('total_hits', 0):,}")
        print(f"   Misses: {overall.get('total_misses', 0):,}")
        
        cache_stats = stats.get('cache', {})
        print(f"\n💾 Cache:")
        print(f"   Size: {cache_stats.get('size_mb', 0):.1f} MB / {cache_stats.get('max_size_mb', 0):.0f} MB")
        print(f"   Entries: {cache_stats.get('num_entries', 0):,}")
        print(f"   Evictions: {cache_stats.get('evictions', 0):,}")
        print(f"   Expirations: {cache_stats.get('expirations', 0):,}")
        
        print(f"\n🔮 Prefetch:")
        print(f"   Strategy: {cache_stats.get('prefetch_strategy', 'N/A')}")
        print(f"   Hit rate: {cache_stats.get('prefetch_hit_rate', 0)*100:.1f}%")
        print(f"   Hits: {cache_stats.get('prefetch_hits', 0):,}")
        print(f"   Misses: {cache_stats.get('prefetch_misses', 0):,}")
        print(f"   Queue size: {cache_stats.get('prefetch_queue_size', 0)}")
        
        print("=" * 60)
    
    def close(self):
        """Clean up resources"""
        for layer in self.layers:
            layer.close()
        self.cache.shutdown()
    
    def __del__(self):
        self.close()