"""
Range-indexed linear layers with LRU+TTL cache and prefetching
Location: torch-range-indexed/src/layers.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import time

from .cache import LRUTTLCacheWithPrefetch


class RangeIndexedLinear(nn.Module):
    """
    Linear layer with LRU+TTL cache and async prefetching.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        metadata: List[Dict],
        weight_file_path: str,
        cache: LRUTTLCacheWithPrefetch,
        device: str = 'cuda',
        use_prefetch: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.use_prefetch = use_prefetch
        
        # Build sorted metadata
        sorted_metadata = sorted(metadata, key=lambda x: x['min'])
        self.min_vals = np.array([m['min'] for m in sorted_metadata])
        self.max_vals = np.array([m['max'] for m in sorted_metadata])
        self.group_to_metadata = {i: m for i, m in enumerate(sorted_metadata)}
        
        # Store group metadata for prefetch
        for group_id, meta in self.group_to_metadata.items():
            cache.set_group_metadata(group_id, meta['offset'], meta['size'] * 4)
        
        self.cache = cache
        self.weight_file_path = weight_file_path
        self.weight_file = None
        
        # Statistics
        self.range_checks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Open weight file
        self._open_weight_file()
    
    def _open_weight_file(self):
        """Open weight file (lazy)"""
        if self.weight_file is None:
            self.weight_file = open(self.weight_file_path, 'rb')
    
    def _find_group(self, input_val: float) -> Optional[Dict]:
        """Find group containing input value using binary search"""
        self.range_checks += 1
        
        # Binary search
        left, right = 0, len(self.min_vals) - 1
        while left <= right:
            mid = (left + right) // 2
            if input_val < self.min_vals[mid]:
                right = mid - 1
            elif input_val > self.max_vals[mid]:
                left = mid + 1
            else:
                return self.group_to_metadata[mid]
        
        return None
    
    def _load_from_nvme(self, offset: int, size: int) -> torch.Tensor:
        """Load weights from NVMe file"""
        self._open_weight_file()
        self.weight_file.seek(offset)
        weight_bytes = self.weight_file.read(size)
        weights = np.frombuffer(weight_bytes, dtype=np.float32)
        return torch.from_numpy(weights).to(self.device)
    
    def _get_weight(self, input_val: float, input_pos: int, group_meta: Dict) -> torch.Tensor:
        """Get weight using cache and prefetch"""
        group_id = group_meta['group_idx']
        
        # Record access for prefetch learning
        if self.use_prefetch:
            self.cache.record_access(group_id)
        
        # Try cache
        weights = self.cache.get(group_id)
        
        if weights is None:
            self.cache_misses += 1
            # Load from NVMe (cache miss)
            weights = self._load_from_nvme(group_meta['offset'], group_meta['size'] * 4)
            self.cache.put(group_id, weights)
        else:
            self.cache_hits += 1
        
        # Extract specific weight
        pos_in_group = input_pos - group_meta['start_pos']
        if 0 <= pos_in_group < len(weights):
            return weights[pos_in_group]
        
        return torch.tensor(0.0, device=self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with cached weights"""
        batch_size, dim = x.shape
        output = torch.zeros(batch_size, self.out_features, device=self.device)
        
        # For each output neuron
        for out_idx in range(self.out_features):
            neuron_metadata = [m for m in self.group_to_metadata.values() 
                               if m.get('output_idx') == out_idx]
            
            if not neuron_metadata:
                continue
            
            for in_idx in range(dim):
                input_val = x[0, in_idx].item() if batch_size == 1 else x[:, in_idx].mean().item()
                group_meta = self._find_group(input_val)
                
                if group_meta is None:
                    continue
                
                weight = self._get_weight(input_val, in_idx, group_meta)
                output[0, out_idx] += input_val * weight
        
        return output
    
    def get_stats(self) -> Dict:
        """Get cache and prefetch statistics"""
        total = self.cache_hits + self.cache_misses
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / total if total > 0 else 0,
            'range_checks': self.range_checks,
            'cache_stats': self.cache.get_full_stats() if hasattr(self.cache, 'get_full_stats') else self.cache.get_stats(),
        }
    
    def close(self):
        """Close file handle"""
        if self.weight_file:
            self.weight_file.close()
            self.weight_file = None
    
    def __del__(self):
        self.close()