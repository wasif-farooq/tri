"""Range-indexed linear layers"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class RangeIndexedLinear(nn.Module):
    """
    Linear layer using range-indexed weights stored on NVMe.
    
    This is the core implementation of the range-indexed caching idea.
    Only metadata (min/max ranges) stays in VRAM; weights are loaded
    from NVMe on demand using range-based filtering.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        metadata: List of metadata dicts for this layer's weight groups
        weight_file_path: Path to NVMe weight file
        cache: Cache instance for weight groups
        device: Device to run on ('cuda' or 'cpu')
    
    Example:
        >>> layer = RangeIndexedLinear(4096, 4096, metadata, "model.weights.bin", cache)
        >>> output = layer(input_tensor)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        metadata: List[Dict],
        weight_file_path: str,
        cache: Any,
        device: str = 'cuda'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.metadata = metadata
        self.weight_file_path = weight_file_path
        self.cache = cache
        self.device = device
        
        # Build index for fast range lookup
        self._build_range_index()
        
        # Open weight file (kept open for performance)
        self.weight_file = open(weight_file_path, 'rb')
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.range_checks = 0
    
    def _build_range_index(self):
        """Build index for fast min/max range lookup"""
        # Sort metadata by min value for binary search
        self.metadata_sorted = sorted(self.metadata, key=lambda x: x['min'])
        self.min_values = [m['min'] for m in self.metadata_sorted]
        self.max_values = [m['max'] for m in self.metadata_sorted]
    
    def _find_group(self, input_val: float) -> Optional[Dict]:
        """
        Find which group contains this input value using binary search.
        
        This is the core range-checking algorithm!
        """
        self.range_checks += 1
        
        # Binary search on min values
        left, right = 0, len(self.min_values) - 1
        while left <= right:
            mid = (left + right) // 2
            if input_val < self.min_values[mid]:
                right = mid - 1
            elif input_val > self.max_values[mid]:
                left = mid + 1
            else:
                return self.metadata_sorted[mid]
        
        # Fallback: linear search (should rarely happen)
        for meta in self.metadata:
            if meta['min'] <= input_val <= meta['max']:
                return meta
        
        return None
    
    def _load_weight_group(self, group_meta: Dict) -> torch.Tensor:
        """Load a weight group from NVMe (cache miss path)"""
        self.weight_file.seek(group_meta['offset'])
        weight_bytes = self.weight_file.read(group_meta['size'] * 4)
        weights = np.frombuffer(weight_bytes, dtype=np.float32)
        weights_tensor = torch.from_numpy(weights).to(self.device)
        
        # Cache it
        group_key = (group_meta['layer_name'], group_meta['group_idx'])
        self.cache.put(hash(group_key), weights_tensor)
        
        return weights_tensor
    
    def _get_weight(self, input_val: float, input_pos: int) -> torch.Tensor:
        """
        Get weight for a specific input using range filtering.
        
        This is the main algorithm:
        1. Find group containing this input value (range check)
        2. Check cache for that group
        3. If not in cache, load from NVMe
        4. Return the specific weight
        """
        # Step 1: Find group via range check
        group_meta = self._find_group(input_val)
        if group_meta is None:
            return torch.tensor(0.0, device=self.device)
        
        # Step 2: Check cache
        group_key = hash((group_meta['layer_name'], group_meta['group_idx']))
        weights = self.cache.get(group_key)
        
        if weights is None:
            # Step 3: Cache miss - load from NVMe
            self.cache_misses += 1
            weights = self._load_weight_group(group_meta)
        else:
            # Step 4: Cache hit!
            self.cache_hits += 1
        
        # Step 5: Extract specific weight
        pos_in_group = input_pos - group_meta['start_pos']
        if 0 <= pos_in_group < len(weights):
            return weights[pos_in_group]
        
        return torch.tensor(0.0, device=self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using range-indexed weights.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=self.device)
        
        # For each output neuron
        for out_idx in range(self.out_features):
            # Get metadata for this output neuron's weights
            neuron_metadata = [m for m in self.metadata if m.get('output_idx') == out_idx]
            
            if not neuron_metadata:
                continue
            
            # For each input position
            for in_idx in range(self.in_features):
                input_val = x[0, in_idx].item() if batch_size == 1 else x[:, in_idx].mean().item()
                weight = self._get_weight(input_val, in_idx)
                output[0, out_idx] += input_val * weight
        
        return output
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / total if total > 0 else 0,
            'range_checks': self.range_checks,
        }
    
    def close(self):
        """Close the weight file handle"""
        if hasattr(self, 'weight_file'):
            self.weight_file.close()
    
    def __del__(self):
        self.close()