"""Cache system for weight groups"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any
import torch
import numpy as np


class LRUCache:
    """
    LRU (Least Recently Used) cache for weight groups in VRAM.
    
    This cache keeps frequently accessed weight groups in GPU memory,
    dramatically reducing NVMe reads for repeated patterns.
    
    Args:
        max_size_mb: Maximum cache size in megabytes
        
    Example:
        >>> cache = LRUCache(max_size_mb=4000)  # 4GB cache
        >>> cache.put(group_id, weights_tensor)
        >>> weights = cache.get(group_id)
    """
    
    def __init__(self, max_size_mb: int = 4000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[torch.Tensor]:
        """
        Retrieve weights from cache.
        
        Args:
            key: Unique identifier for the weight group
            
        Returns:
            Weight tensor if in cache, None otherwise
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: Any, weights: torch.Tensor):
        """
        Add weights to cache, evicting oldest if needed.
        
        Args:
            key: Unique identifier for the weight group
            weights: Weight tensor to cache
        """
        weight_size = weights.element_size() * weights.numel()
        
        # Evict until enough space
        while self.current_size + weight_size > self.max_size_bytes and self.cache:
            oldest_key, oldest_weights = self.cache.popitem(last=False)
            self.current_size -= oldest_weights.element_size() * oldest_weights.numel()
        
        # Add to cache
        self.cache[key] = weights
        self.current_size += weight_size
    
    def clear(self):
        """Clear all cached weights"""
        self.cache.clear()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self.hits + self.misses
        return {
            'size_mb': self.current_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'num_entries': len(self.cache),
            'hit_rate': self.hits / total if total > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
        }


class PredictivePrefetchCache(LRUCache):
    """
    Enhanced cache with predictive prefetching.
    
    This cache analyzes access patterns and preloads likely next
    weight groups, hiding NVMe latency.
    
    Args:
        max_size_mb: Maximum cache size in megabytes
        prefetch_ahead: Number of groups to prefetch (default: 3)
        history_size: Number of accesses to remember (default: 1000)
    """
    
    def __init__(self, max_size_mb: int = 4000, prefetch_ahead: int = 3, 
                 history_size: int = 1000):
        super().__init__(max_size_mb)
        self.prefetch_ahead = prefetch_ahead
        self.history_size = history_size
        self.access_history = []
        self.prefetched = set()
        self.prefetch_hits = 0
    
    def record_access(self, key: Any):
        """Record access pattern for prediction"""
        self.access_history.append(key)
        if len(self.access_history) > self.history_size:
            self.access_history = self.access_history[-self.history_size:]
    
    def predict_next_keys(self) -> List[Any]:
        """
        Predict next keys based on access patterns.
        
        Uses a simple sequential prediction for now.
        More sophisticated predictors can be plugged in.
        """
        if len(self.access_history) < 2:
            return []
        
        # Check for sequential pattern
        last_key = self.access_history[-1]
        second_last = self.access_history[-2]
        
        if isinstance(last_key, int) and isinstance(second_last, int):
            diff = last_key - second_last
            return [last_key + diff * i for i in range(1, self.prefetch_ahead + 1)]
        
        return []
    
    def get_prefetch_keys(self, current_key: Any) -> List[Any]:
        """Get keys to prefetch based on current access"""
        self.record_access(current_key)
        predicted = self.predict_next_keys()
        return [k for k in predicted if k not in self.cache and k not in self.prefetched]
    
    def mark_prefetched(self, key: Any):
        """Mark a key as prefetched"""
        self.prefetched.add(key)
    
    def get(self, key: Any) -> Optional[torch.Tensor]:
        """Get with prefetch hit tracking"""
        if key in self.prefetched:
            self.prefetch_hits += 1
            self.prefetched.discard(key)
        return super().get(key)
    
    def get_stats(self) -> Dict:
        """Get enhanced statistics"""
        stats = super().get_stats()
        stats['prefetch_hits'] = self.prefetch_hits
        stats['prefetch_hit_rate'] = self.prefetch_hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return stats