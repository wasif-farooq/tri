"""
LRU + TTL Cache with Async Prefetch Support
Location: torch-range-indexed/src/cache.py
"""

import time
import threading
import queue
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np


class LRUTTLCache:
    """
    LRU (Least Recently Used) cache with TTL (Time-To-Live) expiration.
    
    Features:
    - Automatically evicts least recently used items when full
    - Removes expired items based on TTL
    - Thread-safe for concurrent access
    - Tracks hit/miss statistics
    
    Args:
        max_size_mb: Maximum cache size in megabytes
        ttl_seconds: Time-to-live in seconds (default: 60)
        cleanup_interval_seconds: How often to run TTL cleanup (default: 30)
    """
    
    def __init__(self, max_size_mb: int = 8000, ttl_seconds: int = 60, 
                 cleanup_interval_seconds: int = 30):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval_seconds
        
        # Cache storage: group_key -> (weights, access_time, load_time)
        self.cache: OrderedDict = OrderedDict()
        self.current_size_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self._stop_cleanup = False
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        print(f"✅ LRU+TTL Cache initialized:")
        print(f"   Max size: {max_size_mb} MB")
        print(f"   TTL: {ttl_seconds} seconds")
        print(f"   Cleanup interval: {cleanup_interval} seconds")
    
    def get(self, key: Any) -> Optional[torch.Tensor]:
        """
        Retrieve weights from cache.
        
        Args:
            key: Unique identifier for the weight group
            
        Returns:
            Weight tensor if in cache and not expired, None otherwise
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            weights, access_time, load_time = self.cache[key]
            
            # Check TTL expiration
            if time.time() - access_time > self.ttl_seconds:
                # Expired! Remove it
                self._remove_entry(key)
                self.expirations += 1
                self.misses += 1
                return None
            
            # Update access time and move to end (most recent)
            self.cache.move_to_end(key)
            self.cache[key] = (weights, time.time(), load_time)
            self.hits += 1
            
            return weights
    
    def put(self, key: Any, weights: torch.Tensor):
        """
        Add weights to cache.
        
        Args:
            key: Unique identifier for the weight group
            weights: Weight tensor to cache
        """
        with self.lock:
            weight_size = weights.element_size() * weights.numel()
            
            # If key already exists, remove old entry first
            if key in self.cache:
                self._remove_entry(key)
            
            # Evict oldest entries until enough space
            while self.current_size_bytes + weight_size > self.max_size_bytes and self.cache:
                oldest_key, _ = self.cache.popitem(last=False)
                oldest_weights, _, _ = self.cache[oldest_key] if oldest_key in self.cache else (None, None, None)
                if oldest_weights is not None:
                    self.current_size_bytes -= oldest_weights.element_size() * oldest_weights.numel()
                self.evictions += 1
            
            # Add new entry
            self.cache[key] = (weights, time.time(), time.time())
            self.current_size_bytes += weight_size
    
    def _remove_entry(self, key: Any):
        """Remove a specific entry from cache"""
        if key in self.cache:
            weights, _, _ = self.cache[key]
            self.current_size_bytes -= weights.element_size() * weights.numel()
            del self.cache[key]
    
    def _cleanup_worker(self):
        """Background thread that removes expired entries"""
        while not self._stop_cleanup:
            time.sleep(self.cleanup_interval)
            self.cleanup_expired()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        with self.lock:
            now = time.time()
            expired_keys = []
            
            for key, (_, access_time, _) in self.cache.items():
                if now - access_time > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.expirations += 1
            
            return len(expired_keys)
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'num_entries': len(self.cache),
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'expirations': self.expirations,
                'ttl_seconds': self.ttl_seconds,
            }
    
    def shutdown(self):
        """Stop the cleanup thread"""
        self._stop_cleanup = True
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
    
    def __del__(self):
        self.shutdown()


class LRUTTLCacheWithPrefetch(LRUTTLCache):
    """
    LRU + TTL cache with intelligent async prefetching.
    
    Extends base cache with:
    - Background loading of predicted next weight groups
    - Multiple prefetch strategies (sequential, history, attention)
    - Prefetch hit tracking
    
    Args:
        max_size_mb: Maximum cache size in megabytes
        ttl_seconds: Time-to-live in seconds
        prefetch_ahead: Number of groups to prefetch (default: 3)
        prefetch_strategy: 'sequential', 'history', or 'attention' (default: 'history')
        nvme_loader: Function to load weights from NVMe
    """
    
    def __init__(self, max_size_mb: int = 8000, ttl_seconds: int = 60,
                 prefetch_ahead: int = 3, prefetch_strategy: str = 'history',
                 nvme_loader=None):
        super().__init__(max_size_mb, ttl_seconds)
        
        self.prefetch_ahead = prefetch_ahead
        self.prefetch_strategy = prefetch_strategy
        self.nvme_loader = nvme_loader
        
        # Access history for prediction
        self.access_history = []
        self.access_patterns = {}  # (prev, next) -> count
        self.max_history = 1000
        
        # Prefetch tracking
        self.prefetch_queue = queue.Queue()
        self.prefetched_keys = set()
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        # Group metadata (needed for loading)
        self.group_metadata = {}  # group_id -> (offset, size)
        
        # Start prefetch worker thread
        self._stop_prefetch = False
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
        print(f"✅ Async Prefetch enabled:")
        print(f"   Strategy: {prefetch_strategy}")
        print(f"   Prefetch ahead: {prefetch_ahead}")
    
    def set_group_metadata(self, group_id: int, offset: int, size: int):
        """Store metadata for a group (needed for loading)"""
        self.group_metadata[group_id] = (offset, size)
    
    def _prefetch_worker(self):
        """Background thread that loads predicted groups"""
        while not self._stop_prefetch:
            try:
                # Wait for prefetch tasks (non-blocking with timeout)
                try:
                    group_id = self.prefetch_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check if still needed (not already in cache)
                with self.lock:
                    if group_id in self.cache:
                        self.prefetch_queue.task_done()
                        continue
                
                # Load from NVMe
                if self.nvme_loader and group_id in self.group_metadata:
                    offset, size = self.group_metadata[group_id]
                    weights = self.nvme_loader(offset, size)
                    
                    # Add to cache (may evict)
                    if weights is not None:
                        self.put(group_id, weights)
                        with self.lock:
                            self.prefetched_keys.add(group_id)
                
                self.prefetch_queue.task_done()
                
            except Exception as e:
                print(f"Prefetch worker error: {e}")
    
    def record_access(self, group_id: int):
        """
        Record a group access for pattern learning.
        Called when a group is actually used.
        """
        with self.lock:
            # Record for history
            if len(self.access_history) > 0:
                prev_group = self.access_history[-1]
                pattern = (prev_group, group_id)
                self.access_patterns[pattern] = self.access_patterns.get(pattern, 0) + 1
            
            self.access_history.append(group_id)
            
            # Keep history size bounded
            while len(self.access_history) > self.max_history:
                oldest = self.access_history.pop(0)
                # Clean up patterns involving oldest
                to_remove = [p for p in self.access_patterns if p[0] == oldest or p[1] == oldest]
                for p in to_remove:
                    del self.access_patterns[p]
            
            # Check if this was a prefetch hit
            if group_id in self.prefetched_keys:
                self.prefetch_hits += 1
                self.prefetched_keys.discard(group_id)
            else:
                self.prefetch_misses += 1
            
            # Predict and queue next groups
            predicted = self._predict_next_groups(group_id)
            for pred_id in predicted:
                if pred_id not in self.cache and pred_id not in self.prefetched_keys:
                    self.prefetch_queue.put(pred_id)
    
    def _predict_next_groups(self, current_group: int) -> List[int]:
        """
        Predict next groups based on selected strategy.
        
        Strategies:
        - sequential: Just load next N groups (simple)
        - history: Use recorded access patterns (medium)
        - attention: Use attention scores (advanced, requires attention data)
        """
        if self.prefetch_strategy == 'sequential':
            return self._predict_sequential(current_group)
        elif self.prefetch_strategy == 'history':
            return self._predict_from_history(current_group)
        elif self.prefetch_strategy == 'attention':
            return self._predict_from_attention(current_group)
        else:
            return self._predict_sequential(current_group)
    
    def _predict_sequential(self, current_group: int) -> List[int]:
        """Simple sequential prediction"""
        predicted = []
        for i in range(1, self.prefetch_ahead + 1):
            next_group = current_group + i
            if next_group in self.group_metadata:
                predicted.append(next_group)
        return predicted
    
    def _predict_from_history(self, current_group: int) -> List[int]:
        """Predict based on historical access patterns"""
        # Find most common next groups for this current group
        candidates = []
        for (prev, next_g), count in self.access_patterns.items():
            if prev == current_group:
                candidates.append((next_g, count))
        
        # Sort by frequency
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        predicted = [c[0] for c in candidates[:self.prefetch_ahead]]
        
        # Fallback to sequential if not enough history
        if len(predicted) < self.prefetch_ahead:
            sequential = self._predict_sequential(current_group)
            for s in sequential:
                if s not in predicted:
                    predicted.append(s)
                    if len(predicted) >= self.prefetch_ahead:
                        break
        
        return predicted[:self.prefetch_ahead]
    
    def _predict_from_attention(self, current_group: int) -> List[int]:
        """
        Predict based on attention scores.
        This requires attention data from the model.
        """
        # Placeholder - actual implementation would use attention scores
        # For now, fall back to history
        return self._predict_from_history(current_group)
    
    def update_attention_scores(self, attention_scores: torch.Tensor):
        """
        Update attention scores for better prediction.
        Called during forward pass if attention data is available.
        """
        # Store attention scores for prediction
        # Implementation depends on model architecture
        pass
    
    def get_prefetch_stats(self) -> Dict:
        """Get prefetch-specific statistics"""
        total = self.prefetch_hits + self.prefetch_misses
        hit_rate = self.prefetch_hits / total if total > 0 else 0
        
        return {
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'prefetch_hit_rate': hit_rate,
            'prefetch_queue_size': self.prefetch_queue.qsize(),
            'prefetch_strategy': self.prefetch_strategy,
            'prefetch_ahead': self.prefetch_ahead,
        }
    
    def get_full_stats(self) -> Dict:
        """Get complete statistics"""
        stats = self.get_stats()
        stats.update(self.get_prefetch_stats())
        return stats
    
    def shutdown(self):
        """Stop background threads"""
        self._stop_prefetch = True
        super().shutdown()
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5)