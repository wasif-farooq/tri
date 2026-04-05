"""
Async Prefetching Module
Location: torch-range-indexed/src/prefetch.py
"""

import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
import torch


class PrefetchManager:
    """
    Manages async prefetching of weight groups.
    
    This class handles the coordination between the inference engine
    and the background prefetch thread.
    """
    
    def __init__(self, nvme_loader: Callable, max_queue_size: int = 100):
        """
        Args:
            nvme_loader: Function that takes (offset, size) and returns weights tensor
            max_queue_size: Maximum number of pending prefetch requests
        """
        self.nvme_loader = nvme_loader
        self.prefetch_queue = queue.Queue(maxsize=max_queue_size)
        self.prefetched = set()
        self.loading = set()  # Currently loading
        self.lock = threading.RLock()
        
        # Statistics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_requests = 0
        
        # Start worker
        self._stop = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
    
    def _worker_loop(self):
        """Background thread that actually loads weights"""
        while not self._stop:
            try:
                group_id, offset, size, callback = self.prefetch_queue.get(timeout=0.1)
                
                with self.lock:
                    self.loading.add(group_id)
                
                # Load from NVMe
                weights = self.nvme_loader(offset, size)
                
                with self.lock:
                    self.loading.discard(group_id)
                    if weights is not None:
                        self.prefetched.add(group_id)
                
                # Callback if provided
                if callback:
                    callback(group_id, weights)
                
                self.prefetch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch worker error: {e}")
    
    def prefetch(self, group_id: int, offset: int, size: int, callback=None):
        """
        Request a weight group to be prefetched.
        
        Args:
            group_id: Unique identifier for the group
            offset: Offset in NVMe file
            size: Size of the group in bytes
            callback: Optional callback when loading completes
        """
        with self.lock:
            if group_id in self.prefetched or group_id in self.loading:
                return
            
            self.prefetch_requests += 1
            self.prefetch_queue.put((group_id, offset, size, callback))
    
    def mark_used(self, group_id: int) -> bool:
        """
        Mark that a group was actually used.
        Returns True if it was prefetched (hit), False otherwise.
        """
        with self.lock:
            if group_id in self.prefetched:
                self.prefetched.discard(group_id)
                self.prefetch_hits += 1
                return True
            else:
                self.prefetch_misses += 1
                return False
    
    def get_stats(self) -> Dict:
        """Get prefetch statistics"""
        total = self.prefetch_hits + self.prefetch_misses
        return {
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'prefetch_hit_rate': self.prefetch_hits / total if total > 0 else 0,
            'prefetch_requests': self.prefetch_requests,
            'queue_size': self.prefetch_queue.qsize(),
            'prefetched_count': len(self.prefetched),
            'loading_count': len(self.loading),
        }
    
    def shutdown(self):
        """Stop the prefetch worker"""
        self._stop = True
        if self._worker.is_alive():
            self._worker.join(timeout=5)