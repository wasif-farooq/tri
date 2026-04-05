"""
GPU Range Checker - Python wrapper for CUDA kernel
Location: torch-range-indexed/src/gpu/range_checker.py

This module provides GPU-accelerated range checking for your
range-indexed weight caching system.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import os
from pathlib import Path

# Try to import CUDA extension
try:
    from torch.utils.cpp_extension import load_inline
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class GPURangeChecker:
    """
    GPU-accelerated range checker for finding which weight group
    each input value belongs to.
    
    This is the core optimization for your range-indexed system.
    It runs on the GPU and processes thousands of inputs in parallel.
    
    Usage:
        # Initialize with metadata
        checker = GPURangeChecker(min_vals, max_vals)
        
        # Find groups for all inputs in parallel
        groups = checker.find_groups(input_tensor)  # Super fast!
    """
    
    def __init__(self, min_vals: np.ndarray, max_vals: np.ndarray, device: str = 'cuda'):
        """
        Args:
            min_vals: Array of min values for each group (sorted by min)
            max_vals: Array of max values for each group (aligned with min_vals)
            device: GPU device to use ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.num_groups = len(min_vals)
        
        # Move metadata to GPU (this is the only VRAM usage!)
        self.min_tensor = torch.from_numpy(min_vals.astype(np.float32)).to(self.device)
        self.max_tensor = torch.from_numpy(max_vals.astype(np.float32)).to(self.device)
        
        # Compile or load CUDA kernel
        self.kernel = None
        if self.device == 'cuda' and CUDA_AVAILABLE:
            self._compile_kernel()
        
        print(f"✅ GPU Range Checker initialized")
        print(f"   Groups: {self.num_groups:,}")
        print(f"   Device: {self.device}")
        print(f"   Metadata VRAM: {self._get_metadata_size_mb():.2f} MB")
        if self.kernel:
            print(f"   GPU Kernel: COMPILED ✓")
        else:
            print(f"   GPU Kernel: FALLBACK (CPU)")
    
    def _get_metadata_size_mb(self) -> float:
        """Calculate metadata size in VRAM"""
        element_size = self.min_tensor.element_size()
        return (element_size * self.num_groups * 2) / 1024 / 1024
    
    def _compile_kernel(self):
        """Compile CUDA kernel using PyTorch's inline compiler"""
        # Read CUDA kernel source
        kernel_path = Path(__file__).parent / "range_kernel.cu"
        
        if not kernel_path.exists():
            print(f"⚠️ CUDA kernel file not found: {kernel_path}")
            print("   Falling back to CPU implementation")
            self.kernel = None
            return
        
        with open(kernel_path, 'r') as f:
            cuda_source = f.read()
        
        try:
            # Compile the CUDA kernel
            self.kernel = load_inline(
                name="range_checker",
                cpp_sources="""
                    #include <torch/extension.h>
                    #include <cuda_runtime.h>
                    
                    // Declare CUDA functions
                    extern void find_ranges_kernel(
                        const float* inputs,
                        const float* min_vals,
                        const float* max_vals,
                        int* output_groups,
                        int num_inputs,
                        int num_groups
                    );
                    
                    extern void find_ranges_kernel_shared(
                        const float* inputs,
                        const float* min_vals,
                        const float* max_vals,
                        int* output_groups,
                        int num_inputs,
                        int num_groups
                    );
                    
                    // Python interface
                    torch::Tensor find_ranges(
                        torch::Tensor inputs,
                        torch::Tensor min_vals,
                        torch::Tensor max_vals
                    ) {
                        auto num_inputs = inputs.numel();
                        auto num_groups = min_vals.numel();
                        auto output = torch::empty({num_inputs}, torch::dtype(torch::kInt32).device(inputs.device()));
                        
                        const int threads = 256;
                        const int blocks = (num_inputs + threads - 1) / threads;
                        
                        find_ranges_kernel(
                            inputs.data_ptr<float>(),
                            min_vals.data_ptr<float>(),
                            max_vals.data_ptr<float>(),
                            output.data_ptr<int>(),
                            num_inputs,
                            num_groups
                        );
                        
                        return output;
                    }
                    
                    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                        m.def("find_ranges", &find_ranges, "Find ranges for inputs");
                    }
                """,
                cuda_sources=cuda_source,
                verbose=False
            )
            print("   CUDA kernel compiled successfully!")
        except Exception as e:
            print(f"⚠️ CUDA kernel compilation failed: {e}")
            print("   Falling back to CPU implementation")
            self.kernel = None
    
    def find_groups(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Find which group each input value belongs to.
        
        This runs in parallel on the GPU for ALL inputs at once!
        
        Args:
            input_values: Tensor of shape (batch_size, dim) or (dim,)
            
        Returns:
            Tensor of group indices (-1 if not found)
        """
        # Ensure input is on the correct device
        if input_values.device != self.device:
            input_values = input_values.to(self.device)
        
        # Flatten input
        original_shape = input_values.shape
        flat_input = input_values.flatten()
        num_inputs = flat_input.shape[0]
        
        if self.kernel is not None and self.device == 'cuda':
            # GPU KERNEL PATH - SUPER FAST!
            # All inputs processed in parallel by thousands of GPU cores
            output_groups = self.kernel.find_ranges(
                flat_input,
                self.min_tensor,
                self.max_tensor
            )
        else:
            # CPU FALLBACK PATH - Still optimized with numpy
            output_groups = self._find_groups_cpu(flat_input.cpu().numpy())
            output_groups = torch.from_numpy(output_groups).to(self.device)
        
        # Reshape to original shape
        return output_groups.reshape(original_shape)
    
    def _find_groups_cpu(self, input_values: np.ndarray) -> np.ndarray:
        """CPU fallback using numpy's searchsorted (still fast, but not parallel)"""
        min_vals = self.min_tensor.cpu().numpy()
        max_vals = self.max_tensor.cpu().numpy()
        
        # Use binary search via numpy
        indices = np.searchsorted(min_vals, input_values, side='right') - 1
        indices = np.where(indices >= 0, indices, -1)
        
        # Verify the found group actually contains the value
        valid = (indices >= 0) & (input_values <= max_vals[indices])
        indices[~valid] = -1
        
        return indices.astype(np.int32)
    
    def get_stats(self) -> dict:
        """Get statistics about the range checker"""
        return {
            'num_groups': self.num_groups,
            'metadata_mb': self._get_metadata_size_mb(),
            'device': self.device,
            'using_gpu': self.kernel is not None,
            'gpu_available': CUDA_AVAILABLE
        }