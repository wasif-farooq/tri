"""Utility functions for TRI"""

import torch
import psutil
from typing import Dict, Optional


def calculate_optimal_cache_size(available_vram_mb: int) -> int:
    """
    Calculate optimal cache size based on available VRAM.
    
    Args:
        available_vram_mb: Available VRAM in megabytes
        
    Returns:
        Optimal cache size in megabytes
    """
    # Reserve 2GB for KV cache and activations
    reserved_mb = 2048
    
    # Reserve 1.5GB for metadata (for 70B model)
    metadata_mb = 1536
    
    cache_mb = available_vram_mb - reserved_mb - metadata_mb
    
    # Ensure minimum cache
    return max(256, cache_mb)


def get_device_info() -> Dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'devices': []
    }
    
    if info['cuda_available']:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'id': i,
                'name': props.name,
                'vram_mb': props.total_memory / 1024 / 1024,
                'compute_capability': f"{props.major}.{props.minor}"
            })
    else:
        info['devices'].append({
            'id': 0,
            'name': 'CPU',
            'ram_mb': psutil.virtual_memory().total / 1024 / 1024,
            'compute_capability': 'N/A'
        })
    
    return info


def estimate_model_size(num_parameters: int, dtype: str = 'fp16') -> float:
    """
    Estimate model size in gigabytes.
    
    Args:
        num_parameters: Number of parameters
        dtype: Data type ('fp32', 'fp16', 'int8', 'int4')
        
    Returns:
        Estimated size in GB
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5,
    }.get(dtype, 2)
    
    return num_parameters * bytes_per_param / 1024 / 1024 / 1024


def estimate_vram_requirement_tri(num_parameters: int, 
                                   group_size: int = 1000) -> Dict:
    """
    Estimate VRAM requirements for TRI system.
    
    Args:
        num_parameters: Number of parameters
        group_size: Group size for indexing
        
    Returns:
        Dictionary with VRAM estimates
    """
    num_groups = (num_parameters + group_size - 1) // group_size
    
    # Metadata: 48 bytes per group (min, max, mean, std, offset, size)
    metadata_mb = num_groups * 48 / 1024 / 1024
    
    # Cache: 20% of model size
    model_size_gb = estimate_model_size(num_parameters, 'fp16')
    cache_gb = model_size_gb * 0.2
    
    # KV cache: 2GB for 2048 context
    kv_cache_gb = 2.0
    
    # Activations: 1GB
    activations_gb = 1.0
    
    total_gb = metadata_mb / 1024 + cache_gb + kv_cache_gb + activations_gb
    
    return {
        'metadata_mb': metadata_mb,
        'cache_gb': cache_gb,
        'kv_cache_gb': kv_cache_gb,
        'activations_gb': activations_gb,
        'total_gb': total_gb,
    }


def format_bytes(bytes: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"