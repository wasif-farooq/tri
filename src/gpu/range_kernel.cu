/*
 * CUDA Kernel for Parallel Range Checking
 * Location: torch-range-indexed/src/gpu/range_kernel.cu
 * 
 * This kernel runs on the GPU and checks which weight group
 * each input value belongs to, in parallel across thousands of threads.
 */

#include <cuda_runtime.h>

// Binary search on sorted min/max arrays
__device__ int binary_search_range(
    float value,
    const float* min_vals,
    const float* max_vals,
    int num_groups
) {
    int left = 0;
    int right = num_groups - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        if (value < min_vals[mid]) {
            right = mid - 1;
        } else if (value > max_vals[mid]) {
            left = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

// Main kernel: Find ranges for ALL inputs in parallel
extern "C" __global__ void find_ranges_kernel(
    const float* __restrict__ inputs,      // [num_inputs]
    const float* __restrict__ min_vals,    // [num_groups]
    const float* __restrict__ max_vals,    // [num_groups]
    int* __restrict__ output_groups,       // [num_inputs]
    int num_inputs,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_inputs) {
        float input_val = inputs[idx];
        int group_id = binary_search_range(input_val, min_vals, max_vals, num_groups);
        output_groups[idx] = group_id;
    }
}

// Optimized kernel with shared memory (faster for repeated access)
extern "C" __global__ void find_ranges_kernel_shared(
    const float* __restrict__ inputs,
    const float* __restrict__ min_vals,
    const float* __restrict__ max_vals,
    int* __restrict__ output_groups,
    int num_inputs,
    int num_groups
) {
    extern __shared__ float shared_min[];
    float* shared_max = &shared_min[blockDim.x];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative loading: all threads load metadata together
    if (threadIdx.x < num_groups && blockIdx.x == 0) {
        shared_min[threadIdx.x] = min_vals[threadIdx.x];
        shared_max[threadIdx.x] = max_vals[threadIdx.x];
    }
    __syncthreads();
    
    if (idx < num_inputs) {
        float input_val = inputs[idx];
        int left = 0;
        int right = num_groups - 1;
        
        while (left <= right) {
            int mid = (left + right) / 2;
            
            if (input_val < shared_min[mid]) {
                right = mid - 1;
            } else if (input_val > shared_max[mid]) {
                left = mid + 1;
            } else {
                output_groups[idx] = mid;
                return;
            }
        }
        output_groups[idx] = -1;
    }
}