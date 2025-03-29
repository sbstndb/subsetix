#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> 
#include <exception>                 
#include <stdexcept>                

namespace
{
    // Taken an interval b in a set set1, where is the first interval in set2 that may intersect with b ?	
    __device__ int lower_bound_end(const int* B_end, int n, int value) {
        int left = 0;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (B_end[mid] <= value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    // Taken an interval b in a set set1, where is the last interval in set2 that may intersect with b ? 
    __device__ int lower_bound_begin(const int* B_begin, int n, int value) {
        int left = 0;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (B_begin[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __device__ float max_device(int a, int b) {
        return (a > b) ? a : b;
    }

    __device__ float min_device(int a, int b) {
        return (a < b) ? a : b;
    }

// First step : compute the array size. 
// We want to store the intersection fast and in a parallel way. 
// Hence, the strategy is to precompute the array size
// Hence, each interval can write its intersection without any array locking 
// This need to be investiguated : we can do much better :-)
    __global__ void intersection_count_kernel(
        const int* d_a_begin, const int* d_a_end, int a_size,
        const int* d_b_begin, const int* d_b_end, int b_size,
        int* d_per_thread_counts) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < a_size) {
            int a_begin = d_a_begin[i];
            int a_end   = d_a_end[i];
            int local_count = 0;

            int j_min = lower_bound_end(  d_b_end,   b_size, a_begin);
            int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

            for (int j = j_min; j < j_max && j < b_size; j++) {
                int b_begin = d_b_begin[j];
                int b_end   = d_b_end[j];
                int inter_begin = max_device(a_begin, b_begin);
                int inter_end   = min_device(a_end, b_end);

                if (inter_begin < inter_end) {
                    local_count++;
                }
            }
            d_per_thread_counts[i] = local_count;
        }
    }

// Second step : We have the preallocated array. 
// Each thread take an interval and try to find intersection in [lower_bound_begin, lower_bound_end]
// Then, it writes the intersections in the output subset array. 
    __global__ void intersection_write_kernel(
        const int* d_a_begin, const int* d_a_end, int a_size,
        const int* d_b_begin, const int* d_b_end, int b_size,
        const int* d_output_offsets, 
        int* d_r_begin, int* d_r_end,
        int* d_a_idx,   int* d_b_idx)   
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < a_size) {
            int a_begin = d_a_begin[i];
            int a_end   = d_a_end[i];
            int write_offset = d_output_offsets[i];
            int local_write_idx = 0;

            int j_min = lower_bound_end(  d_b_end,   b_size, a_begin);
            int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

            for (int j = j_min; j < j_max && j < b_size; j++) {
                int b_begin = d_b_begin[j];
                int b_end = d_b_end[j];
                int inter_begin = max_device(a_begin, b_begin);
                int inter_end   = min_device(a_end,   b_end);

                if (inter_begin < inter_end) {
                    int pos = write_offset + local_write_idx;
                    d_r_begin[pos] = inter_begin;
                    d_r_end[pos] = inter_end;
                    d_a_idx[pos] = i;
                    d_b_idx[pos] = j;
                    local_write_idx++;
                }
            }
        }
    }

} 



cudaError_t findIntervalIntersections(
    const int* d_a_begin, const int* d_a_end, int a_size,
    const int* d_b_begin, const int* d_b_end, int b_size,
    int** d_r_begin, int** d_r_end,
    int** d_a_idx, int** d_b_idx,
    int* total_intersections_count)
{

    *d_r_begin = nullptr;
    *d_r_end = nullptr;
    *d_a_idx = nullptr;
    *d_b_idx = nullptr;
    *total_intersections_count = 0;


    cudaError_t err = cudaSuccess;

    thrust::device_vector<int> d_per_thread_counts;
    thrust::device_vector<int> d_output_offsets;
    d_per_thread_counts.resize(a_size);
    d_output_offsets.resize(a_size); // exclusive_scan needs size n for output


    // Kernel Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (a_size + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1 : counting for preallocation
    intersection_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end, a_size,
        d_b_begin, d_b_end, b_size,
        thrust::raw_pointer_cast(d_per_thread_counts.data())
    );

    // Prefix Sum (Scan) using Thrust
    thrust::exclusive_scan(thrust::device, // Execute on device
                       d_per_thread_counts.begin(),
                       d_per_thread_counts.end(),
                       d_output_offsets.begin());


    // Determine Total Count
    int h_total_count = 0;
    if (a_size > 0) {
        // Total count = last offset + last count
        // Need to copy the last elements back to host
        int last_offset_val = 0;
        int last_count_val = 0;
            // Accessing .back() directly involves D->H copy
            last_offset_val = d_output_offsets.back();
            last_count_val  = d_per_thread_counts.back();
            h_total_count   = last_offset_val + last_count_val;
    }

    *total_intersections_count = h_total_count;

    // Allocate Final Output Arrays and Write Intersections
    if (h_total_count > 0) {
        size_t results_size_bytes = (size_t)h_total_count * sizeof(int);
        size_t indices_size_bytes = (size_t)h_total_count * sizeof(int);

        // Allocate output device memory - use the pointers passed by the caller
        cudaMalloc(d_r_begin, results_size_bytes);
        cudaMalloc(d_r_end,   results_size_bytes);
        cudaMalloc(d_a_idx,   indices_size_bytes);
        cudaMalloc(d_b_idx,   indices_size_bytes);

        // Launch Write Kernel
        intersection_write_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_a_begin, d_a_end, a_size,
            d_b_begin, d_b_end, b_size,
            thrust::raw_pointer_cast(d_output_offsets.data()),
            *d_r_begin, *d_r_end, 
            *d_a_idx,   *d_b_idx
        );

    } // End if (h_total_count > 0)

    return cudaSuccess;
}

void freeIntersectionResults(int* d_r_begin, int* d_r_end, int* d_a_idx, int* d_b_idx) {
    CUDA_CHECK(cudaFree(d_r_begin));
    CUDA_CHECK(cudaFree(d_r_end));
    CUDA_CHECK(cudaFree(d_a_idx));
    CUDA_CHECK(cudaFree(d_b_idx));
}

