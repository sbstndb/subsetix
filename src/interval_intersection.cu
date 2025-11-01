#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> 
#include <exception>                 
#include <stdexcept>                

namespace
{
    __device__ int max_device(int a, int b) {
        return (a > b) ? a : b;
    }

    __device__ int min_device(int a, int b) {
        return (a < b) ? a : b;
    }

    __global__ void row_intersection_count_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        const int* d_b_begin,
        const int* d_b_end,
        const int* d_b_row_offsets,
        int b_row_count,
        int* d_per_row_counts)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        if (row >= b_row_count) {
            d_per_row_counts[row] = 0;
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];
        int b_start = d_b_row_offsets[row];
        int b_finish = d_b_row_offsets[row + 1];

        if (a_start >= a_finish || b_start >= b_finish) {
            d_per_row_counts[row] = 0;
            return;
        }

        int i = a_start;
        int j = b_start;
        int count = 0;

        while (i < a_finish && j < b_finish) {
            int inter_begin = max_device(d_a_begin[i], d_b_begin[j]);
            int inter_end   = min_device(d_a_end[i],   d_b_end[j]);
            if (inter_begin < inter_end) {
                ++count;
            }

            int a_interval_end = d_a_end[i];
            int b_interval_end = d_b_end[j];

            if (a_interval_end < b_interval_end) {
                ++i;
            } else if (b_interval_end < a_interval_end) {
                ++j;
            } else {
                ++i;
                ++j;
            }
        }
        d_per_row_counts[row] = count;
    }

    __global__ void row_intersection_write_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        const int* d_b_begin,
        const int* d_b_end,
        const int* d_b_row_offsets,
        int b_row_count,
        const int* d_a_row_to_y,
        const int* d_a_row_to_z,
        const int* d_output_offsets,
        int* d_r_z_idx,
        int* d_r_y_idx,
        int* d_r_begin,
        int* d_r_end,
        int* d_a_idx,
        int* d_b_idx)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        if (row >= b_row_count) {
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];
        int b_start = d_b_row_offsets[row];
        int b_finish = d_b_row_offsets[row + 1];

        if (a_start >= a_finish || b_start >= b_finish) {
            return;
        }

        int i = a_start;
        int j = b_start;
        int write_offset = d_output_offsets[row];
        int local_index = 0;

        int y_value = d_a_row_to_y ? d_a_row_to_y[row] : row;
        int z_value = d_a_row_to_z ? d_a_row_to_z[row] : 0;

        while (i < a_finish && j < b_finish) {
            int inter_begin = max_device(d_a_begin[i], d_b_begin[j]);
            int inter_end   = min_device(d_a_end[i],   d_b_end[j]);

            if (inter_begin < inter_end) {
                int pos = write_offset + local_index;
                if (d_r_z_idx) {
                    d_r_z_idx[pos] = z_value;
                }
                d_r_y_idx[pos] = y_value;
                d_r_begin[pos] = inter_begin;
                d_r_end[pos]   = inter_end;
                d_a_idx[pos]   = i;
                d_b_idx[pos]   = j;
                ++local_index;
            }

            int a_interval_end = d_a_end[i];
            int b_interval_end = d_b_end[j];

            if (a_interval_end < b_interval_end) {
                ++i;
            } else if (b_interval_end < a_interval_end) {
                ++j;
            } else {
                ++i;
                ++j;
            }
        }
    }

} 



cudaError_t findVolumeIntersections(
    const int* d_a_begin, const int* d_a_end, int a_interval_count,
    const int* d_a_row_offsets, const int* d_a_row_to_y, const int* d_a_row_to_z, int a_row_count,
    const int* d_b_begin, const int* d_b_end, int b_interval_count,
    const int* d_b_row_offsets, int b_row_count,
    int** d_r_z_idx, int** d_r_y_idx, int** d_r_begin, int** d_r_end,
    int** d_a_idx, int** d_b_idx,
    int* total_intersections_count)
{

    if (d_r_z_idx) {
        *d_r_z_idx = nullptr;
    }
    if (d_r_y_idx) {
        *d_r_y_idx = nullptr;
    }
    if (d_r_begin) {
        *d_r_begin = nullptr;
    }
    if (d_r_end) {
        *d_r_end = nullptr;
    }
    if (d_a_idx) {
        *d_a_idx = nullptr;
    }
    if (d_b_idx) {
        *d_b_idx = nullptr;
    }
    if (total_intersections_count) {
        *total_intersections_count = 0;
    }


    cudaError_t err = cudaSuccess;

    if (a_interval_count <= 0 || b_interval_count <= 0 || a_row_count <= 0 || b_row_count <= 0) {
        return cudaSuccess;
    }

    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }

    thrust::device_vector<int> d_per_row_counts;
    thrust::device_vector<int> d_output_offsets;
    d_per_row_counts.resize(a_row_count);
    d_output_offsets.resize(a_row_count); // exclusive_scan needs size n for output


    // Kernel Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1 : counting for preallocation
    row_intersection_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        thrust::raw_pointer_cast(d_per_row_counts.data())
    );

    err = KERNEL_CHECK();
    if (err != cudaSuccess) {
        return err;
    }

    // Prefix Sum (Scan) using Thrust
    thrust::exclusive_scan(thrust::device, // Execute on device
                       d_per_row_counts.begin(),
                       d_per_row_counts.end(),
                       d_output_offsets.begin());


    // Determine Total Count
    int h_total_count = 0;
    if (a_row_count > 0) {
        // Total count = last offset + last count
        // Need to copy the last elements back to host
        int last_offset_val = 0;
        int last_count_val = 0;
            // Accessing .back() directly involves D->H copy
            last_offset_val = d_output_offsets.back();
            last_count_val  = d_per_row_counts.back();
            h_total_count   = last_offset_val + last_count_val;
    }

    if (total_intersections_count) {
        *total_intersections_count = h_total_count;
    }

    // Allocate Final Output Arrays and Write Intersections
    if (h_total_count > 0) {
        size_t results_size_bytes = (size_t)h_total_count * sizeof(int);
        size_t indices_size_bytes = (size_t)h_total_count * sizeof(int);

        // Allocate output device memory - use the pointers passed by the caller
        if (d_r_z_idx) {
            cudaMalloc(d_r_z_idx, results_size_bytes);
        }
        if (d_r_y_idx) {
            cudaMalloc(d_r_y_idx, results_size_bytes);
        }
        if (d_r_begin) {
            cudaMalloc(d_r_begin, results_size_bytes);
        }
        if (d_r_end) {
            cudaMalloc(d_r_end, results_size_bytes);
        }
        if (d_a_idx) {
            cudaMalloc(d_a_idx, indices_size_bytes);
        }
        if (d_b_idx) {
            cudaMalloc(d_b_idx, indices_size_bytes);
        }

        // Launch Write Kernel
        row_intersection_write_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_a_begin, d_a_end,
            d_a_row_offsets, a_row_count,
            d_b_begin, d_b_end,
            d_b_row_offsets, b_row_count,
            d_a_row_to_y, d_a_row_to_z,
            thrust::raw_pointer_cast(d_output_offsets.data()),
            d_r_z_idx ? *d_r_z_idx : nullptr,
            d_r_y_idx ? *d_r_y_idx : nullptr,
            d_r_begin ? *d_r_begin : nullptr,
            d_r_end ? *d_r_end : nullptr,
            d_a_idx ? *d_a_idx : nullptr,
            d_b_idx ? *d_b_idx : nullptr
        );

        err = KERNEL_CHECK();
        if (err != cudaSuccess) {
            return err;
        }

    } // End if (h_total_count > 0)

    return cudaSuccess;
}

cudaError_t findIntervalIntersections(
    const int* d_a_begin, const int* d_a_end, int a_interval_count,
    const int* d_a_y_offsets, int a_y_count,
    const int* d_b_begin, const int* d_b_end, int b_interval_count,
    const int* d_b_y_offsets, int b_y_count,
    int** d_r_y_idx, int** d_r_begin, int** d_r_end,
    int** d_a_idx, int** d_b_idx,
    int* total_intersections_count)
{
    if (d_r_y_idx) {
        *d_r_y_idx = nullptr;
    }
    if (d_r_begin) {
        *d_r_begin = nullptr;
    }
    if (d_r_end) {
        *d_r_end = nullptr;
    }
    if (d_a_idx) {
        *d_a_idx = nullptr;
    }
    if (d_b_idx) {
        *d_b_idx = nullptr;
    }
    if (total_intersections_count) {
        *total_intersections_count = 0;
    }

    if (a_interval_count <= 0 || b_interval_count <= 0 || a_y_count <= 0 || b_y_count <= 0) {
        return cudaSuccess;
    }

    if (a_y_count != b_y_count) {
        return cudaErrorInvalidValue;
    }

    const int row_count = a_y_count;

    cudaError_t err = findVolumeIntersections(
        d_a_begin, d_a_end, a_interval_count,
        d_a_y_offsets, nullptr, nullptr, row_count,
        d_b_begin, d_b_end, b_interval_count,
        d_b_y_offsets, row_count,
        nullptr,
        d_r_y_idx, d_r_begin, d_r_end,
        d_a_idx, d_b_idx,
        total_intersections_count);

    return err;
}

void freeVolumeIntersectionResults(int* d_r_z_idx, int* d_r_y_idx, int* d_r_begin, int* d_r_end, int* d_a_idx, int* d_b_idx) {
    if (d_r_z_idx) CUDA_CHECK(cudaFree(d_r_z_idx));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
}

void freeIntervalResults(int* d_r_y_idx, int* d_r_begin, int* d_r_end, int* d_a_idx, int* d_b_idx) {
    freeVolumeIntersectionResults(nullptr, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
}
