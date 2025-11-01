#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> 
#include <exception>                 
#include <stdexcept>                
#include <vector>

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

    __device__ int max_device(int a, int b) {
        return (a > b) ? a : b;
    }

    __device__ int min_device(int a, int b) {
        return (a < b) ? a : b;
    }

    __device__ int find_row_for_interval(const int* offsets, int row_count, int idx) {
        int left = 0;
        int right = row_count; // offsets has size row_count + 1
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (offsets[mid + 1] <= idx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __global__ void intersection_count_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        int a_interval_count,
        const int* d_b_begin,
        const int* d_b_end,
        const int* d_b_row_offsets,
        int b_row_count,
        int* d_per_interval_counts)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= a_interval_count) {
            return;
        }

        int row = find_row_for_interval(d_a_row_offsets, a_row_count, idx);
        if (row < 0 || row >= a_row_count) {
            d_per_interval_counts[idx] = 0;
            return;
        }
        int b_start = (row < b_row_count) ? d_b_row_offsets[row] : 0;
        int b_end = (row < b_row_count) ? d_b_row_offsets[row + 1] : 0;
        int b_count = b_end - b_start;

        if (b_count <= 0) {
            d_per_interval_counts[idx] = 0;
            return;
        }

        const int* b_begin_line = d_b_begin + b_start;
        const int* b_end_line   = d_b_end   + b_start;

        int a_begin = d_a_begin[idx];
        int a_end   = d_a_end[idx];

        int j_min = lower_bound_end(b_end_line, b_count, a_begin);
        int j_max = lower_bound_begin(b_begin_line, b_count, a_end);

        int local_count = 0;
        for (int j = j_min; j < j_max; ++j) {
            int inter_begin = max_device(a_begin, b_begin_line[j]);
            int inter_end   = min_device(a_end,   b_end_line[j]);

            if (inter_begin < inter_end) {
                local_count++;
            }
        }
        d_per_interval_counts[idx] = local_count;
    }

    __global__ void intersection_write_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        int a_interval_count,
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
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= a_interval_count) {
            return;
        }

        int row = find_row_for_interval(d_a_row_offsets, a_row_count, idx);
        if (row < 0 || row >= a_row_count) {
            return;
        }
        int b_start = (row < b_row_count) ? d_b_row_offsets[row] : 0;
        int b_end = (row < b_row_count) ? d_b_row_offsets[row + 1] : 0;
        int b_count = b_end - b_start;

        if (b_count <= 0) {
            return;
        }

        const int* b_begin_line = d_b_begin + b_start;
        const int* b_end_line   = d_b_end   + b_start;

        int a_begin = d_a_begin[idx];
        int a_end   = d_a_end[idx];
        int write_offset = d_output_offsets[idx];
        int local_write_idx = 0;

        int j_min = lower_bound_end(b_end_line, b_count, a_begin);
        int j_max = lower_bound_begin(b_begin_line, b_count, a_end);

        for (int j = j_min; j < j_max; ++j) {
            int inter_begin = max_device(a_begin, b_begin_line[j]);
            int inter_end   = min_device(a_end,   b_end_line[j]);

            if (inter_begin < inter_end) {
                int pos = write_offset + local_write_idx;
                if (d_r_z_idx) {
                    d_r_z_idx[pos] = d_a_row_to_z[row];
                }
                d_r_y_idx[pos] = d_a_row_to_y[row];
                d_r_begin[pos] = inter_begin;
                d_r_end[pos]   = inter_end;
                d_a_idx[pos]   = idx;
                d_b_idx[pos]   = b_start + j;
                local_write_idx++;
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

    thrust::device_vector<int> d_per_interval_counts;
    thrust::device_vector<int> d_output_offsets;
    d_per_interval_counts.resize(a_interval_count);
    d_output_offsets.resize(a_interval_count); // exclusive_scan needs size n for output


    // Kernel Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (a_interval_count + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1 : counting for preallocation
    intersection_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count, a_interval_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        thrust::raw_pointer_cast(d_per_interval_counts.data())
    );

    err = KERNEL_CHECK();
    if (err != cudaSuccess) {
        return err;
    }

    // Prefix Sum (Scan) using Thrust
    thrust::exclusive_scan(thrust::device, // Execute on device
                       d_per_interval_counts.begin(),
                       d_per_interval_counts.end(),
                       d_output_offsets.begin());


    // Determine Total Count
    int h_total_count = 0;
    if (a_interval_count > 0) {
        // Total count = last offset + last count
        // Need to copy the last elements back to host
        int last_offset_val = 0;
        int last_count_val = 0;
            // Accessing .back() directly involves D->H copy
            last_offset_val = d_output_offsets.back();
            last_count_val  = d_per_interval_counts.back();
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
        intersection_write_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_a_begin, d_a_end,
            d_a_row_offsets, a_row_count, a_interval_count,
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
    std::vector<int> h_row_to_y(row_count);
    std::vector<int> h_row_to_z(row_count, 0);
    for (int row = 0; row < row_count; ++row) {
        h_row_to_y[row] = row;
    }

    int* d_row_to_y = nullptr;
    int* d_row_to_z = nullptr;
    const size_t row_map_bytes = static_cast<size_t>(row_count) * sizeof(int);

    cudaError_t err = cudaMalloc(&d_row_to_y, row_map_bytes);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMalloc(&d_row_to_z, row_map_bytes);
    if (err != cudaSuccess) {
        CUDA_CHECK(cudaFree(d_row_to_y));
        return err;
    }

    err = cudaMemcpy(d_row_to_y, h_row_to_y.data(), row_map_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        CUDA_CHECK(cudaFree(d_row_to_y));
        CUDA_CHECK(cudaFree(d_row_to_z));
        return err;
    }

    err = cudaMemcpy(d_row_to_z, h_row_to_z.data(), row_map_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        CUDA_CHECK(cudaFree(d_row_to_y));
        CUDA_CHECK(cudaFree(d_row_to_z));
        return err;
    }

    err = findVolumeIntersections(
        d_a_begin, d_a_end, a_interval_count,
        d_a_y_offsets, d_row_to_y, d_row_to_z, row_count,
        d_b_begin, d_b_end, b_interval_count,
        d_b_y_offsets, row_count,
        nullptr,
        d_r_y_idx, d_r_begin, d_r_end,
        d_a_idx, d_b_idx,
        total_intersections_count);

    CUDA_CHECK(cudaFree(d_row_to_y));
    CUDA_CHECK(cudaFree(d_row_to_z));
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
