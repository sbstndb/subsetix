#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_ptr.h>
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


cudaError_t computeVolumeIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    int* total_intersections_count)
{
    if (total_intersections_count) {
        *total_intersections_count = 0;
    }

    if (a_row_count <= 0 || b_row_count <= 0) {
        return cudaSuccess;
    }

    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_counts || !d_offsets) {
        return cudaErrorInvalidValue;
    }

    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;

    row_intersection_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_counts);

    cudaError_t err = KERNEL_CHECK();
    if (err != cudaSuccess) {
        return err;
    }

    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(d_counts),
        thrust::device_pointer_cast(d_counts + a_row_count),
        thrust::device_pointer_cast(d_offsets));

    int total = 0;
    if (a_row_count > 0) {
        int last_offset = 0;
        int last_count = 0;
        err = CUDA_CHECK(cudaMemcpy(&last_offset,
                                    d_offsets + (a_row_count - 1),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            return err;
        }
        err = CUDA_CHECK(cudaMemcpy(&last_count,
                                    d_counts + (a_row_count - 1),
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            return err;
        }
        total = last_offset + last_count;
    }

    if (total_intersections_count) {
        *total_intersections_count = total;
    }

    return cudaSuccess;
}

cudaError_t writeVolumeIntersectionsWithOffsets(
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
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx)
{
    if (a_row_count <= 0 || b_row_count <= 0) {
        return cudaSuccess;
    }

    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_offsets || !d_r_y_idx || !d_r_begin || !d_r_end ||
        !d_a_idx || !d_b_idx) {
        return cudaErrorInvalidValue;
    }

    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;

    row_intersection_write_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        d_a_idx,
        d_b_idx);

    return KERNEL_CHECK();
}

cudaError_t computeIntervalIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    int* total_intersections_count)
{
    return computeVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        total_intersections_count);
}

cudaError_t writeIntervalIntersectionsWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx)
{
    return writeVolumeIntersectionsWithOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        d_a_idx,
        d_b_idx);
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
    if (!d_r_y_idx || !d_r_begin || !d_r_end || !d_a_idx || !d_b_idx) {
        return cudaErrorInvalidValue;
    }

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

    thrust::device_vector<int> d_per_row_counts(a_row_count);
    thrust::device_vector<int> d_output_offsets(a_row_count);

    int computed_total = 0;
    err = computeVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        thrust::raw_pointer_cast(d_per_row_counts.data()),
        thrust::raw_pointer_cast(d_output_offsets.data()),
        &computed_total);
    if (err != cudaSuccess) {
        return err;
    }

    if (total_intersections_count) {
        *total_intersections_count = computed_total;
    }

    if (computed_total <= 0) {
        return cudaSuccess;
    }

    size_t results_size_bytes = static_cast<size_t>(computed_total) * sizeof(int);
    size_t indices_size_bytes = static_cast<size_t>(computed_total) * sizeof(int);

    int* local_r_z = nullptr;
    int* local_r_y = nullptr;
    int* local_r_begin = nullptr;
    int* local_r_end = nullptr;
    int* local_a_idx = nullptr;
    int* local_b_idx = nullptr;

    auto cleanup_outputs = [&]() {
        freeVolumeIntersectionResults(local_r_z, local_r_y, local_r_begin, local_r_end, local_a_idx, local_b_idx);
        local_r_z = local_r_y = local_r_begin = local_r_end = local_a_idx = local_b_idx = nullptr;
        if (d_r_z_idx) *d_r_z_idx = nullptr;
        if (d_r_y_idx) *d_r_y_idx = nullptr;
        if (d_r_begin) *d_r_begin = nullptr;
        if (d_r_end) *d_r_end = nullptr;
        if (d_a_idx) *d_a_idx = nullptr;
        if (d_b_idx) *d_b_idx = nullptr;
        if (total_intersections_count) {
            *total_intersections_count = 0;
        }
    };

    if (d_r_z_idx) {
        err = CUDA_CHECK(cudaMalloc(&local_r_z, results_size_bytes));
        if (err != cudaSuccess) {
            cleanup_outputs();
            return err;
        }
        *d_r_z_idx = local_r_z;
    }

    err = CUDA_CHECK(cudaMalloc(&local_r_y, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_y_idx = local_r_y;

    err = CUDA_CHECK(cudaMalloc(&local_r_begin, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_begin = local_r_begin;

    err = CUDA_CHECK(cudaMalloc(&local_r_end, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_end = local_r_end;

    err = CUDA_CHECK(cudaMalloc(&local_a_idx, indices_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_a_idx = local_a_idx;

    err = CUDA_CHECK(cudaMalloc(&local_b_idx, indices_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_b_idx = local_b_idx;

    err = writeVolumeIntersectionsWithOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        thrust::raw_pointer_cast(d_output_offsets.data()),
        d_r_z_idx ? *d_r_z_idx : nullptr,
        *d_r_y_idx,
        *d_r_begin,
        *d_r_end,
        *d_a_idx,
        *d_b_idx);

    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }

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
    if (!d_r_y_idx || !d_r_begin || !d_r_end || !d_a_idx || !d_b_idx) {
        return cudaErrorInvalidValue;
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

    if (a_interval_count <= 0 || b_interval_count <= 0 || a_y_count <= 0 || b_y_count <= 0) {
        return cudaSuccess;
    }

    if (a_y_count != b_y_count) {
       return cudaErrorInvalidValue;
    }

    thrust::device_vector<int> d_counts(a_y_count);
    thrust::device_vector<int> d_offsets(a_y_count);

    int computed_total = 0;
    cudaError_t err = computeIntervalIntersectionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        thrust::raw_pointer_cast(d_counts.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        &computed_total);
    if (err != cudaSuccess) {
        return err;
    }

    if (total_intersections_count) {
        *total_intersections_count = computed_total;
    }

    if (computed_total <= 0) {
        return cudaSuccess;
    }

    size_t results_size_bytes = static_cast<size_t>(computed_total) * sizeof(int);
    size_t indices_size_bytes = static_cast<size_t>(computed_total) * sizeof(int);

    int* local_r_y = nullptr;
    int* local_r_begin = nullptr;
    int* local_r_end = nullptr;
    int* local_a_idx = nullptr;
    int* local_b_idx = nullptr;

    auto cleanup_outputs = [&]() {
        freeIntervalResults(local_r_y, local_r_begin, local_r_end, local_a_idx, local_b_idx);
        local_r_y = local_r_begin = local_r_end = local_a_idx = local_b_idx = nullptr;
        if (d_r_y_idx) *d_r_y_idx = nullptr;
        if (d_r_begin) *d_r_begin = nullptr;
        if (d_r_end) *d_r_end = nullptr;
        if (d_a_idx) *d_a_idx = nullptr;
        if (d_b_idx) *d_b_idx = nullptr;
        if (total_intersections_count) {
            *total_intersections_count = 0;
        }
    };

    err = CUDA_CHECK(cudaMalloc(&local_r_y, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_y_idx = local_r_y;

    err = CUDA_CHECK(cudaMalloc(&local_r_begin, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_begin = local_r_begin;

    err = CUDA_CHECK(cudaMalloc(&local_r_end, results_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_r_end = local_r_end;

    err = CUDA_CHECK(cudaMalloc(&local_a_idx, indices_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_a_idx = local_a_idx;

    err = CUDA_CHECK(cudaMalloc(&local_b_idx, indices_size_bytes));
    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }
    *d_b_idx = local_b_idx;

    err = writeIntervalIntersectionsWithOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        thrust::raw_pointer_cast(d_offsets.data()),
        *d_r_y_idx,
        *d_r_begin,
        *d_r_end,
        *d_a_idx,
        *d_b_idx);

    if (err != cudaSuccess) {
        cleanup_outputs();
        return err;
    }

    return cudaSuccess;
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
