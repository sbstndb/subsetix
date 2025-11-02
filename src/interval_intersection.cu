#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/device/device_scan.cuh>
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

    template <typename Emit>
    __device__ void union_traverse(const int* d_a_begin,
                                   const int* d_a_end,
                                   int a_start,
                                   int a_finish,
                                   const int* d_b_begin,
                                   const int* d_b_end,
                                   int b_start,
                                   int b_finish,
                                   Emit& emit)
    {
        int i = a_start;
        int j = b_start;
        bool has_active = false;
        int active_begin = 0;
        int active_end = 0;

        while (i < a_finish || j < b_finish) {
            const bool take_a = (j >= b_finish) ||
                                (i < a_finish && d_a_begin[i] <= d_b_begin[j]);
            int seg_begin = 0;
            int seg_end = 0;
            if (take_a) {
                seg_begin = d_a_begin[i];
                seg_end = d_a_end[i];
                ++i;
            } else {
                seg_begin = d_b_begin[j];
                seg_end = d_b_end[j];
                ++j;
            }

            if (!has_active) {
                active_begin = seg_begin;
                active_end = seg_end;
                has_active = true;
                continue;
            }

            if (seg_begin <= active_end) {
                if (seg_end > active_end) {
                    active_end = seg_end;
                }
            } else {
                emit.emit(active_begin, active_end);
                active_begin = seg_begin;
                active_end = seg_end;
            }
        }

        if (has_active) {
            emit.emit(active_begin, active_end);
        }
    }

    template <typename Emit>
    __device__ void difference_traverse(const int* d_a_begin,
                                        const int* d_a_end,
                                        int a_start,
                                        int a_finish,
                                        const int* d_b_begin,
                                        const int* d_b_end,
                                        int b_start,
                                        int b_finish,
                                        Emit& emit)
    {
        int i = a_start;
        int j = b_start;

        while (i < a_finish) {
            int seg_begin = d_a_begin[i];
            int seg_end = d_a_end[i];
            int cursor = seg_begin;

            while (j < b_finish && d_b_end[j] <= cursor) {
                ++j;
            }

            while (j < b_finish && cursor < seg_end && d_b_begin[j] < seg_end) {
                int overlap_begin = max_device(cursor, d_b_begin[j]);
                int overlap_end = min_device(seg_end, d_b_end[j]);

                if (overlap_begin > cursor) {
                    emit.emit(cursor, overlap_begin);
                }

                if (overlap_end >= seg_end) {
                    cursor = seg_end;
                    break;
                }

                cursor = overlap_end;
                ++j;
                while (j < b_finish && d_b_end[j] <= cursor) {
                    ++j;
                }
            }

            if (cursor < seg_end) {
                emit.emit(cursor, seg_end);
            }

            ++i;
        }
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

    __global__ void write_total_from_prefix_kernel(const int* d_counts,
                                                   const int* d_offsets,
                                                   int row_count,
                                                   int* d_total)
    {
        if (!d_total) {
            return;
        }
        if (threadIdx.x == 0) {
            int total = 0;
            if (row_count > 0 && d_counts && d_offsets) {
                total = d_offsets[row_count - 1] + d_counts[row_count - 1];
            }
            d_total[0] = total;
        }
    }

    __global__ void row_union_count_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        const int* d_b_begin,
        const int* d_b_end,
        const int* d_b_row_offsets,
        int b_row_count,
        int* d_counts)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];

        int b_start = 0;
        int b_finish = 0;
        if (row < b_row_count) {
            b_start = d_b_row_offsets[row];
            b_finish = d_b_row_offsets[row + 1];
        }

        struct CountEmitter {
            int count;
            __device__ CountEmitter() : count(0) {}
            __device__ void emit(int begin, int end) {
                if (begin < end) {
                    ++count;
                }
            }
        };

        CountEmitter emitter;
        union_traverse(d_a_begin, d_a_end, a_start, a_finish,
                       d_b_begin, d_b_end, b_start, b_finish,
                       emitter);
        d_counts[row] = emitter.count;
    }

    __global__ void row_union_write_kernel(
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
        int* d_r_end)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];

        int b_start = 0;
        int b_finish = 0;
        if (row < b_row_count) {
            b_start = d_b_row_offsets[row];
            b_finish = d_b_row_offsets[row + 1];
        }

        int y_value = d_a_row_to_y ? d_a_row_to_y[row] : row;
        int z_value = d_a_row_to_z ? d_a_row_to_z[row] : 0;
        int write_offset = d_offsets[row];

        struct WriteEmitter {
            int* r_z;
            int* r_y;
            int* r_begin;
            int* r_end;
            int z;
            int y;
            int base;
            int local;
            __device__ WriteEmitter(int* rz,
                                    int* ry,
                                    int* rb,
                                    int* re,
                                    int z_value,
                                    int y_value,
                                    int offset)
                : r_z(rz)
                , r_y(ry)
                , r_begin(rb)
                , r_end(re)
                , z(z_value)
                , y(y_value)
                , base(offset)
                , local(0) {}
            __device__ void emit(int begin, int end) {
                if (begin >= end) {
                    return;
                }
                int pos = base + local;
                if (r_z) {
                    r_z[pos] = z;
                }
                if (r_y) {
                    r_y[pos] = y;
                }
                r_begin[pos] = begin;
                r_end[pos] = end;
                ++local;
            }
        };

        WriteEmitter emitter(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end,
                             z_value, y_value, write_offset);

        union_traverse(d_a_begin, d_a_end, a_start, a_finish,
                       d_b_begin, d_b_end, b_start, b_finish,
                       emitter);
    }

    __global__ void row_difference_count_kernel(
        const int* d_a_begin,
        const int* d_a_end,
        const int* d_a_row_offsets,
        int a_row_count,
        const int* d_b_begin,
        const int* d_b_end,
        const int* d_b_row_offsets,
        int b_row_count,
        int* d_counts)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];

        int b_start = 0;
        int b_finish = 0;
        if (row < b_row_count) {
            b_start = d_b_row_offsets[row];
            b_finish = d_b_row_offsets[row + 1];
        }

        struct CountEmitter {
            int count;
            __device__ CountEmitter() : count(0) {}
            __device__ void emit(int begin, int end) {
                if (begin < end) {
                    ++count;
                }
            }
        };

        CountEmitter emitter;
        difference_traverse(d_a_begin, d_a_end, a_start, a_finish,
                            d_b_begin, d_b_end, b_start, b_finish,
                            emitter);
        d_counts[row] = emitter.count;
    }

    __global__ void row_difference_write_kernel(
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
        int* d_r_end)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= a_row_count) {
            return;
        }

        int a_start = d_a_row_offsets[row];
        int a_finish = d_a_row_offsets[row + 1];

        int b_start = 0;
        int b_finish = 0;
        if (row < b_row_count) {
            b_start = d_b_row_offsets[row];
            b_finish = d_b_row_offsets[row + 1];
        }

        int y_value = d_a_row_to_y ? d_a_row_to_y[row] : row;
        int z_value = d_a_row_to_z ? d_a_row_to_z[row] : 0;
        int write_offset = d_offsets[row];

        struct WriteEmitter {
            int* r_z;
            int* r_y;
            int* r_begin;
            int* r_end;
            int z;
            int y;
            int base;
            int local;
            __device__ WriteEmitter(int* rz,
                                    int* ry,
                                    int* rb,
                                    int* re,
                                    int z_value,
                                    int y_value,
                                    int offset)
                : r_z(rz)
                , r_y(ry)
                , r_begin(rb)
                , r_end(re)
                , z(z_value)
                , y(y_value)
                , base(offset)
                , local(0) {}
            __device__ void emit(int begin, int end) {
                if (begin >= end) {
                    return;
                }
                int pos = base + local;
                if (r_z) {
                    r_z[pos] = z;
                }
                if (r_y) {
                    r_y[pos] = y;
                }
                r_begin[pos] = begin;
                r_end[pos] = end;
                ++local;
            }
        };

        WriteEmitter emitter(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end,
                             z_value, y_value, write_offset);

        difference_traverse(d_a_begin, d_a_end, a_start, a_finish,
                            d_b_begin, d_b_end, b_start, b_finish,
                            emitter);
    }

} 


cudaError_t enqueueVolumeIntersectionOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    if (!d_counts || !d_offsets) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 || b_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t s = stream ? stream : nullptr;

    row_intersection_count_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (d_temp_storage && temp_storage_bytes > 0) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                             temp_storage_bytes,
                                             d_counts,
                                             d_offsets,
                                             a_row_count,
                                             s);
    }

    thrust::exclusive_scan(
        thrust::cuda::par.on(s),
        thrust::device_pointer_cast(d_counts),
        thrust::device_pointer_cast(d_counts + a_row_count),
        thrust::device_pointer_cast(d_offsets));

    return cudaSuccess;
}

cudaError_t enqueueVolumeIntersectionWrite(
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
    int* d_b_idx,
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 || b_row_count == 0) {
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
    cudaStream_t s = stream ? stream : nullptr;

    row_intersection_write_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
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

    return cudaGetLastError();
}

cudaError_t enqueueVolumeUnionOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    if (!d_counts || !d_offsets) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 && b_row_count == 0) {
        return cudaSuccess;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t s = stream ? stream : nullptr;

    row_union_count_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (d_temp_storage && temp_storage_bytes > 0) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                             temp_storage_bytes,
                                             d_counts,
                                             d_offsets,
                                             a_row_count,
                                             s);
    }

    thrust::exclusive_scan(
        thrust::cuda::par.on(s),
        thrust::device_pointer_cast(d_counts),
        thrust::device_pointer_cast(d_counts + a_row_count),
        thrust::device_pointer_cast(d_offsets));

    return cudaSuccess;
}

cudaError_t enqueueVolumeUnionWrite(
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
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 && b_row_count == 0) {
        return cudaSuccess;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_offsets || !d_r_begin || !d_r_end || !d_r_y_idx) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t s = stream ? stream : nullptr;

    row_union_write_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end);

    return cudaGetLastError();
}

cudaError_t enqueueVolumeDifferenceOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    if (!d_counts || !d_offsets) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 && b_row_count == 0) {
        return cudaSuccess;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t s = stream ? stream : nullptr;

    row_difference_count_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (d_temp_storage && temp_storage_bytes > 0) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                             temp_storage_bytes,
                                             d_counts,
                                             d_offsets,
                                             a_row_count,
                                             s);
    }

    thrust::exclusive_scan(
        thrust::cuda::par.on(s),
        thrust::device_pointer_cast(d_counts),
        thrust::device_pointer_cast(d_counts + a_row_count),
        thrust::device_pointer_cast(d_offsets));

    return cudaSuccess;
}

cudaError_t enqueueVolumeDifferenceWrite(
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
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 && b_row_count == 0) {
        return cudaSuccess;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_offsets || !d_r_begin || !d_r_end || !d_r_y_idx) {
        return cudaErrorInvalidValue;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (a_row_count + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t s = stream ? stream : nullptr;

    row_difference_write_kernel<<<blocksPerGrid, threadsPerBlock, 0, s>>>(
        d_a_begin, d_a_end,
        d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end,
        d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end);

    return cudaGetLastError();
}

cudaError_t enqueueIntervalIntersectionOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return enqueueVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets, stream, d_temp_storage, temp_storage_bytes);
}

cudaError_t enqueueIntervalIntersectionWrite(
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
    int* d_b_idx,
    cudaStream_t stream)
{
    return enqueueVolumeIntersectionWrite(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        d_a_idx,
        d_b_idx,
        stream);
}

cudaError_t enqueueIntervalUnionOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return enqueueVolumeUnionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        stream, d_temp_storage, temp_storage_bytes);
}

cudaError_t enqueueIntervalUnionWrite(
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
    cudaStream_t stream)
{
    return enqueueVolumeUnionWrite(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
}

cudaError_t enqueueIntervalDifferenceOffsets(
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
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return enqueueVolumeDifferenceOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        stream, d_temp_storage, temp_storage_bytes);
}

cudaError_t enqueueIntervalDifferenceWrite(
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
    cudaStream_t stream)
{
    return enqueueVolumeDifferenceWrite(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
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
    int* total_intersections_count,
    cudaStream_t stream)
{
    if (total_intersections_count) {
        *total_intersections_count = 0;
    }

    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 || b_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_counts || !d_offsets) {
        return cudaErrorInvalidValue;
    }

    VolumeIntersectionOffsetsGraphConfig cfg{};
    cudaError_t err = cudaSuccess;
    void* d_scan_temp = nullptr;
    int* d_total = nullptr;
    VolumeIntersectionGraph graph{};
    size_t scan_temp_bytes = 0;

    cudaStream_t cub_stream = stream ? stream : 0;
    err = cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_temp_bytes,
        d_counts,
        d_offsets,
        a_row_count,
        cub_stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    if (scan_temp_bytes == 0) {
        scan_temp_bytes = sizeof(int);
    }

    if (scan_temp_bytes > 0) {
        err = cudaMalloc(&d_scan_temp, scan_temp_bytes);
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    if (total_intersections_count) {
        err = cudaMalloc(&d_total, sizeof(int));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    cfg.d_a_begin = d_a_begin;
    cfg.d_a_end = d_a_end;
    cfg.d_a_row_offsets = d_a_row_offsets;
    cfg.a_row_count = a_row_count;
    cfg.d_b_begin = d_b_begin;
    cfg.d_b_end = d_b_end;
    cfg.d_b_row_offsets = d_b_row_offsets;
    cfg.b_row_count = b_row_count;
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = d_scan_temp;
    cfg.scan_temp_storage_bytes = scan_temp_bytes;
    cfg.d_total = total_intersections_count ? d_total : nullptr;

    err = createVolumeIntersectionOffsetsGraph(&graph, cfg, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    err = launchVolumeIntersectionGraph(graph, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    {
        cudaStream_t sync_stream = stream ? stream : graph.stream;
        err = cudaStreamSynchronize(sync_stream);
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    if (total_intersections_count) {
        err = CUDA_CHECK(cudaMemcpy(total_intersections_count,
                                    d_total,
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

cleanup:
    destroyVolumeIntersectionGraph(&graph);
    if (d_total) {
        cudaFree(d_total);
    }
    if (d_scan_temp) {
        cudaFree(d_scan_temp);
    }
    return err;
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
    int* d_b_idx,
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0 || b_row_count == 0) {
        return cudaSuccess;
    }
    if (!d_a_begin || !d_a_end || !d_a_row_offsets ||
        !d_b_begin || !d_b_end || !d_b_row_offsets ||
        !d_offsets || !d_r_y_idx || !d_r_begin || !d_r_end ||
        !d_a_idx || !d_b_idx) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;
    VolumeIntersectionGraph graph{};

    VolumeIntersectionWriteGraphConfig cfg{};
    cfg.d_a_begin = d_a_begin;
    cfg.d_a_end = d_a_end;
    cfg.d_a_row_offsets = d_a_row_offsets;
    cfg.a_row_count = a_row_count;
    cfg.d_a_row_to_y = d_a_row_to_y;
    cfg.d_a_row_to_z = d_a_row_to_z;
    cfg.d_b_begin = d_b_begin;
    cfg.d_b_end = d_b_end;
    cfg.d_b_row_offsets = d_b_row_offsets;
    cfg.b_row_count = b_row_count;
    cfg.d_offsets = d_offsets;
    cfg.d_r_z_idx = d_r_z_idx;
    cfg.d_r_y_idx = d_r_y_idx;
    cfg.d_r_begin = d_r_begin;
    cfg.d_r_end = d_r_end;
    cfg.d_a_idx = d_a_idx;
    cfg.d_b_idx = d_b_idx;
    cfg.total_capacity = (d_r_begin && d_r_end && d_r_y_idx) ? 1 : 0;

    err = createVolumeIntersectionWriteGraph(&graph, cfg, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    err = launchVolumeIntersectionGraph(graph, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    {
        cudaStream_t sync_stream = stream ? stream : graph.stream;
        err = cudaStreamSynchronize(sync_stream);
    }

cleanup:
    destroyVolumeIntersectionGraph(&graph);
    return err;
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
    int* total_intersections_count,
    cudaStream_t stream)
{
    return computeVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        total_intersections_count,
        stream);
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
    int* d_b_idx,
    cudaStream_t stream)
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
        d_b_idx,
        stream);
}

cudaError_t computeVolumeUnionOffsets(
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
    int* total_intervals_count,
    cudaStream_t stream)
{
    if (total_intervals_count) {
        *total_intervals_count = 0;
    }

    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeUnionOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_counts, d_offsets,
        stream,
        nullptr,
        0);
    if (err != cudaSuccess) {
        return err;
    }

    if (!total_intervals_count) {
        cudaStream_t s = stream ? stream : nullptr;
        return cudaStreamSynchronize(s);
    }

    cudaStream_t s = stream ? stream : nullptr;
    int last_offset = 0;
    int last_count = 0;
    err = CUDA_CHECK(cudaMemcpyAsync(&last_offset,
                                     d_offsets + (a_row_count - 1),
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     s));
    if (err != cudaSuccess) {
        return err;
    }
    err = CUDA_CHECK(cudaMemcpyAsync(&last_count,
                                     d_counts + (a_row_count - 1),
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     s));
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        return err;
    }

    *total_intervals_count = last_offset + last_count;
    return cudaSuccess;
}

cudaError_t writeVolumeUnionWithOffsets(
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
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeUnionWrite(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
    if (err != cudaSuccess) {
        return err;
    }

    cudaStream_t s = stream ? stream : nullptr;
    return cudaStreamSynchronize(s);
}

cudaError_t computeIntervalUnionOffsets(
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
    int* total_intervals_count,
    cudaStream_t stream)
{
    return computeVolumeUnionOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        total_intervals_count,
        stream);
}

cudaError_t writeIntervalUnionWithOffsets(
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
    cudaStream_t stream)
{
    return writeVolumeUnionWithOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
}

cudaError_t computeVolumeDifferenceOffsets(
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
    int* total_intervals_count,
    cudaStream_t stream)
{
    if (total_intervals_count) {
        *total_intervals_count = 0;
    }

    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeDifferenceOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_counts, d_offsets,
        stream,
        nullptr,
        0);
    if (err != cudaSuccess) {
        return err;
    }

    if (!total_intervals_count) {
        cudaStream_t s = stream ? stream : nullptr;
        return cudaStreamSynchronize(s);
    }

    cudaStream_t s = stream ? stream : nullptr;
    int last_offset = 0;
    int last_count = 0;
    err = CUDA_CHECK(cudaMemcpyAsync(&last_offset,
                                     d_offsets + (a_row_count - 1),
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     s));
    if (err != cudaSuccess) {
        return err;
    }
    err = CUDA_CHECK(cudaMemcpyAsync(&last_count,
                                     d_counts + (a_row_count - 1),
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     s));
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        return err;
    }

    *total_intervals_count = last_offset + last_count;
    return cudaSuccess;
}

cudaError_t writeVolumeDifferenceWithOffsets(
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
    cudaStream_t stream)
{
    if (a_row_count < 0 || b_row_count < 0) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count != b_row_count) {
        return cudaErrorInvalidValue;
    }
    if (a_row_count == 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeDifferenceWrite(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
    if (err != cudaSuccess) {
        return err;
    }

    cudaStream_t s = stream ? stream : nullptr;
    return cudaStreamSynchronize(s);
}

cudaError_t computeIntervalDifferenceOffsets(
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
    int* total_intervals_count,
    cudaStream_t stream)
{
    return computeVolumeDifferenceOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        d_counts, d_offsets,
        total_intervals_count,
        stream);
}

cudaError_t writeIntervalDifferenceWithOffsets(
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
    cudaStream_t stream)
{
    return writeVolumeDifferenceWithOffsets(
        d_a_begin, d_a_end, d_a_y_offsets, a_y_count,
        d_b_begin, d_b_end, d_b_y_offsets, b_y_count,
        nullptr, nullptr,
        d_offsets,
        nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        stream);
}


cudaError_t createIntervalIntersectionOffsetsGraph(IntervalIntersectionGraph* graph,
                                                   const IntervalIntersectionOffsetsGraphConfig& config,
                                                   cudaStream_t stream)
{
    if (!graph) {
        return cudaErrorInvalidValue;
    }

    destroyIntervalIntersectionGraph(graph);

    if (config.a_y_count < 0 || config.b_y_count < 0 ||
        config.a_y_count != config.b_y_count) {
        return cudaErrorInvalidValue;
    }

    const bool has_rows = config.a_y_count > 0;
    if (has_rows) {
        if (!config.d_a_begin || !config.d_a_end || !config.d_a_y_offsets ||
            !config.d_b_begin || !config.d_b_end || !config.d_b_y_offsets ||
            !config.d_counts || !config.d_offsets ||
            !config.d_scan_temp_storage || config.scan_temp_storage_bytes == 0) {
            return cudaErrorInvalidValue;
        }
    }

    cudaStream_t capture_stream = stream;
    bool owns_stream = false;
    if (!capture_stream) {
        cudaError_t err_create = cudaStreamCreate(&capture_stream);
        if (err_create != cudaSuccess) {
            return err_create;
        }
        owns_stream = true;
    }

    cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    if (has_rows) {
        err = enqueueIntervalIntersectionOffsets(
            config.d_a_begin, config.d_a_end, config.d_a_y_offsets, config.a_y_count,
            config.d_b_begin, config.d_b_end, config.d_b_y_offsets, config.b_y_count,
            config.d_counts, config.d_offsets,
            capture_stream,
            config.d_scan_temp_storage,
            config.scan_temp_storage_bytes);
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        if (config.d_total) {
            write_total_from_prefix_kernel<<<1, 1, 0, capture_stream>>>(
                config.d_counts,
                config.d_offsets,
                config.a_y_count,
                config.d_total);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaStreamEndCapture(capture_stream, nullptr);
                if (owns_stream) {
                    cudaStreamDestroy(capture_stream);
                }
                return err;
            }
        }
    } else if (config.d_total) {
        write_total_from_prefix_kernel<<<1, 1, 0, capture_stream>>>(
            nullptr,
            nullptr,
            0,
            config.d_total);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }
    }

    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(capture_stream, &captured_graph);
    if (err != cudaSuccess) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    graph->graph = captured_graph;
    graph->exec = exec;
    graph->stream = capture_stream;
    graph->owns_stream = owns_stream;
    return cudaSuccess;
}

cudaError_t createIntervalIntersectionWriteGraph(IntervalIntersectionGraph* graph,
                                                 const IntervalIntersectionWriteGraphConfig& config,
                                                 cudaStream_t stream)
{
    if (!graph) {
        return cudaErrorInvalidValue;
    }

    destroyIntervalIntersectionGraph(graph);

    if (config.a_y_count < 0 || config.b_y_count < 0 ||
        config.a_y_count != config.b_y_count ||
        config.total_capacity < 0) {
        return cudaErrorInvalidValue;
    }

    const bool has_rows = config.a_y_count > 0;
    const bool has_outputs = config.total_capacity > 0;
    if (has_rows) {
        if (!config.d_a_begin || !config.d_a_end || !config.d_a_y_offsets ||
            !config.d_b_begin || !config.d_b_end || !config.d_b_y_offsets ||
            !config.d_offsets) {
            return cudaErrorInvalidValue;
        }
        if (has_outputs) {
            if (!config.d_r_y_idx || !config.d_r_begin || !config.d_r_end ||
                !config.d_a_idx || !config.d_b_idx) {
                return cudaErrorInvalidValue;
            }
        }
    }

    cudaStream_t capture_stream = stream;
    bool owns_stream = false;
    if (!capture_stream) {
        cudaError_t err_create = cudaStreamCreate(&capture_stream);
        if (err_create != cudaSuccess) {
            return err_create;
        }
        owns_stream = true;
    }

    cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    if (has_rows && has_outputs) {
        err = enqueueIntervalIntersectionWrite(
            config.d_a_begin, config.d_a_end, config.d_a_y_offsets, config.a_y_count,
            config.d_b_begin, config.d_b_end, config.d_b_y_offsets, config.b_y_count,
            config.d_offsets,
            config.d_r_y_idx,
            config.d_r_begin,
            config.d_r_end,
            config.d_a_idx,
            config.d_b_idx,
            capture_stream);
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }
    }

    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(capture_stream, &captured_graph);
    if (err != cudaSuccess) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    graph->graph = captured_graph;
    graph->exec = exec;
    graph->stream = capture_stream;
    graph->owns_stream = owns_stream;
    return cudaSuccess;
}

cudaError_t launchIntervalIntersectionGraph(const IntervalIntersectionGraph& graph,
                                            cudaStream_t stream)
{
    if (!graph.exec) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t launch_stream = stream ? stream : graph.stream;
    return cudaGraphLaunch(graph.exec, launch_stream);
}

void destroyIntervalIntersectionGraph(IntervalIntersectionGraph* graph)
{
    if (!graph) {
        return;
    }
    if (graph->exec) {
        cudaGraphExecDestroy(graph->exec);
    }
    if (graph->graph) {
        cudaGraphDestroy(graph->graph);
    }
    if (graph->owns_stream && graph->stream) {
        cudaStreamDestroy(graph->stream);
    }
    graph->graph = nullptr;
    graph->exec = nullptr;
    graph->stream = nullptr;
    graph->owns_stream = false;
}

cudaError_t createVolumeIntersectionOffsetsGraph(VolumeIntersectionGraph* graph,
                                                 const VolumeIntersectionOffsetsGraphConfig& config,
                                                 cudaStream_t stream)
{
    if (!graph) {
        return cudaErrorInvalidValue;
    }

    destroyVolumeIntersectionGraph(graph);

    if (config.a_row_count < 0 || config.b_row_count < 0 ||
        config.a_row_count != config.b_row_count) {
        return cudaErrorInvalidValue;
    }

    const bool has_rows = config.a_row_count > 0;
    if (has_rows) {
        if (!config.d_a_begin || !config.d_a_end || !config.d_a_row_offsets ||
            !config.d_b_begin || !config.d_b_end || !config.d_b_row_offsets ||
            !config.d_counts || !config.d_offsets ||
            !config.d_scan_temp_storage || config.scan_temp_storage_bytes == 0) {
            return cudaErrorInvalidValue;
        }
    }

    cudaStream_t capture_stream = stream;
    bool owns_stream = false;
    if (!capture_stream) {
        cudaError_t err_create = cudaStreamCreate(&capture_stream);
        if (err_create != cudaSuccess) {
            return err_create;
        }
        owns_stream = true;
    }

    cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    if (has_rows) {
        err = enqueueVolumeIntersectionOffsets(
            config.d_a_begin, config.d_a_end, config.d_a_row_offsets, config.a_row_count,
            config.d_b_begin, config.d_b_end, config.d_b_row_offsets, config.b_row_count,
            config.d_counts, config.d_offsets,
            capture_stream,
            config.d_scan_temp_storage,
            config.scan_temp_storage_bytes);
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        if (config.d_total) {
            write_total_from_prefix_kernel<<<1, 1, 0, capture_stream>>>(
                config.d_counts,
                config.d_offsets,
                config.a_row_count,
                config.d_total);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaStreamEndCapture(capture_stream, nullptr);
                if (owns_stream) {
                    cudaStreamDestroy(capture_stream);
                }
                return err;
            }
        }
    } else if (config.d_total) {
        write_total_from_prefix_kernel<<<1, 1, 0, capture_stream>>>(
            nullptr,
            nullptr,
            0,
            config.d_total);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }
    }

    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(capture_stream, &captured_graph);
    if (err != cudaSuccess) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    graph->graph = captured_graph;
    graph->exec = exec;
    graph->stream = capture_stream;
    graph->owns_stream = owns_stream;
    return cudaSuccess;
}

cudaError_t createVolumeIntersectionWriteGraph(VolumeIntersectionGraph* graph,
                                               const VolumeIntersectionWriteGraphConfig& config,
                                               cudaStream_t stream)
{
    if (!graph) {
        return cudaErrorInvalidValue;
    }

    destroyVolumeIntersectionGraph(graph);

    if (config.a_row_count < 0 || config.b_row_count < 0 ||
        config.a_row_count != config.b_row_count ||
        config.total_capacity < 0) {
        return cudaErrorInvalidValue;
    }

    const bool has_rows = config.a_row_count > 0;
    const bool has_outputs = config.total_capacity > 0;

    if (has_rows) {
        if (!config.d_a_begin || !config.d_a_end || !config.d_a_row_offsets ||
            !config.d_b_begin || !config.d_b_end || !config.d_b_row_offsets ||
            !config.d_offsets) {
            return cudaErrorInvalidValue;
        }

        if (has_outputs) {
            if (!config.d_r_y_idx || !config.d_r_begin ||
                !config.d_r_end || !config.d_a_idx || !config.d_b_idx) {
                return cudaErrorInvalidValue;
            }
        }
    }

    cudaStream_t capture_stream = stream;
    bool owns_stream = false;
    if (!capture_stream) {
        cudaError_t err_create = cudaStreamCreate(&capture_stream);
        if (err_create != cudaSuccess) {
            return err_create;
        }
        owns_stream = true;
    }

    cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    if (has_rows && has_outputs) {
        err = enqueueVolumeIntersectionWrite(
            config.d_a_begin, config.d_a_end, config.d_a_row_offsets, config.a_row_count,
            config.d_b_begin, config.d_b_end, config.d_b_row_offsets, config.b_row_count,
            config.d_a_row_to_y, config.d_a_row_to_z,
            config.d_offsets,
            config.d_r_z_idx,
            config.d_r_y_idx,
            config.d_r_begin,
            config.d_r_end,
            config.d_a_idx,
            config.d_b_idx,
            capture_stream);
        if (err != cudaSuccess) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }
    }

    cudaGraph_t captured_graph = nullptr;
    err = cudaStreamEndCapture(capture_stream, &captured_graph);
    if (err != cudaSuccess) {
        if (captured_graph) {
            cudaGraphDestroy(captured_graph);
        }
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(captured_graph);
        if (owns_stream) {
            cudaStreamDestroy(capture_stream);
        }
        return err;
    }

    graph->graph = captured_graph;
    graph->exec = exec;
    graph->stream = capture_stream;
    graph->owns_stream = owns_stream;
    return cudaSuccess;
}

cudaError_t launchVolumeIntersectionGraph(const VolumeIntersectionGraph& graph,
                                          cudaStream_t stream)
{
    if (!graph.exec) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t launch_stream = stream ? stream : graph.stream;
    return cudaGraphLaunch(graph.exec, launch_stream);
}

void destroyVolumeIntersectionGraph(VolumeIntersectionGraph* graph)
{
    if (!graph) {
        return;
    }
    if (graph->exec) {
        cudaGraphExecDestroy(graph->exec);
    }
    if (graph->graph) {
        cudaGraphDestroy(graph->graph);
    }
    if (graph->owns_stream && graph->stream) {
        cudaStreamDestroy(graph->stream);
    }
    graph->graph = nullptr;
    graph->exec = nullptr;
    graph->stream = nullptr;
    graph->owns_stream = false;
}
