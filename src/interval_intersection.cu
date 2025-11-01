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

    if (a_row_count <= 0 || b_row_count <= 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_counts, d_offsets,
        stream,
        nullptr,
        0);
    if (err != cudaSuccess) {
        return err;
    }

    cudaStream_t s = stream ? stream : nullptr;

    if (!total_intersections_count) {
        return cudaStreamSynchronize(s);
    }

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

    *total_intersections_count = last_offset + last_count;
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
    int* d_b_idx,
    cudaStream_t stream)
{
    if (a_row_count <= 0 || b_row_count <= 0) {
        return cudaSuccess;
    }

    cudaError_t err = enqueueVolumeIntersectionWrite(
        d_a_begin, d_a_end, d_a_row_offsets, a_row_count,
        d_b_begin, d_b_end, d_b_row_offsets, b_row_count,
        d_a_row_to_y, d_a_row_to_z,
        d_offsets,
        d_r_z_idx,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        d_a_idx,
        d_b_idx,
        stream);
    if (err != cudaSuccess) {
        return err;
    }

    cudaStream_t s = stream ? stream : nullptr;
    return cudaStreamSynchronize(s);
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
            if (!config.d_r_z_idx || !config.d_r_y_idx || !config.d_r_begin ||
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


cudaError_t findVolumeIntersections(
    const int* d_a_begin, const int* d_a_end, int a_interval_count,
    const int* d_a_row_offsets, const int* d_a_row_to_y, const int* d_a_row_to_z, int a_row_count,
    const int* d_b_begin, const int* d_b_end, int b_interval_count,
    const int* d_b_row_offsets, int b_row_count,
    int** d_r_z_idx, int** d_r_y_idx, int** d_r_begin, int** d_r_end,
    int** d_a_idx, int** d_b_idx,
    int* total_intersections_count,
    cudaStream_t stream)
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
        &computed_total,
        stream);
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
        *d_b_idx,
        stream);

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
    int* total_intersections_count,
    cudaStream_t stream)
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
        &computed_total,
        stream);
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
        *d_b_idx,
        stream);

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
