#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <cstdio>
#include <vector>
#include <cub/device/device_scan.cuh>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

namespace {

struct SurfaceDevice {
    int* begin = nullptr;
    int* end = nullptr;
    int* offsets = nullptr;
    int interval_count = 0;
    int row_count = 0;
};

SurfaceDevice makeDenseSurface(int rows, int intervals_per_row, int spacing = 4, int width = 1024) {
    SurfaceDevice device;
    device.interval_count = rows * intervals_per_row;
    device.row_count = rows;

    const size_t interval_bytes = static_cast<size_t>(device.interval_count) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(rows + 1) * sizeof(int);

    std::vector<int> h_begin(device.interval_count);
    std::vector<int> h_end(device.interval_count);
    std::vector<int> h_offsets(rows + 1);

    for (int y = 0; y < rows; ++y) {
        h_offsets[y] = y * intervals_per_row;
        for (int i = 0; i < intervals_per_row; ++i) {
            int idx = y * intervals_per_row + i;
            int base = spacing * i;
            h_begin[idx] = base;
            h_end[idx] = base + spacing / 2;
        }
    }
    h_offsets[rows] = device.interval_count;

    CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.offsets, offsets_bytes));
    CUDA_CHECK(cudaMemcpy(device.begin, h_begin.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.end, h_end.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.offsets, h_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));

    return device;
}

void freeSurface(SurfaceDevice& surface) {
    if (surface.begin) CUDA_CHECK(cudaFree(surface.begin));
    if (surface.end) CUDA_CHECK(cudaFree(surface.end));
    if (surface.offsets) CUDA_CHECK(cudaFree(surface.offsets));
    surface = {};
}

struct VolumeDevice {
    int* begin = nullptr;
    int* end = nullptr;
    int* offsets = nullptr;
    int* row_to_y = nullptr;
    int* row_to_z = nullptr;
    int interval_count = 0;
    int row_count = 0;
    int y_dim = 0;
};

VolumeDevice makeDenseVolume(int z_dim, int y_dim, int intervals_per_row, int spacing = 4) {
    VolumeDevice volume;
    volume.row_count = z_dim * y_dim;
    volume.y_dim = y_dim;
    volume.interval_count = volume.row_count * intervals_per_row;

    const size_t interval_bytes = static_cast<size_t>(volume.interval_count) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(volume.row_count + 1) * sizeof(int);
    const size_t row_map_bytes = static_cast<size_t>(volume.row_count) * sizeof(int);

    std::vector<int> h_begin(volume.interval_count);
    std::vector<int> h_end(volume.interval_count);
    std::vector<int> h_offsets(volume.row_count + 1);
    std::vector<int> h_row_to_y(volume.row_count);
    std::vector<int> h_row_to_z(volume.row_count);

    int write_idx = 0;
    for (int z = 0; z < z_dim; ++z) {
        for (int y = 0; y < y_dim; ++y) {
            int row = z * y_dim + y;
            h_offsets[row] = write_idx;
            h_row_to_y[row] = y;
            h_row_to_z[row] = z;
            for (int i = 0; i < intervals_per_row; ++i) {
                int base = spacing * i;
                h_begin[write_idx] = base;
                h_end[write_idx] = base + spacing / 2;
                ++write_idx;
            }
        }
    }
    h_offsets[volume.row_count] = write_idx;

    CUDA_CHECK(cudaMalloc(&volume.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&volume.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&volume.offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&volume.row_to_y, row_map_bytes));
    CUDA_CHECK(cudaMalloc(&volume.row_to_z, row_map_bytes));

    CUDA_CHECK(cudaMemcpy(volume.begin, h_begin.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(volume.end, h_end.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(volume.offsets, h_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(volume.row_to_y, h_row_to_y.data(), row_map_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(volume.row_to_z, h_row_to_z.data(), row_map_bytes, cudaMemcpyHostToDevice));

    return volume;
}

void freeVolume(VolumeDevice& volume) {
    if (volume.begin) CUDA_CHECK(cudaFree(volume.begin));
    if (volume.end) CUDA_CHECK(cudaFree(volume.end));
    if (volume.offsets) CUDA_CHECK(cudaFree(volume.offsets));
    if (volume.row_to_y) CUDA_CHECK(cudaFree(volume.row_to_y));
    if (volume.row_to_z) CUDA_CHECK(cudaFree(volume.row_to_z));
    volume = {};
}

struct ResultBuffers {
    int* z_idx = nullptr;
    int* y_idx = nullptr;
    int* begin = nullptr;
    int* end = nullptr;
    int* a_idx = nullptr;
    int* b_idx = nullptr;
};

ResultBuffers allocResults(int count) {
    ResultBuffers buffers;
    if (count <= 0) {
        return buffers;
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&buffers.z_idx, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.y_idx, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.begin, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.end, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.a_idx, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.b_idx, bytes));
    return buffers;
}

void freeResults(ResultBuffers& buffers) {
    if (buffers.z_idx) CUDA_CHECK(cudaFree(buffers.z_idx));
    if (buffers.y_idx) CUDA_CHECK(cudaFree(buffers.y_idx));
    if (buffers.begin) CUDA_CHECK(cudaFree(buffers.begin));
    if (buffers.end) CUDA_CHECK(cudaFree(buffers.end));
    if (buffers.a_idx) CUDA_CHECK(cudaFree(buffers.a_idx));
    if (buffers.b_idx) CUDA_CHECK(cudaFree(buffers.b_idx));
    buffers = {};
}

float benchmarkClassic2D(const SurfaceDevice& a,
                         const SurfaceDevice& b,
                         int iterations) {
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        total = 0;
        cudaError_t err = findIntervalIntersections(
            a.begin, a.end, a.interval_count,
            a.offsets, a.row_count,
            b.begin, b.end, b.interval_count,
            b.offsets, b.row_count,
            &d_r_y_idx,
            &d_r_begin, &d_r_end,
            &d_a_idx, &d_b_idx,
            &total);
        if (err != cudaSuccess) {
            printf("findIntervalIntersections failed: %s\n", cudaGetErrorString(err));
            break;
        }
        freeIntervalResults(d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        d_r_y_idx = d_r_begin = d_r_end = d_a_idx = d_b_idx = nullptr;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iterations;
}

float benchmarkWorkspaceStream2D(const SurfaceDevice& a,
                                 const SurfaceDevice& b,
                                 int iterations) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    thrust::device_vector<int> d_counts(a.row_count);
    thrust::device_vector<int> d_offsets(a.row_count);

    int total = 0;
    cudaError_t err = computeIntervalIntersectionOffsets(
        a.begin, a.end, a.offsets, a.row_count,
        b.begin, b.end, b.offsets, b.row_count,
        thrust::raw_pointer_cast(d_counts.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        &total,
        stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (err != cudaSuccess) {
        printf("computeIntervalIntersectionOffsets failed: %s\n", cudaGetErrorString(err));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0.0f;
    }

    ResultBuffers buffers = allocResults(total);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; ++i) {
        err = writeIntervalIntersectionsWithOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            thrust::raw_pointer_cast(d_offsets.data()),
            buffers.y_idx,
            buffers.begin,
            buffers.end,
            buffers.a_idx,
            buffers.b_idx,
            stream);
        if (err != cudaSuccess) {
            printf("writeIntervalIntersectionsWithOffsets failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    freeResults(buffers);
    return ms / iterations;
}

float benchmarkWorkspaceMultiStream2D(const SurfaceDevice& a,
                                      const SurfaceDevice& b,
                                      int iterations,
                                      int stream_count) {
    if (stream_count <= 0) {
        return 0.0f;
    }

    std::vector<cudaStream_t> streams(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    const size_t row_bytes = static_cast<size_t>(a.row_count) * sizeof(int);
    std::vector<int*> counts(stream_count, nullptr);
    std::vector<int*> offsets(stream_count, nullptr);
    std::vector<ResultBuffers> buffers(stream_count);

    int total = 0;
    for (int i = 0; i < stream_count; ++i) {
        if (a.row_count > 0) {
            CUDA_CHECK(cudaMalloc(&counts[i], row_bytes));
            CUDA_CHECK(cudaMalloc(&offsets[i], row_bytes));
        }

        int local_total = 0;
        cudaError_t err = computeIntervalIntersectionOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            counts[i], offsets[i],
            &local_total,
            streams[i]);
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        if (err != cudaSuccess) {
            printf("computeIntervalIntersectionOffsets (multi-stream 2D) failed: %s\n", cudaGetErrorString(err));
        }
        if (i == 0) {
            total = local_total;
        }
    }

    if (total > 0) {
        for (int i = 0; i < stream_count; ++i) {
            buffers[i] = allocResults(total);
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < stream_count; ++i) {
            cudaError_t err = writeIntervalIntersectionsWithOffsets(
                a.begin, a.end, a.offsets, a.row_count,
                b.begin, b.end, b.offsets, b.row_count,
                offsets[i],
                buffers[i].y_idx,
                buffers[i].begin,
                buffers[i].end,
                buffers[i].a_idx,
                buffers[i].b_idx,
                streams[i]);
            if (err != cudaSuccess) {
                printf("writeIntervalIntersectionsWithOffsets (multi-stream 2D) failed: %s\n", cudaGetErrorString(err));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < stream_count; ++i) {
        if (buffers[i].y_idx || buffers[i].z_idx) {
            freeResults(buffers[i]);
        }
        if (counts[i]) CUDA_CHECK(cudaFree(counts[i]));
        if (offsets[i]) CUDA_CHECK(cudaFree(offsets[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return ms / (iterations * stream_count);
}

float benchmarkGraph2D(const SurfaceDevice& a,
                       const SurfaceDevice& b,
                       int iterations) {
    if (a.row_count != b.row_count) {
        return 0.0f;
    }

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (a.row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(a.row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (a.row_count > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 temp_storage_bytes,
                                                 d_counts,
                                                 d_offsets,
                                                 a.row_count));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    cudaStream_t graph_stream;
    CUDA_CHECK(cudaStreamCreate(&graph_stream));

    IntervalIntersectionOffsetsGraphConfig offsets_cfg{};
    offsets_cfg.d_a_begin = a.begin;
    offsets_cfg.d_a_end = a.end;
    offsets_cfg.d_a_y_offsets = a.offsets;
    offsets_cfg.a_y_count = a.row_count;
    offsets_cfg.d_b_begin = b.begin;
    offsets_cfg.d_b_end = b.end;
    offsets_cfg.d_b_y_offsets = b.offsets;
    offsets_cfg.b_y_count = b.row_count;
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_temp_storage;
    offsets_cfg.scan_temp_storage_bytes = temp_storage_bytes;
    offsets_cfg.d_total = d_total;

    IntervalIntersectionGraph offsets_graph{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&offsets_graph, offsets_cfg, graph_stream);
    if (err != cudaSuccess) {
        printf("createIntervalIntersectionOffsetsGraph failed: %s\n", cudaGetErrorString(err));
        if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_total) CUDA_CHECK(cudaFree(d_total));
        CUDA_CHECK(cudaStreamDestroy(graph_stream));
        return 0.0f;
    }

    err = launchIntervalIntersectionGraph(offsets_graph, graph_stream);
    if (err != cudaSuccess) {
        printf("launchIntervalIntersectionGraph (offsets) failed: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    ResultBuffers buffers = allocResults(total);

    IntervalIntersectionWriteGraphConfig write_cfg{};
    write_cfg.d_a_begin = a.begin;
    write_cfg.d_a_end = a.end;
    write_cfg.d_a_y_offsets = a.offsets;
    write_cfg.a_y_count = a.row_count;
    write_cfg.d_b_begin = b.begin;
    write_cfg.d_b_end = b.end;
    write_cfg.d_b_y_offsets = b.offsets;
    write_cfg.b_y_count = b.row_count;
    write_cfg.d_offsets = d_offsets;
    write_cfg.d_r_y_idx = buffers.y_idx;
    write_cfg.d_r_begin = buffers.begin;
    write_cfg.d_r_end = buffers.end;
    write_cfg.d_a_idx = buffers.a_idx;
    write_cfg.d_b_idx = buffers.b_idx;
    write_cfg.total_capacity = total;

    IntervalIntersectionGraph write_graph{};
    err = createIntervalIntersectionWriteGraph(&write_graph, write_cfg, graph_stream);
    if (err != cudaSuccess) {
        printf("createIntervalIntersectionWriteGraph failed: %s\n", cudaGetErrorString(err));
        destroyIntervalIntersectionGraph(&offsets_graph);
        freeResults(buffers);
        if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_total) CUDA_CHECK(cudaFree(d_total));
        CUDA_CHECK(cudaStreamDestroy(graph_stream));
        return 0.0f;
    }

    // Warm-up write path to ensure buffers are populated at least once.
    err = launchIntervalIntersectionGraph(write_graph, graph_stream);
    if (err != cudaSuccess) {
        printf("launchIntervalIntersectionGraph (write warm-up) failed: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, graph_stream));
    for (int i = 0; i < iterations; ++i) {
        err = launchIntervalIntersectionGraph(offsets_graph, graph_stream);
        if (err != cudaSuccess) {
            printf("launchIntervalIntersectionGraph (offsets) failed: %s\n", cudaGetErrorString(err));
            break;
        }
        err = launchIntervalIntersectionGraph(write_graph, graph_stream);
        if (err != cudaSuccess) {
            printf("launchIntervalIntersectionGraph (write) failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, graph_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    destroyIntervalIntersectionGraph(&write_graph);
    destroyIntervalIntersectionGraph(&offsets_graph);
    freeResults(buffers);
    if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
    return ms / iterations;
}

float benchmarkClassic3D(const VolumeDevice& a,
                         const VolumeDevice& b,
                         int iterations) {
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        total = 0;
        cudaError_t err = findVolumeIntersections(
            a.begin, a.end, a.interval_count,
            a.offsets, a.row_to_y, a.row_to_z, a.row_count,
            b.begin, b.end, b.interval_count,
            b.offsets, b.row_count,
            &d_r_z_idx, &d_r_y_idx,
            &d_r_begin, &d_r_end,
            &d_a_idx, &d_b_idx,
            &total);
        if (err != cudaSuccess) {
            printf("findVolumeIntersections failed: %s\n", cudaGetErrorString(err));
            break;
        }
        freeVolumeIntersectionResults(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        d_r_z_idx = d_r_y_idx = d_r_begin = d_r_end = d_a_idx = d_b_idx = nullptr;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iterations;
}

float benchmarkWorkspaceStream3D(const VolumeDevice& a,
                                 const VolumeDevice& b,
                                 int iterations) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    thrust::device_vector<int> d_counts(a.row_count);
    thrust::device_vector<int> d_offsets(a.row_count);

    int total = 0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        a.begin, a.end, a.offsets, a.row_count,
        b.begin, b.end, b.offsets, b.row_count,
        thrust::raw_pointer_cast(d_counts.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        &total,
        stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (err != cudaSuccess) {
        printf("computeVolumeIntersectionOffsets failed: %s\n", cudaGetErrorString(err));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0.0f;
    }

    ResultBuffers buffers = allocResults(total);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; ++i) {
        err = writeVolumeIntersectionsWithOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            a.row_to_y, a.row_to_z,
            thrust::raw_pointer_cast(d_offsets.data()),
            buffers.z_idx,
            buffers.y_idx,
            buffers.begin,
            buffers.end,
            buffers.a_idx,
            buffers.b_idx,
            stream);

        if (err != cudaSuccess) {
            printf("writeVolumeIntersectionsWithOffsets failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    freeResults(buffers);
    return ms / iterations;
}

float benchmarkWorkspaceMultiStream3D(const VolumeDevice& a,
                                      const VolumeDevice& b,
                                      int iterations,
                                      int stream_count) {
    if (stream_count <= 0) {
        return 0.0f;
    }

    std::vector<cudaStream_t> streams(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    const size_t row_bytes = static_cast<size_t>(a.row_count) * sizeof(int);
    std::vector<int*> counts(stream_count, nullptr);
    std::vector<int*> offsets(stream_count, nullptr);
    std::vector<ResultBuffers> buffers(stream_count);

    int total = 0;
    for (int i = 0; i < stream_count; ++i) {
        if (a.row_count > 0) {
            CUDA_CHECK(cudaMalloc(&counts[i], row_bytes));
            CUDA_CHECK(cudaMalloc(&offsets[i], row_bytes));
        }

        int local_total = 0;
        cudaError_t err = computeVolumeIntersectionOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            counts[i], offsets[i],
            &local_total,
            streams[i]);
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        if (err != cudaSuccess) {
            printf("computeVolumeIntersectionOffsets (multi-stream 3D) failed: %s\n", cudaGetErrorString(err));
        }
        if (i == 0) {
            total = local_total;
        }
    }

    if (total > 0) {
        for (int i = 0; i < stream_count; ++i) {
            buffers[i] = allocResults(total);
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < stream_count; ++i) {
            cudaError_t err = writeVolumeIntersectionsWithOffsets(
                a.begin, a.end, a.offsets, a.row_count,
                b.begin, b.end, b.offsets, b.row_count,
                a.row_to_y, a.row_to_z,
                offsets[i],
                buffers[i].z_idx,
                buffers[i].y_idx,
                buffers[i].begin,
                buffers[i].end,
                buffers[i].a_idx,
                buffers[i].b_idx,
                streams[i]);
            if (err != cudaSuccess) {
                printf("writeVolumeIntersectionsWithOffsets (multi-stream 3D) failed: %s\n", cudaGetErrorString(err));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < stream_count; ++i) {
        if (buffers[i].z_idx) freeResults(buffers[i]);
        if (counts[i]) CUDA_CHECK(cudaFree(counts[i]));
        if (offsets[i]) CUDA_CHECK(cudaFree(offsets[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return ms / (iterations * stream_count);
}

float benchmarkGraph3D(const VolumeDevice& a,
                       const VolumeDevice& b,
                       int iterations) {
    if (a.row_count != b.row_count) {
        return 0.0f;
    }

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (a.row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(a.row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (a.row_count > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 temp_storage_bytes,
                                                 d_counts,
                                                 d_offsets,
                                                 a.row_count));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    cudaStream_t graph_stream;
    CUDA_CHECK(cudaStreamCreate(&graph_stream));

    VolumeIntersectionOffsetsGraphConfig offsets_cfg{};
    offsets_cfg.d_a_begin = a.begin;
    offsets_cfg.d_a_end = a.end;
    offsets_cfg.d_a_row_offsets = a.offsets;
    offsets_cfg.a_row_count = a.row_count;
    offsets_cfg.d_b_begin = b.begin;
    offsets_cfg.d_b_end = b.end;
    offsets_cfg.d_b_row_offsets = b.offsets;
    offsets_cfg.b_row_count = b.row_count;
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_temp_storage;
    offsets_cfg.scan_temp_storage_bytes = temp_storage_bytes;
    offsets_cfg.d_total = d_total;

    VolumeIntersectionGraph offsets_graph{};
    cudaError_t err = createVolumeIntersectionOffsetsGraph(&offsets_graph, offsets_cfg, graph_stream);
    if (err != cudaSuccess) {
        printf("createVolumeIntersectionOffsetsGraph failed: %s\n", cudaGetErrorString(err));
        if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_total) CUDA_CHECK(cudaFree(d_total));
        CUDA_CHECK(cudaStreamDestroy(graph_stream));
        return 0.0f;
    }

    err = launchVolumeIntersectionGraph(offsets_graph, graph_stream);
    if (err != cudaSuccess) {
        printf("launchVolumeIntersectionGraph (offsets) failed: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));

    ResultBuffers buffers = allocResults(total);

    VolumeIntersectionWriteGraphConfig write_cfg{};
    write_cfg.d_a_begin = a.begin;
    write_cfg.d_a_end = a.end;
    write_cfg.d_a_row_offsets = a.offsets;
    write_cfg.a_row_count = a.row_count;
    write_cfg.d_a_row_to_y = a.row_to_y;
    write_cfg.d_a_row_to_z = a.row_to_z;
    write_cfg.d_b_begin = b.begin;
    write_cfg.d_b_end = b.end;
    write_cfg.d_b_row_offsets = b.offsets;
    write_cfg.b_row_count = b.row_count;
    write_cfg.d_offsets = d_offsets;
    write_cfg.d_r_z_idx = buffers.z_idx;
    write_cfg.d_r_y_idx = buffers.y_idx;
    write_cfg.d_r_begin = buffers.begin;
    write_cfg.d_r_end = buffers.end;
    write_cfg.d_a_idx = buffers.a_idx;
    write_cfg.d_b_idx = buffers.b_idx;
    write_cfg.total_capacity = total;

    VolumeIntersectionGraph write_graph{};
    err = createVolumeIntersectionWriteGraph(&write_graph, write_cfg, graph_stream);
    if (err != cudaSuccess) {
        printf("createVolumeIntersectionWriteGraph failed: %s\n", cudaGetErrorString(err));
        destroyVolumeIntersectionGraph(&offsets_graph);
        freeResults(buffers);
        if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_total) CUDA_CHECK(cudaFree(d_total));
        CUDA_CHECK(cudaStreamDestroy(graph_stream));
        return 0.0f;
    }

    err = launchVolumeIntersectionGraph(write_graph, graph_stream);
    if (err != cudaSuccess) {
        printf("launchVolumeIntersectionGraph (write warm-up) failed: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, graph_stream));
    for (int i = 0; i < iterations; ++i) {
        err = launchVolumeIntersectionGraph(offsets_graph, graph_stream);
        if (err != cudaSuccess) {
            printf("launchVolumeIntersectionGraph (offsets) failed: %s\n", cudaGetErrorString(err));
            break;
        }
        err = launchVolumeIntersectionGraph(write_graph, graph_stream);
        if (err != cudaSuccess) {
            printf("launchVolumeIntersectionGraph (write) failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, graph_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    destroyVolumeIntersectionGraph(&write_graph);
    destroyVolumeIntersectionGraph(&offsets_graph);
    freeResults(buffers);
    if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
    return ms / iterations;
}

} // namespace

int main() {
    constexpr int rows2D = 1024;
    constexpr int intervalsPerRow2D = 256;
    constexpr int rows3DY = 256;
    constexpr int rows3DZ = 256;
    constexpr int intervalsPerRow3D = 64;
    constexpr int smallRows3DY = 32;
    constexpr int smallRows3DZ = 32;
    constexpr int smallIntervalsPerRow3D = 4;
    constexpr int iterations = 100;

    SurfaceDevice rect = makeDenseSurface(rows2D, intervalsPerRow2D);
    SurfaceDevice circ = makeDenseSurface(rows2D, intervalsPerRow2D);

    VolumeDevice box = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D);
    VolumeDevice sph = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D);
    VolumeDevice boxSmall = makeDenseVolume(smallRows3DZ, smallRows3DY, smallIntervalsPerRow3D);
    VolumeDevice sphSmall = makeDenseVolume(smallRows3DZ, smallRows3DY, smallIntervalsPerRow3D);

    int stream_count = 4;

    float classic2D = benchmarkClassic2D(rect, circ, iterations);
    float workspace2D = benchmarkWorkspaceStream2D(rect, circ, iterations);
    float multiStream2D = benchmarkWorkspaceMultiStream2D(rect, circ, iterations, stream_count);
    float graph2D = benchmarkGraph2D(rect, circ, iterations);

    float classic3D = benchmarkClassic3D(box, sph, iterations);
    float workspace3D = benchmarkWorkspaceStream3D(box, sph, iterations);
    float multiStream3D = benchmarkWorkspaceMultiStream3D(box, sph, iterations, stream_count);
    float graph3D = benchmarkGraph3D(box, sph, iterations);

    float classic3DSmall = benchmarkClassic3D(boxSmall, sphSmall, iterations);
    float workspace3DSmall = benchmarkWorkspaceStream3D(boxSmall, sphSmall, iterations);
    float multiStream3DSmall = benchmarkWorkspaceMultiStream3D(boxSmall, sphSmall, iterations, stream_count);
    float graph3DSmall = benchmarkGraph3D(boxSmall, sphSmall, iterations);

    printf("Benchmark (%d iterations)\n", iterations);
    printf("2D classic:   %.3f ms/iter\n", classic2D);
    printf("2D workspace: %.3f ms/iter\n", workspace2D);
    printf("2D workspace %d streams: %.3f ms/iter\n", stream_count, multiStream2D);
    printf("2D graph:     %.3f ms/iter\n", graph2D);
    printf("3D classic:   %.3f ms/iter\n", classic3D);
    printf("3D workspace: %.3f ms/iter\n", workspace3D);
    printf("3D workspace %d streams: %.3f ms/iter\n", stream_count, multiStream3D);
    printf("3D graph:     %.3f ms/iter\n", graph3D);
    printf("3D SMALL classic:   %.3f ms/iter\n", classic3DSmall);
    printf("3D SMALL workspace: %.3f ms/iter\n", workspace3DSmall);
    printf("3D SMALL workspace %d streams: %.3f ms/iter\n", stream_count, multiStream3DSmall);
    printf("3D SMALL graph:     %.3f ms/iter\n", graph3DSmall);

    freeSurface(rect);
    freeSurface(circ);
    freeVolume(box);
    freeVolume(sph);
    freeVolume(boxSmall);
    freeVolume(sphSmall);
    return 0;
}
