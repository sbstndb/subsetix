#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <cstdio>
#include <vector>
#include <cub/device/device_scan.cuh>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"
#include "surface_chain_builder.cuh"
#include "surface_generator.hpp"

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

SurfaceDevice makeDenseSurfaceShifted(int rows,
                                     int intervals_per_row,
                                     int spacing = 4,
                                     int shift = 1 << 20) {
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
            int base = spacing * i + shift;
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

// Copy a host SurfaceIntervals into device buffers compatible with the 2D API.
SurfaceDevice copySurfaceToDevice(const SurfaceIntervals& host) {
    SurfaceDevice device;
    device.interval_count = host.intervalCount();
    device.row_count = host.height;

    const size_t interval_bytes = static_cast<size_t>(device.interval_count) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(host.y_offsets.size()) * sizeof(int);

    if (device.interval_count > 0) {
        CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
        CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
        CUDA_CHECK(cudaMemcpy(device.begin, host.x_begin.data(), interval_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.end, host.x_end.data(), interval_bytes, cudaMemcpyHostToDevice));
    }

    if (!host.y_offsets.empty()) {
        CUDA_CHECK(cudaMalloc(&device.offsets, offsets_bytes));
        CUDA_CHECK(cudaMemcpy(device.offsets, host.y_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    }

    return device;
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

VolumeDevice makeDenseVolumeShifted(int z_dim,
                                   int y_dim,
                                   int intervals_per_row,
                                   int spacing = 4,
                                   int shift = 1 << 20) {
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
                int base = spacing * i + shift;
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

cudaError_t runClassicIntersection2D(const SurfaceDevice& a,
                                     const SurfaceDevice& b) {
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    cudaError_t err = cudaSuccess;
    int total = 0;
    const int rows = a.row_count;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
        if (err != cudaSuccess) goto cleanup;
    }
    err = computeIntervalIntersectionOffsets(
        a.begin, a.end, a.offsets, a.row_count,
        b.begin, b.end, b.offsets, b.row_count,
        d_counts, d_offsets,
        &total,
        nullptr);
    if (err != cudaSuccess) goto cleanup;

    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_r_y_idx, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_r_begin, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_r_end, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_a_idx, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_b_idx, bytes));
        if (err != cudaSuccess) goto cleanup;

        err = writeIntervalIntersectionsWithOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            d_offsets,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            d_a_idx,
            d_b_idx,
            nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

cleanup:
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    return err;
}

cudaError_t runClassicIntersection3D(const VolumeDevice& a,
                                     const VolumeDevice& b) {
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    cudaError_t err = cudaSuccess;
    int total = 0;
    const int rows = a.row_count;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
        if (err != cudaSuccess) goto cleanup;
    }
    err = computeVolumeIntersectionOffsets(
        a.begin, a.end, a.offsets, a.row_count,
        b.begin, b.end, b.offsets, b.row_count,
        d_counts, d_offsets,
        &total,
        nullptr);
    if (err != cudaSuccess) goto cleanup;

    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_r_z_idx, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_r_y_idx, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_r_begin, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_r_end, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_a_idx, bytes));
        if (err != cudaSuccess) goto cleanup;
        err = CUDA_CHECK(cudaMalloc(&d_b_idx, bytes));
        if (err != cudaSuccess) goto cleanup;

        err = writeVolumeIntersectionsWithOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            a.row_to_y, a.row_to_z,
            d_offsets,
            d_r_z_idx,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            d_a_idx,
            d_b_idx,
            nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

cleanup:
    if (d_r_z_idx) CUDA_CHECK(cudaFree(d_r_z_idx));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    return err;
}

float benchmarkClassic2D(const SurfaceDevice& a,
                         const SurfaceDevice& b,
                         int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        cudaError_t err = runClassicIntersection2D(a, b);
        if (err != cudaSuccess) {
            printf("classic 2D iteration failed: %s\n", cudaGetErrorString(err));
            break;
        }
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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // If total == 0, the write path has no capacity and would be invalid.
    // In this empty case, measure the offsets stage only.
    if (total == 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) {
            err = computeIntervalIntersectionOffsets(
                a.begin, a.end, a.offsets, a.row_count,
                b.begin, b.end, b.offsets, b.row_count,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                &total,
                stream);
            if (err != cudaSuccess) {
                printf("computeIntervalIntersectionOffsets (empty) failed: %s\n", cudaGetErrorString(err));
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
        return ms / iterations;
    }

    ResultBuffers buffers = allocResults(total);

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
    if (total == 0) {
        // Measure offsets-only across streams when intersection is empty.
        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < stream_count; ++i) {
                cudaError_t err = computeIntervalIntersectionOffsets(
                    a.begin, a.end, a.offsets, a.row_count,
                    b.begin, b.end, b.offsets, b.row_count,
                    counts[i], offsets[i],
                    &total,
                    streams[i]);
                if (err != cudaSuccess) {
                    printf("computeIntervalIntersectionOffsets (multi-stream 2D empty) failed: %s\n", cudaGetErrorString(err));
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
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
    }
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

float benchmarkSurfaceChainExecutor2D(const SurfaceDevice& a,
                                      const SurfaceDevice& b,
                                      int iterations) {
    subsetix::SurfaceDescriptor a_desc{};
    a_desc.view.d_begin = a.begin;
    a_desc.view.d_end = a.end;
    a_desc.view.d_row_offsets = a.offsets;
    a_desc.view.row_count = a.row_count;
    a_desc.interval_count = a.interval_count;

    subsetix::SurfaceDescriptor b_desc{};
    b_desc.view.d_begin = b.begin;
    b_desc.view.d_end = b.end;
    b_desc.view.d_row_offsets = b.offsets;
    b_desc.view.row_count = b.row_count;
    b_desc.interval_count = b.interval_count;

    subsetix::SurfaceChainBuilder builder;
    auto a_handle = builder.add_input(a_desc);
    auto b_handle = builder.add_input(b_desc);
    auto union_handle = builder.add_union(a_handle, b_handle);
    builder.add_difference(union_handle, b_handle);

    subsetix::SurfaceChainRunner runner(builder);
    CUDA_CHECK(runner.prepare());
    CUDA_CHECK(runner.run());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(runner.run());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

float benchmarkClassic2DSequence(
    const std::vector<std::pair<const SurfaceDevice*, const SurfaceDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& pair : pairs) {
            const SurfaceDevice* a = pair.first;
            const SurfaceDevice* b = pair.second;
            if (!a || !b) {
                continue;
            }

            cudaError_t err = runClassicIntersection2D(*a, *b);
            if (err != cudaSuccess) {
                printf("classic sequence 2D failed: %s\n", cudaGetErrorString(err));
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
    return ms / iterations;
}

float benchmarkWorkspace2DSequence(
    const std::vector<std::pair<const SurfaceDevice*, const SurfaceDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    struct Context {
        int* d_counts = nullptr;
        int* d_offsets = nullptr;
        ResultBuffers buffers;
        cudaStream_t stream = nullptr;
        int total = 0;
        int row_count = 0;
    };

    std::vector<Context> contexts(pairs.size());

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const SurfaceDevice* a = pairs[idx].first;
        const SurfaceDevice* b = pairs[idx].second;
        Context& ctx = contexts[idx];
        if (!a || !b) {
            continue;
        }
        ctx.row_count = a->row_count;
        if (ctx.row_count > 0) {
            const size_t row_bytes = static_cast<size_t>(ctx.row_count) * sizeof(int);
            CUDA_CHECK(cudaMalloc(&ctx.d_counts, row_bytes));
            CUDA_CHECK(cudaMalloc(&ctx.d_offsets, row_bytes));
            CUDA_CHECK(cudaStreamCreate(&ctx.stream));

            cudaError_t err = computeIntervalIntersectionOffsets(
                a->begin, a->end, a->offsets, a->row_count,
                b->begin, b->end, b->offsets, b->row_count,
                ctx.d_counts, ctx.d_offsets,
                &ctx.total,
                ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (err != cudaSuccess) {
                printf("computeIntervalIntersectionOffsets (sequence workspace 2D setup) failed: %s\n", cudaGetErrorString(err));
            }
            if (ctx.total > 0) {
                ctx.buffers = allocResults(ctx.total);
            }
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            const SurfaceDevice* a = pairs[idx].first;
            const SurfaceDevice* b = pairs[idx].second;
            Context& ctx = contexts[idx];
            if (!a || !b || ctx.row_count == 0) {
                continue;
            }

            cudaError_t err = computeIntervalIntersectionOffsets(
                a->begin, a->end, a->offsets, a->row_count,
                b->begin, b->end, b->offsets, b->row_count,
                ctx.d_counts, ctx.d_offsets,
                nullptr,
                ctx.stream);
            if (err != cudaSuccess) {
                printf("computeIntervalIntersectionOffsets (sequence workspace 2D) failed: %s\n", cudaGetErrorString(err));
            }

            if (ctx.total > 0) {
                err = writeIntervalIntersectionsWithOffsets(
                    a->begin, a->end, a->offsets, a->row_count,
                    b->begin, b->end, b->offsets, b->row_count,
                    ctx.d_offsets,
                    ctx.buffers.y_idx,
                    ctx.buffers.begin,
                    ctx.buffers.end,
                    ctx.buffers.a_idx,
                    ctx.buffers.b_idx,
                    ctx.stream);
                if (err != cudaSuccess) {
                    printf("writeIntervalIntersectionsWithOffsets (sequence workspace 2D) failed: %s\n", cudaGetErrorString(err));
                }
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

    for (Context& ctx : contexts) {
        if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        if (ctx.d_counts) CUDA_CHECK(cudaFree(ctx.d_counts));
        if (ctx.d_offsets) CUDA_CHECK(cudaFree(ctx.d_offsets));
        freeResults(ctx.buffers);
    }

    return ms / iterations;
}

float benchmarkGraph2DSequence(
    const std::vector<std::pair<const SurfaceDevice*, const SurfaceDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    struct Context {
        IntervalIntersectionGraph offsets_graph{};
        IntervalIntersectionGraph write_graph{};
        int* d_counts = nullptr;
        int* d_offsets = nullptr;
        void* d_temp_storage = nullptr;
        size_t temp_bytes = 0;
        int* d_total = nullptr;
        ResultBuffers buffers{};
        cudaStream_t stream = nullptr;
        int total_capacity = 0;
        int row_count = 0;
    };

    std::vector<Context> contexts(pairs.size());

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const SurfaceDevice* a = pairs[idx].first;
        const SurfaceDevice* b = pairs[idx].second;
        Context& ctx = contexts[idx];
        if (!a || !b) {
            continue;
        }

        ctx.row_count = a->row_count;
        if (ctx.row_count == 0) {
            continue;
        }

        const size_t row_bytes = static_cast<size_t>(ctx.row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&ctx.d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&ctx.d_offsets, row_bytes));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 ctx.temp_bytes,
                                                 ctx.d_counts,
                                                 ctx.d_offsets,
                                                 ctx.row_count));
        if (ctx.temp_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&ctx.d_temp_storage, ctx.temp_bytes));
        }
        CUDA_CHECK(cudaMalloc(&ctx.d_total, sizeof(int)));

        IntervalIntersectionOffsetsGraphConfig offsets_cfg{};
        offsets_cfg.d_a_begin = a->begin;
        offsets_cfg.d_a_end = a->end;
        offsets_cfg.d_a_y_offsets = a->offsets;
        offsets_cfg.a_y_count = a->row_count;
        offsets_cfg.d_b_begin = b->begin;
        offsets_cfg.d_b_end = b->end;
        offsets_cfg.d_b_y_offsets = b->offsets;
        offsets_cfg.b_y_count = b->row_count;
        offsets_cfg.d_counts = ctx.d_counts;
        offsets_cfg.d_offsets = ctx.d_offsets;
        offsets_cfg.d_scan_temp_storage = ctx.d_temp_storage;
        offsets_cfg.scan_temp_storage_bytes = ctx.temp_bytes;
        offsets_cfg.d_total = ctx.d_total;

        cudaError_t err = createIntervalIntersectionOffsetsGraph(&ctx.offsets_graph, offsets_cfg, ctx.stream);
        if (err != cudaSuccess) {
            printf("createIntervalIntersectionOffsetsGraph (sequence 2D) failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        err = launchIntervalIntersectionGraph(ctx.offsets_graph, ctx.stream);
        if (err != cudaSuccess) {
            printf("launchIntervalIntersectionGraph (offsets sequence 2D setup) failed: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        int total = 0;
        CUDA_CHECK(cudaMemcpy(&total, ctx.d_total, sizeof(int), cudaMemcpyDeviceToHost));
        ctx.total_capacity = total;
        if (ctx.total_capacity > 0) {
            ctx.buffers = allocResults(ctx.total_capacity);
        }

        IntervalIntersectionWriteGraphConfig write_cfg{};
        write_cfg.d_a_begin = a->begin;
        write_cfg.d_a_end = a->end;
        write_cfg.d_a_y_offsets = a->offsets;
        write_cfg.a_y_count = a->row_count;
        write_cfg.d_b_begin = b->begin;
        write_cfg.d_b_end = b->end;
        write_cfg.d_b_y_offsets = b->offsets;
        write_cfg.b_y_count = b->row_count;
        write_cfg.d_offsets = ctx.d_offsets;
        write_cfg.d_r_y_idx = ctx.buffers.y_idx;
        write_cfg.d_r_begin = ctx.buffers.begin;
        write_cfg.d_r_end = ctx.buffers.end;
        write_cfg.d_a_idx = ctx.buffers.a_idx;
        write_cfg.d_b_idx = ctx.buffers.b_idx;
        write_cfg.total_capacity = ctx.total_capacity;

        err = createIntervalIntersectionWriteGraph(&ctx.write_graph, write_cfg, ctx.stream);
        if (err != cudaSuccess) {
            printf("createIntervalIntersectionWriteGraph (sequence 2D) failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        if (ctx.total_capacity > 0) {
            err = launchIntervalIntersectionGraph(ctx.write_graph, ctx.stream);
            if (err != cudaSuccess) {
                printf("launchIntervalIntersectionGraph (write warm-up sequence 2D) failed: %s\n", cudaGetErrorString(err));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (Context& ctx : contexts) {
            if (!ctx.stream || ctx.row_count == 0) {
                continue;
            }
            cudaError_t err = launchIntervalIntersectionGraph(ctx.offsets_graph, ctx.stream);
            if (err != cudaSuccess) {
                printf("launchIntervalIntersectionGraph (offsets sequence 2D) failed: %s\n", cudaGetErrorString(err));
            }
            if (ctx.total_capacity > 0) {
                err = launchIntervalIntersectionGraph(ctx.write_graph, ctx.stream);
                if (err != cudaSuccess) {
                    printf("launchIntervalIntersectionGraph (write sequence 2D) failed: %s\n", cudaGetErrorString(err));
                }
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

    for (Context& ctx : contexts) {
        destroyIntervalIntersectionGraph(&ctx.write_graph);
        destroyIntervalIntersectionGraph(&ctx.offsets_graph);
        if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        if (ctx.d_counts) CUDA_CHECK(cudaFree(ctx.d_counts));
        if (ctx.d_offsets) CUDA_CHECK(cudaFree(ctx.d_offsets));
        if (ctx.d_temp_storage) CUDA_CHECK(cudaFree(ctx.d_temp_storage));
        if (ctx.d_total) CUDA_CHECK(cudaFree(ctx.d_total));
        freeResults(ctx.buffers);
    }

    return ms / iterations;
}

float benchmarkClassic3D(const VolumeDevice& a,
                         const VolumeDevice& b,
                         int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        cudaError_t err = runClassicIntersection3D(a, b);
        if (err != cudaSuccess) {
            printf("classic 3D iteration failed: %s\n", cudaGetErrorString(err));
            break;
        }
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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (total == 0) {
        // Measure offsets-only in empty case.
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) {
            err = computeVolumeIntersectionOffsets(
                a.begin, a.end, a.offsets, a.row_count,
                b.begin, b.end, b.offsets, b.row_count,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                &total,
                stream);
            if (err != cudaSuccess) {
                printf("computeVolumeIntersectionOffsets (empty) failed: %s\n", cudaGetErrorString(err));
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
        return ms / iterations;
    }

    ResultBuffers buffers = allocResults(total);

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
    if (total == 0) {
        // Measure offsets-only across streams when empty.
        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < stream_count; ++i) {
                cudaError_t err = computeVolumeIntersectionOffsets(
                    a.begin, a.end, a.offsets, a.row_count,
                    b.begin, b.end, b.offsets, b.row_count,
                    counts[i], offsets[i],
                    &total,
                    streams[i]);
                if (err != cudaSuccess) {
                    printf("computeVolumeIntersectionOffsets (multi-stream 3D empty) failed: %s\n", cudaGetErrorString(err));
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
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
    }
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

float benchmarkClassic3DSequence(
    const std::vector<std::pair<const VolumeDevice*, const VolumeDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& pair : pairs) {
            const VolumeDevice* a = pair.first;
            const VolumeDevice* b = pair.second;
            if (!a || !b) {
                continue;
            }

            cudaError_t err = runClassicIntersection3D(*a, *b);
            if (err != cudaSuccess) {
                printf("classic sequence 3D failed: %s\n", cudaGetErrorString(err));
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
    return ms / iterations;
}

float benchmarkWorkspace3DSequence(
    const std::vector<std::pair<const VolumeDevice*, const VolumeDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    struct Context {
        int* d_counts = nullptr;
        int* d_offsets = nullptr;
        ResultBuffers buffers;
        cudaStream_t stream = nullptr;
        int total = 0;
        int row_count = 0;
    };

    std::vector<Context> contexts(pairs.size());

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const VolumeDevice* a = pairs[idx].first;
        const VolumeDevice* b = pairs[idx].second;
        Context& ctx = contexts[idx];
        if (!a || !b) {
            continue;
        }

        ctx.row_count = a->row_count;
        if (ctx.row_count > 0) {
            const size_t row_bytes = static_cast<size_t>(ctx.row_count) * sizeof(int);
            CUDA_CHECK(cudaMalloc(&ctx.d_counts, row_bytes));
            CUDA_CHECK(cudaMalloc(&ctx.d_offsets, row_bytes));
            CUDA_CHECK(cudaStreamCreate(&ctx.stream));

            cudaError_t err = computeVolumeIntersectionOffsets(
                a->begin, a->end, a->offsets, a->row_count,
                b->begin, b->end, b->offsets, b->row_count,
                ctx.d_counts, ctx.d_offsets,
                &ctx.total,
                ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (err != cudaSuccess) {
                printf("computeVolumeIntersectionOffsets (sequence workspace 3D setup) failed: %s\n", cudaGetErrorString(err));
            }
            if (ctx.total > 0) {
                ctx.buffers = allocResults(ctx.total);
            }
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            const VolumeDevice* a = pairs[idx].first;
            const VolumeDevice* b = pairs[idx].second;
            Context& ctx = contexts[idx];
            if (!a || !b || ctx.row_count == 0) {
                continue;
            }

            cudaError_t err = computeVolumeIntersectionOffsets(
                a->begin, a->end, a->offsets, a->row_count,
                b->begin, b->end, b->offsets, b->row_count,
                ctx.d_counts, ctx.d_offsets,
                nullptr,
                ctx.stream);
            if (err != cudaSuccess) {
                printf("computeVolumeIntersectionOffsets (sequence workspace 3D) failed: %s\n", cudaGetErrorString(err));
            }

            if (ctx.total > 0) {
                err = writeVolumeIntersectionsWithOffsets(
                    a->begin, a->end, a->offsets, a->row_count,
                    b->begin, b->end, b->offsets, b->row_count,
                    a->row_to_y, a->row_to_z,
                    ctx.d_offsets,
                    ctx.buffers.z_idx,
                    ctx.buffers.y_idx,
                    ctx.buffers.begin,
                    ctx.buffers.end,
                    ctx.buffers.a_idx,
                    ctx.buffers.b_idx,
                    ctx.stream);
                if (err != cudaSuccess) {
                    printf("writeVolumeIntersectionsWithOffsets (sequence workspace 3D) failed: %s\n", cudaGetErrorString(err));
                }
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

    for (Context& ctx : contexts) {
        if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        if (ctx.d_counts) CUDA_CHECK(cudaFree(ctx.d_counts));
        if (ctx.d_offsets) CUDA_CHECK(cudaFree(ctx.d_offsets));
        freeResults(ctx.buffers);
    }

    return ms / iterations;
}

float benchmarkGraph3DSequence(
    const std::vector<std::pair<const VolumeDevice*, const VolumeDevice*>>& pairs,
    int iterations) {
    if (pairs.empty()) {
        return 0.0f;
    }

    struct Context {
        VolumeIntersectionGraph offsets_graph{};
        VolumeIntersectionGraph write_graph{};
        int* d_counts = nullptr;
        int* d_offsets = nullptr;
        void* d_temp_storage = nullptr;
        size_t temp_bytes = 0;
        int* d_total = nullptr;
        ResultBuffers buffers{};
        cudaStream_t stream = nullptr;
        int total_capacity = 0;
        int row_count = 0;
        const int* d_row_to_y = nullptr;
        const int* d_row_to_z = nullptr;
    };

    std::vector<Context> contexts(pairs.size());

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const VolumeDevice* a = pairs[idx].first;
        const VolumeDevice* b = pairs[idx].second;
        Context& ctx = contexts[idx];
        if (!a || !b) {
            continue;
        }

        ctx.row_count = a->row_count;
        ctx.d_row_to_y = a->row_to_y;
        ctx.d_row_to_z = a->row_to_z;
        if (ctx.row_count == 0) {
            continue;
        }

        const size_t row_bytes = static_cast<size_t>(ctx.row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&ctx.d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&ctx.d_offsets, row_bytes));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 ctx.temp_bytes,
                                                 ctx.d_counts,
                                                 ctx.d_offsets,
                                                 ctx.row_count));
        if (ctx.temp_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&ctx.d_temp_storage, ctx.temp_bytes));
        }
        CUDA_CHECK(cudaMalloc(&ctx.d_total, sizeof(int)));

        VolumeIntersectionOffsetsGraphConfig offsets_cfg{};
        offsets_cfg.d_a_begin = a->begin;
        offsets_cfg.d_a_end = a->end;
        offsets_cfg.d_a_row_offsets = a->offsets;
        offsets_cfg.a_row_count = a->row_count;
        offsets_cfg.d_b_begin = b->begin;
        offsets_cfg.d_b_end = b->end;
        offsets_cfg.d_b_row_offsets = b->offsets;
        offsets_cfg.b_row_count = b->row_count;
        offsets_cfg.d_counts = ctx.d_counts;
        offsets_cfg.d_offsets = ctx.d_offsets;
        offsets_cfg.d_scan_temp_storage = ctx.d_temp_storage;
        offsets_cfg.scan_temp_storage_bytes = ctx.temp_bytes;
        offsets_cfg.d_total = ctx.d_total;

        cudaError_t err = createVolumeIntersectionOffsetsGraph(&ctx.offsets_graph, offsets_cfg, ctx.stream);
        if (err != cudaSuccess) {
            printf("createVolumeIntersectionOffsetsGraph (sequence 3D) failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        err = launchVolumeIntersectionGraph(ctx.offsets_graph, ctx.stream);
        if (err != cudaSuccess) {
            printf("launchVolumeIntersectionGraph (offsets sequence 3D setup) failed: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        int total = 0;
        CUDA_CHECK(cudaMemcpy(&total, ctx.d_total, sizeof(int), cudaMemcpyDeviceToHost));
        ctx.total_capacity = total;
        if (ctx.total_capacity > 0) {
            ctx.buffers = allocResults(ctx.total_capacity);
        }

        VolumeIntersectionWriteGraphConfig write_cfg{};
        write_cfg.d_a_begin = a->begin;
        write_cfg.d_a_end = a->end;
        write_cfg.d_a_row_offsets = a->offsets;
        write_cfg.a_row_count = a->row_count;
        write_cfg.d_a_row_to_y = a->row_to_y;
        write_cfg.d_a_row_to_z = a->row_to_z;
        write_cfg.d_b_begin = b->begin;
        write_cfg.d_b_end = b->end;
        write_cfg.d_b_row_offsets = b->offsets;
        write_cfg.b_row_count = b->row_count;
        write_cfg.d_offsets = ctx.d_offsets;
        write_cfg.d_r_z_idx = ctx.buffers.z_idx;
        write_cfg.d_r_y_idx = ctx.buffers.y_idx;
        write_cfg.d_r_begin = ctx.buffers.begin;
        write_cfg.d_r_end = ctx.buffers.end;
        write_cfg.d_a_idx = ctx.buffers.a_idx;
        write_cfg.d_b_idx = ctx.buffers.b_idx;
        write_cfg.total_capacity = ctx.total_capacity;

        err = createVolumeIntersectionWriteGraph(&ctx.write_graph, write_cfg, ctx.stream);
        if (err != cudaSuccess) {
            printf("createVolumeIntersectionWriteGraph (sequence 3D) failed: %s\n", cudaGetErrorString(err));
            continue;
        }

        if (ctx.total_capacity > 0) {
            err = launchVolumeIntersectionGraph(ctx.write_graph, ctx.stream);
            if (err != cudaSuccess) {
                printf("launchVolumeIntersectionGraph (write warm-up sequence 3D) failed: %s\n", cudaGetErrorString(err));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        for (Context& ctx : contexts) {
            if (!ctx.stream || ctx.row_count == 0) {
                continue;
            }
            cudaError_t err = launchVolumeIntersectionGraph(ctx.offsets_graph, ctx.stream);
            if (err != cudaSuccess) {
                printf("launchVolumeIntersectionGraph (offsets sequence 3D) failed: %s\n", cudaGetErrorString(err));
            }
            if (ctx.total_capacity > 0) {
                err = launchVolumeIntersectionGraph(ctx.write_graph, ctx.stream);
                if (err != cudaSuccess) {
                    printf("launchVolumeIntersectionGraph (write sequence 3D) failed: %s\n", cudaGetErrorString(err));
                }
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

    for (Context& ctx : contexts) {
        destroyVolumeIntersectionGraph(&ctx.write_graph);
        destroyVolumeIntersectionGraph(&ctx.offsets_graph);
        if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        if (ctx.d_counts) CUDA_CHECK(cudaFree(ctx.d_counts));
        if (ctx.d_offsets) CUDA_CHECK(cudaFree(ctx.d_offsets));
        if (ctx.d_temp_storage) CUDA_CHECK(cudaFree(ctx.d_temp_storage));
        if (ctx.d_total) CUDA_CHECK(cudaFree(ctx.d_total));
        freeResults(ctx.buffers);
    }

    return ms / iterations;
}

} // namespace

namespace {

int fractionToCoord(int max_value, double fraction) {
    if (fraction < 0.0) fraction = 0.0;
    else if (fraction > 1.0) fraction = 1.0;
    int value = static_cast<int>(std::round(max_value * fraction));
    if (value < 0) value = 0;
    else if (value > max_value) value = max_value;
    return value;
}

SurfaceIntervals buildCompositeRectangles(int width, int height) {
    SurfaceIntervals rects = generateRectangle(
        width, height,
        fractionToCoord(width, 0.05), fractionToCoord(width, 0.35),
        fractionToCoord(height, 0.10), fractionToCoord(height, 0.40));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.30), fractionToCoord(width, 0.55),
        fractionToCoord(height, 0.25), fractionToCoord(height, 0.55)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.60), fractionToCoord(width, 0.90),
        fractionToCoord(height, 0.20), fractionToCoord(height, 0.50)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.15), fractionToCoord(width, 0.40),
        fractionToCoord(height, 0.55), fractionToCoord(height, 0.85)));
    rects = unionSurfaces(rects, generateRectangle(
        width, height,
        fractionToCoord(width, 0.45), fractionToCoord(width, 0.75),
        fractionToCoord(height, 0.05), fractionToCoord(height, 0.25)));
    return rects;
}

SurfaceIntervals buildCompositeCircles(int width, int height,
                                       double dx = 0.0, double dy = 0.0, double dr = 0.0) {
    auto makeCircle = [&](double cx, double cy, double radius_fraction) {
        int center_x = fractionToCoord(width, cx + dx);
        int center_y = fractionToCoord(height, cy + dy);
        int radius = std::max(1, fractionToCoord(width, radius_fraction + dr));
        return generateCircle(width, height, center_x, center_y, radius);
    };

    SurfaceIntervals circles = makeCircle(0.20, 0.30, 0.12);
    circles = unionSurfaces(circles, makeCircle(0.50, 0.20, 0.10));
    circles = unionSurfaces(circles, makeCircle(0.75, 0.60, 0.14));
    circles = unionSurfaces(circles, makeCircle(0.35, 0.75, 0.11));
    circles = unionSurfaces(circles, makeCircle(0.62, 0.48, 0.09));
    return circles;
}

} // namespace

// Run multiple 2D multi-shape intersections concurrently (one stream per pair).
float benchmarkWorkspaceMultiStream2DMultiPairs(
    const std::vector<std::pair<SurfaceDevice, SurfaceDevice>>& pairs,
    int iterations)
{
    const int stream_count = static_cast<int>(pairs.size());
    if (stream_count == 0) return 0.0f;

    std::vector<cudaStream_t> streams(stream_count);
    for (int i = 0; i < stream_count; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    std::vector<int*> d_counts(stream_count, nullptr);
    std::vector<int*> d_offsets(stream_count, nullptr);
    std::vector<ResultBuffers> buffers(stream_count);
    std::vector<int> totals(stream_count, 0);

    // Precompute offsets and allocate outputs per pair.
    for (int i = 0; i < stream_count; ++i) {
        const SurfaceDevice& a = pairs[i].first;
        const SurfaceDevice& b = pairs[i].second;
        const size_t row_bytes = static_cast<size_t>(a.row_count) * sizeof(int);
        if (a.row_count != b.row_count) {
            continue;
        }
        if (a.row_count > 0) {
            CUDA_CHECK(cudaMalloc(&d_counts[i], row_bytes));
            CUDA_CHECK(cudaMalloc(&d_offsets[i], row_bytes));
        }
        int total = 0;
        cudaError_t err = computeIntervalIntersectionOffsets(
            a.begin, a.end, a.offsets, a.row_count,
            b.begin, b.end, b.offsets, b.row_count,
            d_counts[i], d_offsets[i],
            &total,
            streams[i]);
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        if (err != cudaSuccess) {
            printf("computeIntervalIntersectionOffsets (multi-pair 2D) failed: %s\n", cudaGetErrorString(err));
        }
        totals[i] = total;
        if (total > 0) {
            buffers[i] = allocResults(total);
        }
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < stream_count; ++i) {
            const SurfaceDevice& a = pairs[i].first;
            const SurfaceDevice& b = pairs[i].second;
            if (a.row_count == 0 || buffers[i].y_idx == nullptr) {
                continue;
            }
            cudaError_t err = writeIntervalIntersectionsWithOffsets(
                a.begin, a.end, a.offsets, a.row_count,
                b.begin, b.end, b.offsets, b.row_count,
                d_offsets[i],
                buffers[i].y_idx,
                buffers[i].begin,
                buffers[i].end,
                buffers[i].a_idx,
                buffers[i].b_idx,
                streams[i]);
            if (err != cudaSuccess) {
                printf("writeIntervalIntersectionsWithOffsets (multi-pair 2D) failed: %s\n", cudaGetErrorString(err));
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
        if (d_counts[i]) CUDA_CHECK(cudaFree(d_counts[i]));
        if (d_offsets[i]) CUDA_CHECK(cudaFree(d_offsets[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    // Normalize per iteration, per pair.
    return ms / (iterations * stream_count);
}

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
    SurfaceDevice rectMedium = makeDenseSurface(rows2D, intervalsPerRow2D / 4, 8);
    SurfaceDevice circMedium = makeDenseSurface(rows2D, intervalsPerRow2D / 4, 8);
    SurfaceDevice rectSparse = makeDenseSurface(rows2D, intervalsPerRow2D / 16, 16);
    SurfaceDevice circSparse = makeDenseSurface(rows2D, intervalsPerRow2D / 16, 16);
    // Disjoint pair: B is shifted far so intersections are empty
    SurfaceDevice rectDisjointA = makeDenseSurface(rows2D, intervalsPerRow2D);
    SurfaceDevice rectDisjointB = makeDenseSurfaceShifted(rows2D, intervalsPerRow2D);

    VolumeDevice box = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D);
    VolumeDevice sph = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D);
    VolumeDevice boxMedium = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D / 2, 8);
    VolumeDevice sphMedium = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D / 2, 8);
    VolumeDevice boxSmall = makeDenseVolume(smallRows3DZ, smallRows3DY, smallIntervalsPerRow3D);
    VolumeDevice sphSmall = makeDenseVolume(smallRows3DZ, smallRows3DY, smallIntervalsPerRow3D);
    // Disjoint 3D pair
    VolumeDevice boxDisjointA = makeDenseVolume(rows3DZ, rows3DY, intervalsPerRow3D);
    VolumeDevice boxDisjointB = makeDenseVolumeShifted(rows3DZ, rows3DY, intervalsPerRow3D);

    int stream_count = 4;

    std::vector<std::pair<const SurfaceDevice*, const SurfaceDevice*>> surface_pairs = {
        {&rect, &circ},
        {&rectMedium, &circMedium},
        {&rectSparse, &circSparse}
    };

    std::vector<std::pair<const VolumeDevice*, const VolumeDevice*>> volume_pairs = {
        {&box, &sph},
        {&boxMedium, &sphMedium},
        {&boxSmall, &sphSmall}
    };

    float classic2D = benchmarkClassic2D(rect, circ, iterations);
    float workspace2D = benchmarkWorkspaceStream2D(rect, circ, iterations);
    float multiStream2D = benchmarkWorkspaceMultiStream2D(rect, circ, iterations, stream_count);
    float graph2D = benchmarkGraph2D(rect, circ, iterations);
    float chainExec2D = benchmarkSurfaceChainExecutor2D(rect, circ, iterations);
    // Empty intersections (2D)
    float classic2DEmpty = benchmarkClassic2D(rectDisjointA, rectDisjointB, iterations);
    float workspace2DEmpty = benchmarkWorkspaceStream2D(rectDisjointA, rectDisjointB, iterations);
    float multiStream2DEmpty = benchmarkWorkspaceMultiStream2D(rectDisjointA, rectDisjointB, iterations, stream_count);
    float graph2DEmpty = benchmarkGraph2D(rectDisjointA, rectDisjointB, iterations);
    float classic2DSeq = benchmarkClassic2DSequence(surface_pairs, iterations);
    float workspace2DSeq = benchmarkWorkspace2DSequence(surface_pairs, iterations);
    float graph2DSeq = benchmarkGraph2DSequence(surface_pairs, iterations);

    float classic3D = benchmarkClassic3D(box, sph, iterations);
    float workspace3D = benchmarkWorkspaceStream3D(box, sph, iterations);
    float multiStream3D = benchmarkWorkspaceMultiStream3D(box, sph, iterations, stream_count);
    float graph3D = benchmarkGraph3D(box, sph, iterations);
    // Empty intersections (3D)
    float classic3DEmpty = benchmarkClassic3D(boxDisjointA, boxDisjointB, iterations);
    float workspace3DEmpty = benchmarkWorkspaceStream3D(boxDisjointA, boxDisjointB, iterations);
    float multiStream3DEmpty = benchmarkWorkspaceMultiStream3D(boxDisjointA, boxDisjointB, iterations, stream_count);
    float graph3DEmpty = benchmarkGraph3D(boxDisjointA, boxDisjointB, iterations);

    float classic3DSmall = benchmarkClassic3D(boxSmall, sphSmall, iterations);
    float workspace3DSmall = benchmarkWorkspaceStream3D(boxSmall, sphSmall, iterations);
    float multiStream3DSmall = benchmarkWorkspaceMultiStream3D(boxSmall, sphSmall, iterations, stream_count);
    float graph3DSmall = benchmarkGraph3D(boxSmall, sphSmall, iterations);
    float classic3DSeq = benchmarkClassic3DSequence(volume_pairs, iterations);
    float workspace3DSeq = benchmarkWorkspace3DSequence(volume_pairs, iterations);
    float graph3DSeq = benchmarkGraph3DSequence(volume_pairs, iterations);

    printf("Benchmark (%d iterations)\n", iterations);
    printf("2D classic:   %.3f ms/iter\n", classic2D);
    printf("2D workspace: %.3f ms/iter\n", workspace2D);
    printf("2D workspace %d streams: %.3f ms/iter\n", stream_count, multiStream2D);
    printf("2D graph:     %.3f ms/iter\n", graph2D);
    printf("2D (empty) classic:   %.3f ms/iter\n", classic2DEmpty);
    printf("2D (empty) workspace: %.3f ms/iter\n", workspace2DEmpty);
    printf("2D (empty) workspace %d streams: %.3f ms/iter\n", stream_count, multiStream2DEmpty);
    printf("2D (empty) graph:     %.3f ms/iter\n", graph2DEmpty);
    printf("2D seq classic (3 pairs):   %.3f ms/iter\n", classic2DSeq);
    printf("2D seq workspace (3 pairs): %.3f ms/iter\n", workspace2DSeq);
    printf("2D seq graph (3 pairs):     %.3f ms/iter\n", graph2DSeq);
    printf("3D classic:   %.3f ms/iter\n", classic3D);
    printf("3D workspace: %.3f ms/iter\n", workspace3D);
    printf("3D workspace %d streams: %.3f ms/iter\n", stream_count, multiStream3D);
    printf("3D graph:     %.3f ms/iter\n", graph3D);
    printf("3D (empty) classic:   %.3f ms/iter\n", classic3DEmpty);
    printf("3D (empty) workspace: %.3f ms/iter\n", workspace3DEmpty);
    printf("3D (empty) workspace %d streams: %.3f ms/iter\n", stream_count, multiStream3DEmpty);
    printf("3D (empty) graph:     %.3f ms/iter\n", graph3DEmpty);
    printf("3D SMALL classic:   %.3f ms/iter\n", classic3DSmall);
    printf("3D SMALL workspace: %.3f ms/iter\n", workspace3DSmall);
    printf("3D SMALL workspace %d streams: %.3f ms/iter\n", stream_count, multiStream3DSmall);
    printf("3D SMALL graph:     %.3f ms/iter\n", graph3DSmall);
    printf("3D seq classic (3 pairs):   %.3f ms/iter\n", classic3DSeq);
    printf("3D seq workspace (3 pairs): %.3f ms/iter\n", workspace3DSeq);
    printf("3D seq graph (3 pairs):     %.3f ms/iter\n", graph3DSeq);

    // Build multiple 2D multi-shape pairs (rectangles vs circles) to increase occupancy.
    const int width = intervalsPerRow2D * 4;
    const int height = rows2D;
    const int multi_pairs = 8; // one stream per pair
    std::vector<std::pair<SurfaceDevice, SurfaceDevice>> multishape_pairs;
    multishape_pairs.reserve(multi_pairs);
    for (int i = 0; i < multi_pairs; ++i) {
        double dx = (i % 4) * 0.03; // small shifts across pairs
        double dy = (i / 4) * 0.02;
        double dr = (i % 3 == 0) ? -0.01 : 0.0;
        SurfaceIntervals rects = buildCompositeRectangles(width, height);
        SurfaceIntervals circs = buildCompositeCircles(width, height, dx, dy, dr);
        multishape_pairs.emplace_back(copySurfaceToDevice(rects), copySurfaceToDevice(circs));
    }

    float multiShape2DStreams = benchmarkWorkspaceMultiStream2DMultiPairs(multishape_pairs, iterations);
    printf("2D multiforme workspace %d pairs (streams): %.3f ms/iter\n", multi_pairs, multiShape2DStreams);

    for (auto& p : multishape_pairs) {
        freeSurface(p.first);
        freeSurface(p.second);
    }

    // Also measure a single multiforme pair to compare occupancy.
    {
        const int width1 = intervalsPerRow2D * 4;
        const int height1 = rows2D;
        SurfaceIntervals rects1 = buildCompositeRectangles(width1, height1);
        SurfaceIntervals circs1 = buildCompositeCircles(width1, height1);
        int intervals_total = rects1.intervalCount() + circs1.intervalCount();
        std::vector<std::pair<SurfaceDevice, SurfaceDevice>> one_pair;
        one_pair.emplace_back(copySurfaceToDevice(rects1), copySurfaceToDevice(circs1));
        float multiShape2D1 = benchmarkWorkspaceMultiStream2DMultiPairs(one_pair, iterations);
        printf("2D multiforme workspace 1 pair (stream): %.3f ms/iter\n", multiShape2D1);
        if (intervals_total > 0) {
            double ns_per_interval = (multiShape2D1 * 1.0e6) / static_cast<double>(intervals_total);
            printf("2D multiforme workspace 1 pair: %.2f ns/interval (A+B=%d)\n", ns_per_interval, intervals_total);
        }
        for (auto& p : one_pair) {
            freeSurface(p.first);
            freeSurface(p.second);
        }
    }

    freeSurface(rect);
    freeSurface(circ);
    freeSurface(rectMedium);
    freeSurface(circMedium);
    freeSurface(rectSparse);
    freeSurface(circSparse);
    freeSurface(rectDisjointA);
    freeSurface(rectDisjointB);
    freeVolume(box);
    freeVolume(sph);
    freeVolume(boxMedium);
    freeVolume(sphMedium);
    freeVolume(boxSmall);
    freeVolume(sphSmall);
    freeVolume(boxDisjointA);
    freeVolume(boxDisjointB);
    return 0;
}
