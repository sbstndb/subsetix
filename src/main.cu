#include <stdio.h>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

int main() {
    // Simple 3D grid example
    const int z_count = 4;
    const int y_count = 4;
    const int intervals_per_row = 1024;
    const int row_count = z_count * y_count;
    const int total_intervals = row_count * intervals_per_row;

    printf("3D Example: rows=%d (z=%d, y=%d), intervals/row=%d\n",
           row_count, z_count, y_count, intervals_per_row);

    std::vector<int> h_a_row_offsets(row_count + 1, 0);
    std::vector<int> h_b_row_offsets(row_count + 1, 0);
    std::vector<int> h_row_to_y(row_count, 0);
    std::vector<int> h_row_to_z(row_count, 0);

    std::vector<int> h_a_begin(total_intervals);
    std::vector<int> h_a_end(total_intervals);
    std::vector<int> h_b_begin(total_intervals);
    std::vector<int> h_b_end(total_intervals);

    printf("Initializing host data...\n");

    int write_idx = 0;
    for (int z = 0; z < z_count; ++z) {
        for (int y = 0; y < y_count; ++y) {
            const int row = z * y_count + y;
            h_row_to_y[row] = y;
            h_row_to_z[row] = z;
            h_a_row_offsets[row] = write_idx;
            h_b_row_offsets[row] = write_idx;

            const int base = z * 1000000 + y * 10000;
            for (int i = 0; i < intervals_per_row; ++i) {
                const int idx = write_idx + i;
                h_a_begin[idx] = base + 4 * i;
                h_a_end[idx]   = base + 4 * i + 2;
                h_b_begin[idx] = base + 4 * i + 1;
                h_b_end[idx]   = base + 4 * i + 3;
            }

            write_idx += intervals_per_row;
        }
    }
    h_a_row_offsets[row_count] = write_idx;
    h_b_row_offsets[row_count] = write_idx;

    const size_t interval_bytes = static_cast<size_t>(total_intervals) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(row_count + 1) * sizeof(int);
    const size_t row_map_bytes = static_cast<size_t>(row_count) * sizeof(int);

    // Device allocations
    printf("Allocating and copying inputs to device...\n");
    int *d_a_begin = nullptr, *d_a_end = nullptr;
    int *d_b_begin = nullptr, *d_b_end = nullptr;
    int *d_a_row_offsets = nullptr, *d_b_row_offsets = nullptr;
    int *d_row_to_y = nullptr, *d_row_to_z = nullptr;

    cudaMalloc(&d_a_begin, interval_bytes);
    cudaMalloc(&d_a_end, interval_bytes);
    cudaMalloc(&d_b_begin, interval_bytes);
    cudaMalloc(&d_b_end, interval_bytes);
    cudaMalloc(&d_a_row_offsets, offsets_bytes);
    cudaMalloc(&d_b_row_offsets, offsets_bytes);
    cudaMalloc(&d_row_to_y, row_map_bytes);
    cudaMalloc(&d_row_to_z, row_map_bytes);

    cudaMemcpy(d_a_begin, h_a_begin.data(), interval_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end,   h_a_end.data(), interval_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin.data(), interval_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end,   h_b_end.data(), interval_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_row_offsets, h_a_row_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_row_offsets, h_b_row_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_to_y, h_row_to_y.data(), row_map_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_to_z, h_row_to_z.data(), row_map_bytes, cudaMemcpyHostToDevice);

    printf("Calling library function...\n");
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_intersections = 0;

    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        cudaMalloc(&d_counts, row_bytes);
        cudaMalloc(&d_offsets, row_bytes);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a_begin, d_a_end, d_a_row_offsets, row_count,
        d_b_begin, d_b_end, d_b_row_offsets, row_count,
        d_counts, d_offsets,
        &total_intersections,
        nullptr);

    if (err == cudaSuccess && total_intersections > 0) {
        const size_t bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        cudaMalloc(&d_r_z_idx, bytes);
        cudaMalloc(&d_r_y_idx, bytes);
        cudaMalloc(&d_r_begin, bytes);
        cudaMalloc(&d_r_end, bytes);
        cudaMalloc(&d_a_idx, bytes);
        cudaMalloc(&d_b_idx, bytes);
        err = writeVolumeIntersectionsWithOffsets(
            d_a_begin, d_a_end, d_a_row_offsets, row_count,
            d_b_begin, d_b_end, d_b_row_offsets, row_count,
            d_row_to_y, d_row_to_z,
            d_offsets,
            d_r_z_idx, d_r_y_idx,
            d_r_begin, d_r_end,
            d_a_idx, d_b_idx,
            nullptr);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (err != cudaSuccess) {
        printf("Intersection failed: %s\n", cudaGetErrorString(err));
    }

    printf("Time taken by intersection: %.5f ms\n", milliseconds);
    printf("Time taken per interval: %.5f ns\n",
           1000.0f * 1000.0f * milliseconds / total_intervals);
    printf("Total intersections found: %d\n", total_intersections);

    if (total_intersections > 0) {
        printf("Copying results Device -> Host...\n");
        std::vector<int> h_r_z_idx(total_intersections);
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);

        cudaMemcpy(h_r_z_idx.data(), d_r_z_idx, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_a_idx.data(), d_a_idx, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_idx.data(), d_b_idx, results_bytes, cudaMemcpyDeviceToHost);

        const int display_count = (total_intersections < 5) ? total_intersections : 5;
        printf("First %d intersections:\n", display_count);
        for (int i = 0; i < display_count; ++i) {
            printf(" -- Intersection %d: Z=%d, Y=%d, A[%d] & B[%d] -> [%d, %d)\n",
                   i, h_r_z_idx[i], h_r_y_idx[i], h_a_idx[i], h_b_idx[i], h_r_begin[i], h_r_end[i]);
        }
    }

    printf("Freeing arrays...\n");
    if (d_r_z_idx) cudaFree(d_r_z_idx);
    if (d_r_y_idx) cudaFree(d_r_y_idx);
    if (d_r_begin) cudaFree(d_r_begin);
    if (d_r_end) cudaFree(d_r_end);
    if (d_a_idx) cudaFree(d_a_idx);
    if (d_b_idx) cudaFree(d_b_idx);
    if (d_counts) cudaFree(d_counts);
    if (d_offsets) cudaFree(d_offsets);
    cudaFree(d_a_begin);
    cudaFree(d_a_end);
    cudaFree(d_b_begin);
    cudaFree(d_b_end);
    cudaFree(d_a_row_offsets);
    cudaFree(d_b_row_offsets);
    cudaFree(d_row_to_y);
    cudaFree(d_row_to_z);

    return EXIT_SUCCESS;
}
