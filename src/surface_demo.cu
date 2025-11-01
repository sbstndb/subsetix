#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

#include "cuda_utils.cuh"
#include "interval_intersection.cuh"
#include "volume_generator.hpp"

struct DeviceVolume {
    int* begin = nullptr;
    int* end = nullptr;
    int* row_offsets = nullptr;
    int* row_to_y = nullptr;
    int* row_to_z = nullptr;
    int row_count = 0;
    int interval_count = 0;
};

DeviceVolume copyToDevice(const VolumeIntervals& volume) {
    DeviceVolume device;
    device.row_count = volume.rowCount();
    device.interval_count = volume.intervalCount();

    const size_t interval_bytes = static_cast<size_t>(device.interval_count) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(device.row_count + 1) * sizeof(int);
    const size_t row_map_bytes = static_cast<size_t>(device.row_count) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.row_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&device.row_to_y, row_map_bytes));
    CUDA_CHECK(cudaMalloc(&device.row_to_z, row_map_bytes));

    CUDA_CHECK(cudaMemcpy(device.begin, volume.x_begin.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.end, volume.x_end.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.row_offsets, volume.row_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.row_to_y, volume.row_to_y.data(), row_map_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.row_to_z, volume.row_to_z.data(), row_map_bytes, cudaMemcpyHostToDevice));

    return device;
}

void freeDeviceVolume(DeviceVolume& volume) {
    if (volume.begin) CUDA_CHECK(cudaFree(volume.begin));
    if (volume.end) CUDA_CHECK(cudaFree(volume.end));
    if (volume.row_offsets) CUDA_CHECK(cudaFree(volume.row_offsets));
    if (volume.row_to_y) CUDA_CHECK(cudaFree(volume.row_to_y));
    if (volume.row_to_z) CUDA_CHECK(cudaFree(volume.row_to_z));
    volume = {};
}

VolumeIntervals volumeFromIntersections(const std::vector<int>& z_idx,
                                        const std::vector<int>& y_idx,
                                        const std::vector<int>& begin,
                                        const std::vector<int>& end,
                                        int width,
                                        int height,
                                        int depth) {
    VolumeIntervals volume;
    volume.width = width;
    volume.height = height;
    volume.depth = depth;
    const int row_count = volume.rowCount();

    volume.row_offsets.assign(row_count + 1, 0);
    volume.row_to_y.resize(row_count, 0);
    volume.row_to_z.resize(row_count, 0);

    std::vector<std::vector<std::pair<int, int>>> per_row(row_count);
    for (size_t idx = 0; idx < begin.size(); ++idx) {
        int z = z_idx[idx];
        int y = y_idx[idx];
        if (z < 0 || z >= depth || y < 0 || y >= height) {
            continue;
        }
        const int row = z * height + y;
        per_row[row].emplace_back(begin[idx], end[idx]);
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            const int row = z * height + y;
            volume.row_to_y[row] = y;
            volume.row_to_z[row] = z;
            volume.row_offsets[row] = static_cast<int>(volume.x_begin.size());

            auto& intervals = per_row[row];
            std::sort(intervals.begin(), intervals.end(),
                      [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
                          return lhs.first < rhs.first;
                      });
            for (const auto& interval : intervals) {
                if (interval.first < interval.second) {
                    volume.x_begin.push_back(interval.first);
                    volume.x_end.push_back(interval.second);
                }
            }
            volume.row_offsets[row + 1] = static_cast<int>(volume.x_begin.size());
        }
    }

    return volume;
}

int main() {
    constexpr int width = 256;
    constexpr int height = 256;
    constexpr int depth = 128;

    try {
        VolumeIntervals box = generateBox(width, height, depth,
                                          32, 200,
                                          40, 220,
                                          20, 100);

        VolumeIntervals sphere = generateSphere(width, height, depth,
                                                width / 2.0f,
                                                height / 2.0f,
                                                depth / 2.0f,
                                                70.0f);

        auto union_start = std::chrono::high_resolution_clock::now();
        VolumeIntervals volume_union = unionVolumes(box, sphere);
        auto union_end = std::chrono::high_resolution_clock::now();
        double union_ms = std::chrono::duration<double, std::milli>(union_end - union_start).count();

        DeviceVolume d_box = copyToDevice(box);
        DeviceVolume d_sphere = copyToDevice(sphere);

        int* d_r_z_idx = nullptr;
        int* d_r_y_idx = nullptr;
        int* d_r_begin = nullptr;
        int* d_r_end = nullptr;
        int* d_a_idx = nullptr;
        int* d_b_idx = nullptr;
        int total_intersections = 0;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        cudaError_t err = findVolumeIntersections(
            d_box.begin, d_box.end, d_box.interval_count,
            d_box.row_offsets, d_box.row_to_y, d_box.row_to_z, d_box.row_count,
            d_sphere.begin, d_sphere.end, d_sphere.interval_count,
            d_sphere.row_offsets, d_sphere.row_count,
            &d_r_z_idx, &d_r_y_idx,
            &d_r_begin, &d_r_end,
            &d_a_idx, &d_b_idx,
            &total_intersections);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        if (err != cudaSuccess) {
            std::cerr << "CUDA intersection failed: " << cudaGetErrorString(err) << "\n";
            freeDeviceVolume(d_box);
            freeDeviceVolume(d_sphere);
            return EXIT_FAILURE;
        }

        double total_ns = static_cast<double>(milliseconds) * 1.0e6;
        double per_interval_ns = total_intersections > 0
                                     ? total_ns / static_cast<double>(total_intersections)
                                     : 0.0;

        std::cout << "Union time (host): " << union_ms << " ms\n";
        std::cout << "Intersection time (device): " << milliseconds << " ms"
                  << " (" << total_ns << " ns total, "
                  << per_interval_ns << " ns/interval)\n";

        std::vector<int> h_r_z_idx(total_intersections);
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);

        if (total_intersections > 0) {
            size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
            CUDA_CHECK(cudaMemcpy(h_r_z_idx.data(), d_r_z_idx, results_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
        }

        VolumeIntervals volume_intersection = volumeFromIntersections(
            h_r_z_idx, h_r_y_idx, h_r_begin, h_r_end, width, height, depth);

        auto box_mask = rasterizeToMask(box);
        auto sphere_mask = rasterizeToMask(sphere);
        auto union_mask = rasterizeToMask(volume_union);
        auto intersection_mask = rasterizeToMask(volume_intersection);

        writeStructuredPoints("volume_box.vtk", width, height, depth, box_mask);
        writeStructuredPoints("volume_sphere.vtk", width, height, depth, sphere_mask);
        writeStructuredPoints("volume_union.vtk", width, height, depth, union_mask);
        writeStructuredPoints("volume_intersection.vtk", width, height, depth, intersection_mask);

        std::cout << "Generated VTK volumes:\n"
                  << "  volume_box.vtk\n"
                  << "  volume_sphere.vtk\n"
                  << "  volume_union.vtk\n"
                  << "  volume_intersection.vtk\n";

        freeIntersectionResults(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        freeDeviceVolume(d_box);
        freeDeviceVolume(d_sphere);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
