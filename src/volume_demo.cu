#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

namespace {

int fractionToCoord(int max_value, double fraction) {
    if (fraction < 0.0) {
        fraction = 0.0;
    } else if (fraction > 1.0) {
        fraction = 1.0;
    }
    int value = static_cast<int>(std::round(max_value * fraction));
    if (value < 0) {
        value = 0;
    } else if (value > max_value) {
        value = max_value;
    }
    return value;
}

float fractionToFloatCoord(int max_value, double fraction) {
    if (fraction < 0.0) {
        fraction = 0.0;
    } else if (fraction > 1.0) {
        fraction = 1.0;
    }
    return static_cast<float>(max_value * fraction);
}

} // namespace

int main() {
    constexpr int width = 1024;
    constexpr int height = 1024;
    constexpr int depth = 1024;
    constexpr bool WRITE_VTK = false; // disable heavy IO for high-res timing

    try {
        // Build a composite of five axis-aligned boxes.
        auto makeBox = [&](double x0, double x1,
                           double y0, double y1,
                           double z0, double z1) {
            return generateBox(width, height, depth,
                               fractionToCoord(width, x0), fractionToCoord(width, x1),
                               fractionToCoord(height, y0), fractionToCoord(height, y1),
                               fractionToCoord(depth, z0), fractionToCoord(depth, z1));
        };

        VolumeIntervals boxes = makeBox(0.05, 0.25, 0.10, 0.30, 0.10, 0.30);
        boxes = unionVolumes(boxes, makeBox(0.30, 0.55, 0.20, 0.50, 0.20, 0.45));
        boxes = unionVolumes(boxes, makeBox(0.50, 0.75, 0.40, 0.70, 0.35, 0.65));
        boxes = unionVolumes(boxes, makeBox(0.15, 0.40, 0.55, 0.85, 0.55, 0.85));
        boxes = unionVolumes(boxes, makeBox(0.60, 0.85, 0.15, 0.35, 0.70, 0.90));

        // Build a composite of five spheres located in different regions.
        auto makeSphere = [&](double cx, double cy, double cz, double radius_fraction) {
            if (radius_fraction < 0.01) {
                radius_fraction = 0.01;
            } else if (radius_fraction > 0.25) {
                radius_fraction = 0.25;
            }
            return generateSphere(
                width, height, depth,
                fractionToFloatCoord(width, cx),
                fractionToFloatCoord(height, cy),
                fractionToFloatCoord(depth, cz),
                static_cast<float>(width * radius_fraction));
        };

        VolumeIntervals spheres = makeSphere(0.30, 0.35, 0.40, 0.12);
        spheres = unionVolumes(spheres, makeSphere(0.65, 0.35, 0.35, 0.10));
        spheres = unionVolumes(spheres, makeSphere(0.40, 0.65, 0.60, 0.11));
        spheres = unionVolumes(spheres, makeSphere(0.72, 0.68, 0.75, 0.13));
        spheres = unionVolumes(spheres, makeSphere(0.22, 0.72, 0.55, 0.09));

        std::cout << "Composite boxes intervals:   " << boxes.intervalCount() << "\n";
        std::cout << "Composite spheres intervals: " << spheres.intervalCount() << "\n";

        auto union_start = std::chrono::high_resolution_clock::now();
        VolumeIntervals volume_union = unionVolumes(boxes, spheres);
        auto union_end = std::chrono::high_resolution_clock::now();
        double union_ms = std::chrono::duration<double, std::milli>(union_end - union_start).count();

        DeviceVolume d_boxes = copyToDevice(boxes);
        DeviceVolume d_spheres = copyToDevice(spheres);

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
            d_boxes.begin, d_boxes.end, d_boxes.interval_count,
            d_boxes.row_offsets, d_boxes.row_to_y, d_boxes.row_to_z, d_boxes.row_count,
            d_spheres.begin, d_spheres.end, d_spheres.interval_count,
            d_spheres.row_offsets, d_spheres.row_count,
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
            freeDeviceVolume(d_boxes);
            freeDeviceVolume(d_spheres);
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
        std::cout << "Total intersections: " << total_intersections << "\n";

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

        if (WRITE_VTK) {
            auto boxes_mask = rasterizeToMask(boxes);
            auto spheres_mask = rasterizeToMask(spheres);
            auto union_mask = rasterizeToMask(volume_union);
            auto intersection_mask = rasterizeToMask(volume_intersection);

            writeStructuredPoints("volume_boxes.vtk", width, height, depth, boxes_mask);
            writeStructuredPoints("volume_spheres.vtk", width, height, depth, spheres_mask);
            writeStructuredPoints("volume_union.vtk", width, height, depth, union_mask);
            writeStructuredPoints("volume_intersection.vtk", width, height, depth, intersection_mask);

            std::cout << "Generated 3D VTK volumes:\n";
            std::cout << "  volume_boxes.vtk\n";
            std::cout << "  volume_spheres.vtk\n";
            std::cout << "  volume_union.vtk\n";
            std::cout << "  volume_intersection.vtk\n";
        }

        freeVolumeIntersectionResults(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        freeDeviceVolume(d_boxes);
        freeDeviceVolume(d_spheres);
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
