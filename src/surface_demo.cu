#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

#include "cuda_utils.cuh"
#include "interval_intersection.cuh"
#include "surface_generator.hpp"

struct DeviceSurface {
    int* begin = nullptr;
    int* end = nullptr;
    int* offsets = nullptr;
    int interval_count = 0;
};

DeviceSurface copyToDevice(const SurfaceIntervals& surface) {
    DeviceSurface device;
    device.interval_count = surface.intervalCount();

    const size_t interval_bytes = static_cast<size_t>(surface.intervalCount()) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(surface.y_offsets.size()) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.offsets, offsets_bytes));

    CUDA_CHECK(cudaMemcpy(device.begin, surface.x_begin.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.end, surface.x_end.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.offsets, surface.y_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));

    return device;
}

void freeDeviceSurface(DeviceSurface& device) {
    if (device.begin) CUDA_CHECK(cudaFree(device.begin));
    if (device.end) CUDA_CHECK(cudaFree(device.end));
    if (device.offsets) CUDA_CHECK(cudaFree(device.offsets));
    device = {};
}

SurfaceIntervals surfaceFromIntersections(const std::vector<int>& y_idx,
                                          const std::vector<int>& begin,
                                          const std::vector<int>& end,
                                          int width,
                                          int height) {
    SurfaceIntervals surface;
    surface.width = width;
    surface.height = height;
    surface.y_offsets.assign(height + 1, 0);

    std::vector<std::vector<std::pair<int, int>>> per_row(height);
    for (size_t i = 0; i < y_idx.size(); ++i) {
        int y = y_idx[i];
        if (y < 0 || y >= height) {
            continue;
        }
        per_row[y].emplace_back(begin[i], end[i]);
    }

    for (int y = 0; y < height; ++y) {
        surface.y_offsets[y] = static_cast<int>(surface.x_begin.size());
        auto& intervals = per_row[y];
        std::sort(intervals.begin(), intervals.end(),
                  [](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
                      return lhs.first < rhs.first;
                  });
        for (const auto& interval : intervals) {
            if (interval.first < interval.second) {
                surface.x_begin.push_back(interval.first);
                surface.x_end.push_back(interval.second);
            }
        }
        surface.y_offsets[y + 1] = static_cast<int>(surface.x_begin.size());
    }

    return surface;
}

int main() {
    constexpr int intervals_per_row = 1024;
    constexpr int width = intervals_per_row * 4;
    constexpr int height = 1024;
    constexpr bool RUN_HOST_UNION = false;
    constexpr bool WRITE_VTK = false;
    constexpr int total_intervals = height * intervals_per_row;

    try {
        SurfaceIntervals rectangle;
        rectangle.width = width;
        rectangle.height = height;
        rectangle.y_offsets.resize(height + 1);
        rectangle.x_begin.resize(total_intervals);
        rectangle.x_end.resize(total_intervals);

        SurfaceIntervals circle;
        circle.width = width;
        circle.height = height;
        circle.y_offsets.resize(height + 1);
        circle.x_begin.resize(total_intervals);
        circle.x_end.resize(total_intervals);

        for (int y = 0; y < height; ++y) {
            rectangle.y_offsets[y] = y * intervals_per_row;
            circle.y_offsets[y] = y * intervals_per_row;
            for (int i = 0; i < intervals_per_row; ++i) {
                int idx = y * intervals_per_row + i;
                int base = 4 * i;
                rectangle.x_begin[idx] = base;
                rectangle.x_end[idx] = base + 2;
                circle.x_begin[idx] = base + 1;
                circle.x_end[idx] = base + 3;
            }
        }
        rectangle.y_offsets[height] = total_intervals;
        circle.y_offsets[height] = total_intervals;

        [[maybe_unused]] SurfaceIntervals surface_union;
        [[maybe_unused]] double union_ms = 0.0;
        [[maybe_unused]] bool have_union = false;
        if (RUN_HOST_UNION || WRITE_VTK) {
            auto union_start = std::chrono::high_resolution_clock::now();
            surface_union = unionSurfaces(rectangle, circle);
            auto union_end = std::chrono::high_resolution_clock::now();
            union_ms = std::chrono::duration<double, std::milli>(union_end - union_start).count();
            have_union = true;
        }

        DeviceSurface d_rect = copyToDevice(rectangle);
        DeviceSurface d_circle = copyToDevice(circle);

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

        cudaError_t err = findIntervalIntersections(
            d_rect.begin, d_rect.end, d_rect.interval_count,
            d_rect.offsets, height,
            d_circle.begin, d_circle.end, d_circle.interval_count,
            d_circle.offsets, height,
            &d_r_y_idx,
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
            freeDeviceSurface(d_rect);
            freeDeviceSurface(d_circle);
            return EXIT_FAILURE;
        }

        double total_ns = static_cast<double>(milliseconds) * 1.0e6;
        double per_interval_ns = total_intersections > 0
                                     ? total_ns / static_cast<double>(total_intersections)
                                     : 0.0;

        if (RUN_HOST_UNION && have_union) {
            std::cout << "Union time (host): " << union_ms << " ms\n";
        }
        std::cout << "Intersection time (device): " << milliseconds << " ms"
                  << " (" << total_ns << " ns total, "
                  << per_interval_ns << " ns/interval)\n";

        if (WRITE_VTK) {
            std::vector<int> h_r_y_idx(total_intersections);
            std::vector<int> h_r_begin(total_intersections);
            std::vector<int> h_r_end(total_intersections);

            if (total_intersections > 0) {
                const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
                CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
            }

            SurfaceIntervals surface_intersection = surfaceFromIntersections(
                h_r_y_idx, h_r_begin, h_r_end, width, height);

            auto rect_mask = rasterizeToMask(rectangle);
            auto circle_mask = rasterizeToMask(circle);
            auto union_mask = rasterizeToMask(surface_union);
            auto intersection_mask = rasterizeToMask(surface_intersection);

            writeStructuredPoints("surface_rectangle.vtk", width, height, rect_mask);
            writeStructuredPoints("surface_circle.vtk", width, height, circle_mask);
            writeStructuredPoints("surface_union.vtk", width, height, union_mask);
            writeStructuredPoints("surface_intersection.vtk", width, height, intersection_mask);

            std::cout << "Generated 2D VTK surfaces:\n";
            std::cout << "  surface_rectangle.vtk\n";
            std::cout << "  surface_circle.vtk\n";
            std::cout << "  surface_union.vtk\n";
            std::cout << "  surface_intersection.vtk\n";
        }

        freeIntervalResults(d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        freeDeviceSurface(d_rect);
        freeDeviceSurface(d_circle);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
