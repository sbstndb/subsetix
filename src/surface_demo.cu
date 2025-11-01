#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

} // namespace

int main() {
    constexpr int intervals_per_row = 1024;
    constexpr int width = intervals_per_row * 4;
    constexpr int height = 1024;
    constexpr bool RUN_HOST_UNION = false;
    constexpr bool WRITE_VTK = false;

    try {
        // Build a composite set of five rectangles with varying extents.
        SurfaceIntervals rectangles = generateRectangle(
            width, height,
            fractionToCoord(width, 0.05), fractionToCoord(width, 0.35),
            fractionToCoord(height, 0.10), fractionToCoord(height, 0.40));

        rectangles = unionSurfaces(rectangles, generateRectangle(
            width, height,
            fractionToCoord(width, 0.30), fractionToCoord(width, 0.55),
            fractionToCoord(height, 0.25), fractionToCoord(height, 0.55)));

        rectangles = unionSurfaces(rectangles, generateRectangle(
            width, height,
            fractionToCoord(width, 0.60), fractionToCoord(width, 0.90),
            fractionToCoord(height, 0.20), fractionToCoord(height, 0.50)));

        rectangles = unionSurfaces(rectangles, generateRectangle(
            width, height,
            fractionToCoord(width, 0.15), fractionToCoord(width, 0.40),
            fractionToCoord(height, 0.55), fractionToCoord(height, 0.85)));

        rectangles = unionSurfaces(rectangles, generateRectangle(
            width, height,
            fractionToCoord(width, 0.45), fractionToCoord(width, 0.75),
            fractionToCoord(height, 0.05), fractionToCoord(height, 0.25)));

        // Build a composite set of five circles scattered across the domain.
        auto makeCircle = [&](double cx, double cy, double radius_fraction) {
            int center_x = fractionToCoord(width, cx);
            int center_y = fractionToCoord(height, cy);
            int radius = std::max(1, fractionToCoord(width, radius_fraction));
            return generateCircle(width, height, center_x, center_y, radius);
        };

        SurfaceIntervals circles = makeCircle(0.20, 0.30, 0.12);
        circles = unionSurfaces(circles, makeCircle(0.50, 0.20, 0.10));
        circles = unionSurfaces(circles, makeCircle(0.75, 0.60, 0.14));
        circles = unionSurfaces(circles, makeCircle(0.35, 0.75, 0.11));
        circles = unionSurfaces(circles, makeCircle(0.62, 0.48, 0.09));

        std::cout << "Composite rectangles intervals: " << rectangles.intervalCount() << "\n";
        std::cout << "Composite circles intervals:    " << circles.intervalCount() << "\n";

        [[maybe_unused]] SurfaceIntervals surface_union;
        [[maybe_unused]] double union_ms = 0.0;
        [[maybe_unused]] bool have_union = false;
        if (RUN_HOST_UNION || WRITE_VTK) {
            auto union_start = std::chrono::high_resolution_clock::now();
            surface_union = unionSurfaces(rectangles, circles);
            auto union_end = std::chrono::high_resolution_clock::now();
            union_ms = std::chrono::duration<double, std::milli>(union_end - union_start).count();
            have_union = true;
        }

        DeviceSurface d_rectangles = copyToDevice(rectangles);
        DeviceSurface d_circles = copyToDevice(circles);

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
            d_rectangles.begin, d_rectangles.end, d_rectangles.interval_count,
            d_rectangles.offsets, height,
            d_circles.begin, d_circles.end, d_circles.interval_count,
            d_circles.offsets, height,
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
            freeDeviceSurface(d_rectangles);
            freeDeviceSurface(d_circles);
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
        std::cout << "Total intersections: " << total_intersections << "\n";

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

            auto rect_mask = rasterizeToMask(rectangles);
            auto circle_mask = rasterizeToMask(circles);
            std::vector<int> union_mask = have_union ? rasterizeToMask(surface_union)
                                                     : std::vector<int>(static_cast<size_t>(width) * height, 0);
            auto intersection_mask = rasterizeToMask(surface_intersection);

            writeStructuredPoints("surface_rectangles.vtk", width, height, rect_mask);
            writeStructuredPoints("surface_circles.vtk", width, height, circle_mask);
            if (have_union) {
                writeStructuredPoints("surface_union.vtk", width, height, union_mask);
            }
            writeStructuredPoints("surface_intersection.vtk", width, height, intersection_mask);

            std::cout << "Generated 2D VTK surfaces:\n";
            std::cout << "  surface_rectangles.vtk\n";
            std::cout << "  surface_circles.vtk\n";
            if (have_union) {
                std::cout << "  surface_union.vtk\n";
            }
            std::cout << "  surface_intersection.vtk\n";
        }

        freeIntervalResults(d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
        freeDeviceSurface(d_rectangles);
        freeDeviceSurface(d_circles);
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
