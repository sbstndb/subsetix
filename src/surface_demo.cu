#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "surface_chain_builder.cuh"
#include "surface_generator.hpp"

namespace
{
    struct DeviceSurface
    {
        int* d_begin = nullptr;
        int* d_end = nullptr;
        int* d_offsets = nullptr;
        int interval_count = 0;
        int row_count = 0;
    };

    DeviceSurface copy_surface_to_device(const SurfaceIntervals& surface)
    {
        DeviceSurface device;
        device.interval_count = surface.intervalCount();
        device.row_count = surface.height;
        if (surface.intervalCount() == 0) {
            CUDA_CHECK(cudaMalloc(&device.d_offsets, (surface.height + 1) * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(device.d_offsets,
                                  surface.y_offsets.data(),
                                  (surface.height + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            return device;
        }

        const size_t interval_bytes = static_cast<size_t>(surface.intervalCount()) * sizeof(int);
        const size_t offsets_bytes = static_cast<size_t>(surface.height + 1) * sizeof(int);

        CUDA_CHECK(cudaMalloc(&device.d_begin, interval_bytes));
        CUDA_CHECK(cudaMalloc(&device.d_end, interval_bytes));
        CUDA_CHECK(cudaMalloc(&device.d_offsets, offsets_bytes));

        CUDA_CHECK(cudaMemcpy(device.d_begin,
                              surface.x_begin.data(),
                              interval_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.d_end,
                              surface.x_end.data(),
                              interval_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.d_offsets,
                              surface.y_offsets.data(),
                              offsets_bytes,
                              cudaMemcpyHostToDevice));
        return device;
    }

    void free_device_surface(DeviceSurface& surface)
    {
        if (surface.d_begin) CUDA_CHECK(cudaFree(surface.d_begin));
        if (surface.d_end) CUDA_CHECK(cudaFree(surface.d_end));
        if (surface.d_offsets) CUDA_CHECK(cudaFree(surface.d_offsets));
        surface = {};
    }
}

int main()
{
    const int width = 2048;
    const int height = 1024;

    auto rectangles = generateRectangle(width, height, width / 20, width / 3, height / 10, height / 3);
    rectangles = unionSurfaces(rectangles, generateRectangle(width, height, width / 4, width / 2, height / 4, height / 2));
    rectangles = unionSurfaces(rectangles, generateRectangle(width, height, width / 2, width * 3 / 4, height / 3, height * 2 / 3));

    auto makeCircle = [&](double cx, double cy, double radius_fraction) {
        int center_x = static_cast<int>(cx * width);
        int center_y = static_cast<int>(cy * height);
        int radius = std::max(1, static_cast<int>(radius_fraction * width));
        return generateCircle(width, height, center_x, center_y, radius);
    };

    auto circles = makeCircle(0.25, 0.35, 0.10);
    circles = unionSurfaces(circles, makeCircle(0.60, 0.45, 0.12));
    circles = unionSurfaces(circles, makeCircle(0.40, 0.70, 0.08));

    std::cout << "Rectangles intervals: " << rectangles.intervalCount() << "\n";
    std::cout << "Circles intervals:    " << circles.intervalCount() << "\n";

    DeviceSurface d_rectangles = copy_surface_to_device(rectangles);
    DeviceSurface d_circles = copy_surface_to_device(circles);

    subsetix::SurfaceDescriptor rect_desc{};
    rect_desc.view.d_begin = d_rectangles.d_begin;
    rect_desc.view.d_end = d_rectangles.d_end;
    rect_desc.view.d_row_offsets = d_rectangles.d_offsets;
    rect_desc.view.row_count = height;
    rect_desc.interval_count = d_rectangles.interval_count;

    subsetix::SurfaceDescriptor circle_desc{};
    circle_desc.view.d_begin = d_circles.d_begin;
    circle_desc.view.d_end = d_circles.d_end;
    circle_desc.view.d_row_offsets = d_circles.d_offsets;
    circle_desc.view.row_count = height;
    circle_desc.interval_count = d_circles.interval_count;

    subsetix::SurfaceChainBuilder builder;
    auto rect_handle = builder.add_input(rect_desc);
    auto circle_handle = builder.add_input(circle_desc);
    auto union_handle = builder.add_union(rect_handle, circle_handle);
    builder.add_difference(union_handle, circle_handle);

    subsetix::SurfaceChainRunner runner(builder);
    CUDA_CHECK(runner.prepare());

    auto run_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(runner.run());
    auto run_end = std::chrono::high_resolution_clock::now();

    const auto& result = runner.result();
    std::cout << "Difference total:  " << result.total << "\n";
    std::cout << "Surface chain executor time (ms): "
              << std::chrono::duration<double, std::milli>(run_end - run_start).count() << "\n";

    if (result.total > 0) {
        std::vector<int> h_y(result.total);
        std::vector<int> h_begin(result.total);
        std::vector<int> h_end(result.total);
        CUDA_CHECK(cudaMemcpy(h_y.data(), result.d_y_idx, result.total * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_begin.data(), result.d_begin, result.total * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), result.d_end, result.total * sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "First difference interval: y=" << h_y[0]
                  << " [" << h_begin[0] << ", " << h_end[0] << ")\n";
    }

    free_device_surface(d_rectangles);
    free_device_surface(d_circles);
    return 0;
}
