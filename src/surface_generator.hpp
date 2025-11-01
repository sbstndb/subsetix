#ifndef SURFACE_GENERATOR_HPP
#define SURFACE_GENERATOR_HPP

#include <string>
#include <utility>
#include <vector>

struct SurfaceIntervals {
    int width = 0;
    int height = 0;
    std::vector<int> x_begin;
    std::vector<int> x_end;
    std::vector<int> y_offsets;

    int intervalCount() const { return static_cast<int>(x_begin.size()); }
};

SurfaceIntervals generateRectangle(int width,
                                   int height,
                                   int x_min,
                                   int x_max,
                                   int y_min,
                                   int y_max);

SurfaceIntervals generateCircle(int width,
                                int height,
                                int center_x,
                                int center_y,
                                int radius);

SurfaceIntervals unionSurfaces(const SurfaceIntervals& lhs,
                               const SurfaceIntervals& rhs);

std::vector<int> rasterizeToMask(const SurfaceIntervals& surface);

void writeStructuredPoints(const std::string& filename,
                           int width,
                           int height,
                           const std::vector<int>& data);

#endif // SURFACE_GENERATOR_HPP
