#include "surface_generator.hpp"
#include "interval_merge.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

int clampInt(int value, int min_value, int max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

} // namespace

SurfaceIntervals generateRectangle(int width,
                                   int height,
                                   int x_min,
                                   int x_max,
                                   int y_min,
                                   int y_max) {
    SurfaceIntervals surface;
    surface.width = width;
    surface.height = height;
    surface.y_offsets.resize(height + 1, 0);

    x_min = clampInt(x_min, 0, width);
    x_max = clampInt(x_max, 0, width);
    y_min = clampInt(y_min, 0, height);
    y_max = clampInt(y_max, 0, height);

    if (x_min >= x_max || y_min >= y_max) {
        return surface;
    }

    for (int y = 0; y < height; ++y) {
        surface.y_offsets[y] = static_cast<int>(surface.x_begin.size());
        if (y >= y_min && y < y_max) {
            surface.x_begin.push_back(x_min);
            surface.x_end.push_back(x_max);
        }
        surface.y_offsets[y + 1] = static_cast<int>(surface.x_begin.size());
    }

    return surface;
}

SurfaceIntervals generateCircle(int width,
                                int height,
                                int center_x,
                                int center_y,
                                int radius) {
    SurfaceIntervals surface;
    surface.width = width;
    surface.height = height;
    surface.y_offsets.resize(height + 1, 0);

    if (radius <= 0) {
        return surface;
    }

    for (int y = 0; y < height; ++y) {
        int dy = y - center_y;
        if (std::abs(dy) > radius) {
            surface.y_offsets[y + 1] = surface.y_offsets[y];
            continue;
        }

        double span = std::sqrt(static_cast<double>(radius * radius - dy * dy));
        int x_min = static_cast<int>(std::ceil(center_x - span));
        int x_max = static_cast<int>(std::floor(center_x + span)) + 1;

        x_min = clampInt(x_min, 0, width);
        x_max = clampInt(x_max, 0, width);

        surface.y_offsets[y] = static_cast<int>(surface.x_begin.size());
        if (x_min < x_max) {
            surface.x_begin.push_back(x_min);
            surface.x_end.push_back(x_max);
        }
        surface.y_offsets[y + 1] = static_cast<int>(surface.x_begin.size());
    }

    return surface;
}

SurfaceIntervals unionSurfaces(const SurfaceIntervals& lhs,
                               const SurfaceIntervals& rhs) {
    if (lhs.width != rhs.width || lhs.height != rhs.height) {
        throw std::runtime_error("Surface dimensions must match for union");
    }

    SurfaceIntervals result;
    result.width = lhs.width;
    result.height = lhs.height;
    result.y_offsets.resize(result.height + 1, 0);

    for (int y = 0; y < result.height; ++y) {
        int lhs_start = lhs.y_offsets[y];
        int lhs_end = lhs.y_offsets[y + 1];
        int rhs_start = rhs.y_offsets[y];
        int rhs_end = rhs.y_offsets[y + 1];

        auto merged = mergeIntervals(
            lhs_start < lhs_end ? &lhs.x_begin[lhs_start] : nullptr,
            lhs_start < lhs_end ? &lhs.x_end[lhs_start] : nullptr,
            lhs_end - lhs_start,
            rhs_start < rhs_end ? &rhs.x_begin[rhs_start] : nullptr,
            rhs_start < rhs_end ? &rhs.x_end[rhs_start] : nullptr,
            rhs_end - rhs_start);

        result.y_offsets[y] = static_cast<int>(result.x_begin.size());
        for (const auto& interval : merged) {
            result.x_begin.push_back(interval.first);
            result.x_end.push_back(interval.second);
        }
        result.y_offsets[y + 1] = static_cast<int>(result.x_begin.size());
    }

    return result;
}

std::vector<int> rasterizeToMask(const SurfaceIntervals& surface) {
    std::vector<int> mask(static_cast<size_t>(surface.width) * surface.height, 0);

    for (int y = 0; y < surface.height; ++y) {
        int start = surface.y_offsets[y];
        int end = surface.y_offsets[y + 1];

        for (int idx = start; idx < end; ++idx) {
            int x_begin = clampInt(surface.x_begin[idx], 0, surface.width);
            int x_end = clampInt(surface.x_end[idx], 0, surface.width);
            for (int x = x_begin; x < x_end; ++x) {
                mask[static_cast<size_t>(y) * surface.width + x] = 1;
            }
        }
    }

    return mask;
}

void writeStructuredPoints(const std::string& filename,
                           int width,
                           int height,
                           const std::vector<int>& data) {
    if (static_cast<int>(data.size()) != width * height) {
        throw std::runtime_error("Invalid data size for StructuredPoints writer");
    }

    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Unable to open output file: " + filename);
    }

    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "Surface representation\n";
    ofs << "ASCII\n";
    ofs << "DATASET STRUCTURED_POINTS\n";
    ofs << "DIMENSIONS " << width << " " << height << " 1\n";
    ofs << "ORIGIN 0 0 0\n";
    ofs << "SPACING 1 1 1\n";
    ofs << "POINT_DATA " << width * height << "\n";
    ofs << "SCALARS value int 1\n";
    ofs << "LOOKUP_TABLE default\n";

    for (size_t idx = 0; idx < data.size(); ++idx) {
        ofs << data[idx] << "\n";
    }
}
