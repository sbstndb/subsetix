#include "volume_generator.hpp"
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

VolumeIntervals generateBox(int width,
                            int height,
                            int depth,
                            int x_min,
                            int x_max,
                            int y_min,
                            int y_max,
                            int z_min,
                            int z_max) {
    VolumeIntervals volume;
    volume.width = width;
    volume.height = height;
    volume.depth = depth;
    volume.row_to_y.resize(volume.rowCount(), 0);
    volume.row_to_z.resize(volume.rowCount(), 0);
    volume.row_offsets.resize(volume.rowCount() + 1, 0);

    x_min = clampInt(x_min, 0, width);
    x_max = clampInt(x_max, 0, width);
    y_min = clampInt(y_min, 0, height);
    y_max = clampInt(y_max, 0, height);
    z_min = clampInt(z_min, 0, depth);
    z_max = clampInt(z_max, 0, depth);

    if (x_min >= x_max || y_min >= y_max || z_min >= z_max) {
        return volume;
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            const int row = z * height + y;
            volume.row_to_y[row] = y;
            volume.row_to_z[row] = z;
            if (z >= z_min && z < z_max && y >= y_min && y < y_max) {
                volume.row_offsets[row] = static_cast<int>(volume.x_begin.size());
                volume.x_begin.push_back(x_min);
                volume.x_end.push_back(x_max);
                volume.row_offsets[row + 1] = static_cast<int>(volume.x_begin.size());
            } else {
                volume.row_offsets[row + 1] = static_cast<int>(volume.x_begin.size());
            }
        }
    }

    return volume;
}

VolumeIntervals generateSphere(int width,
                               int height,
                               int depth,
                               float center_x,
                               float center_y,
                               float center_z,
                               float radius) {
    VolumeIntervals volume;
    volume.width = width;
    volume.height = height;
    volume.depth = depth;
    volume.row_to_y.resize(volume.rowCount(), 0);
    volume.row_to_z.resize(volume.rowCount(), 0);
    volume.row_offsets.resize(volume.rowCount() + 1, 0);

    if (radius <= 0.0f) {
        return volume;
    }

    int write_idx = 0;
    for (int z = 0; z < depth; ++z) {
        float dz = static_cast<float>(z) - center_z;
        float cross_radius_sq = radius * radius - dz * dz;
        for (int y = 0; y < height; ++y) {
            const int row = z * height + y;
            volume.row_to_y[row] = y;
            volume.row_to_z[row] = z;
            volume.row_offsets[row] = write_idx;

            if (cross_radius_sq <= 0.0f) {
                volume.row_offsets[row + 1] = write_idx;
                continue;
            }

            float dy = static_cast<float>(y) - center_y;
            float span_sq = cross_radius_sq - dy * dy;
            if (span_sq <= 0.0f) {
                volume.row_offsets[row + 1] = write_idx;
                continue;
            }

            float span = std::sqrt(span_sq);
            int x_min = static_cast<int>(std::floor(center_x - span));
            int x_max = static_cast<int>(std::ceil(center_x + span));

            x_min = clampInt(x_min, 0, width);
            x_max = clampInt(x_max, 0, width);

            if (x_min < x_max) {
                volume.x_begin.push_back(x_min);
                volume.x_end.push_back(x_max);
                ++write_idx;
            }

            volume.row_offsets[row + 1] = write_idx;
        }
    }

    return volume;
}

VolumeIntervals unionVolumes(const VolumeIntervals& lhs,
                             const VolumeIntervals& rhs) {
    if (lhs.width != rhs.width || lhs.height != rhs.height || lhs.depth != rhs.depth) {
        throw std::runtime_error("Volume dimensions must match for union");
    }

    VolumeIntervals result;
    result.width = lhs.width;
    result.height = lhs.height;
    result.depth = lhs.depth;
    result.row_to_y = lhs.row_to_y;
    result.row_to_z = lhs.row_to_z;
    result.row_offsets.resize(result.rowCount() + 1, 0);

    for (int row = 0; row < result.rowCount(); ++row) {
        int lhs_start = lhs.row_offsets[row];
        int lhs_end = lhs.row_offsets[row + 1];
        int rhs_start = rhs.row_offsets[row];
        int rhs_end = rhs.row_offsets[row + 1];

        auto merged = mergeIntervals(
            lhs_start < lhs_end ? &lhs.x_begin[lhs_start] : nullptr,
            lhs_start < lhs_end ? &lhs.x_end[lhs_start] : nullptr,
            lhs_end - lhs_start,
            rhs_start < rhs_end ? &rhs.x_begin[rhs_start] : nullptr,
            rhs_start < rhs_end ? &rhs.x_end[rhs_start] : nullptr,
            rhs_end - rhs_start);

        result.row_offsets[row] = static_cast<int>(result.x_begin.size());
        for (const auto& interval : merged) {
            result.x_begin.push_back(interval.first);
            result.x_end.push_back(interval.second);
        }
        result.row_offsets[row + 1] = static_cast<int>(result.x_begin.size());
    }

    return result;
}

std::vector<int> rasterizeToMask(const VolumeIntervals& volume) {
    std::vector<int> mask(static_cast<size_t>(volume.width) * volume.height * volume.depth, 0);

    for (int z = 0; z < volume.depth; ++z) {
        for (int y = 0; y < volume.height; ++y) {
            const int row = z * volume.height + y;
            int start = volume.row_offsets[row];
            int end = volume.row_offsets[row + 1];

            for (int idx = start; idx < end; ++idx) {
                int x_begin = clampInt(volume.x_begin[idx], 0, volume.width);
                int x_end = clampInt(volume.x_end[idx], 0, volume.width);
                for (int x = x_begin; x < x_end; ++x) {
                    size_t flat = static_cast<size_t>(z) * volume.height * volume.width +
                                  static_cast<size_t>(y) * volume.width + x;
                    mask[flat] = 1;
                }
            }
        }
    }

    return mask;
}

void writeStructuredPoints(const std::string& filename,
                           int width,
                           int height,
                           int depth,
                           const std::vector<int>& data) {
    if (static_cast<int>(data.size()) != width * height * depth) {
        throw std::runtime_error("Invalid data size for StructuredPoints writer");
    }

    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Unable to open output file: " + filename);
    }

    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "Volume representation\n";
    ofs << "ASCII\n";
    ofs << "DATASET STRUCTURED_POINTS\n";
    ofs << "DIMENSIONS " << width << " " << height << " " << depth << "\n";
    ofs << "ORIGIN 0 0 0\n";
    ofs << "SPACING 1 1 1\n";
    ofs << "POINT_DATA " << width * height * depth << "\n";
    ofs << "SCALARS value int 1\n";
    ofs << "LOOKUP_TABLE default\n";

    for (size_t idx = 0; idx < data.size(); ++idx) {
        ofs << data[idx] << "\n";
    }
}
