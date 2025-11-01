#ifndef VOLUME_GENERATOR_HPP
#define VOLUME_GENERATOR_HPP

#include <string>
#include <utility>
#include <vector>

struct VolumeIntervals {
    int width = 0;
    int height = 0;
    int depth = 0;
    std::vector<int> x_begin;
    std::vector<int> x_end;
    std::vector<int> row_offsets; // size = row_count + 1
    std::vector<int> row_to_y;
    std::vector<int> row_to_z;

    int rowCount() const { return depth * height; }
    int intervalCount() const { return static_cast<int>(x_begin.size()); }
};

VolumeIntervals generateBox(int width,
                            int height,
                            int depth,
                            int x_min,
                            int x_max,
                            int y_min,
                            int y_max,
                            int z_min,
                            int z_max);

VolumeIntervals generateSphere(int width,
                               int height,
                               int depth,
                               float center_x,
                               float center_y,
                               float center_z,
                               float radius);

VolumeIntervals unionVolumes(const VolumeIntervals& lhs,
                             const VolumeIntervals& rhs);

std::vector<int> rasterizeToMask(const VolumeIntervals& volume);

void writeStructuredPoints(const std::string& filename,
                           int width,
                           int height,
                           int depth,
                           const std::vector<int>& data);

#endif // VOLUME_GENERATOR_HPP
