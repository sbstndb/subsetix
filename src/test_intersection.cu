#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

struct VolumeHost {
    int z_count = 0;
    int y_count = 0;
    std::vector<int> row_offsets; // size row_count + 1
    std::vector<int> row_to_y;
    std::vector<int> row_to_z;
    std::vector<int> begin;
    std::vector<int> end;

    int row_count() const { return z_count * y_count; }
    int interval_count() const { return static_cast<int>(begin.size()); }
};

VolumeHost buildVolume(int z_count,
                       int y_count,
                       const std::vector<std::vector<std::pair<int, int>>>& rows) {
    VolumeHost volume;
    volume.z_count = z_count;
    volume.y_count = y_count;
    const int row_count = z_count * y_count;
    if (static_cast<int>(rows.size()) != row_count) {
        throw std::runtime_error("Rows size does not match row count");
    }

    volume.row_offsets.resize(row_count + 1, 0);
    volume.row_to_y.resize(row_count, 0);
    volume.row_to_z.resize(row_count, 0);

    int write_idx = 0;
    for (int z = 0; z < z_count; ++z) {
        for (int y = 0; y < y_count; ++y) {
            const int row = z * y_count + y;
            volume.row_to_y[row] = y;
            volume.row_to_z[row] = z;
            volume.row_offsets[row] = write_idx;
            for (const auto& interval : rows[row]) {
                volume.begin.push_back(interval.first);
                volume.end.push_back(interval.second);
                ++write_idx;
            }
        }
    }
    volume.row_offsets[row_count] = write_idx;
    return volume;
}

struct DeviceVolume {
    int* begin = nullptr;
    int* end = nullptr;
    int* row_offsets = nullptr;
    int* row_to_y = nullptr;
    int* row_to_z = nullptr;
    int row_count = 0;
    int interval_count = 0;
};

DeviceVolume copyToDevice(const VolumeHost& volume, bool copy_row_maps) {
    DeviceVolume device;
    device.row_count = volume.row_count();
    device.interval_count = volume.interval_count();

    const size_t interval_bytes = static_cast<size_t>(device.interval_count) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(device.row_count + 1) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.row_offsets, offsets_bytes));

    CUDA_CHECK(cudaMemcpy(device.begin,
                          volume.begin.data(),
                          interval_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.end,
                          volume.end.data(),
                          interval_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.row_offsets,
                          volume.row_offsets.data(),
                          offsets_bytes,
                          cudaMemcpyHostToDevice));

    if (copy_row_maps) {
        const size_t row_map_bytes = static_cast<size_t>(device.row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&device.row_to_y, row_map_bytes));
        CUDA_CHECK(cudaMalloc(&device.row_to_z, row_map_bytes));
        CUDA_CHECK(cudaMemcpy(device.row_to_y,
                              volume.row_to_y.data(),
                              row_map_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.row_to_z,
                              volume.row_to_z.data(),
                              row_map_bytes,
                              cudaMemcpyHostToDevice));
    }

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

struct IntersectionResult {
    int z_idx;
    int y_idx;
    int r_begin;
    int r_end;
    int a_idx;
    int b_idx;

    bool operator==(const IntersectionResult& other) const {
        return z_idx == other.z_idx && y_idx == other.y_idx &&
               r_begin == other.r_begin && r_end == other.r_end &&
               a_idx == other.a_idx && b_idx == other.b_idx;
    }

    bool operator<(const IntersectionResult& other) const {
        return std::tie(z_idx, y_idx, a_idx, b_idx, r_begin, r_end) <
               std::tie(other.z_idx, other.y_idx, other.a_idx, other.b_idx, other.r_begin, other.r_end);
    }
};

void expectIntersections(const VolumeHost& a,
                         const VolumeHost& b,
                         const std::vector<IntersectionResult>& expected) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_intersections = 0;

    cudaError_t err = findVolumeIntersections(
        d_a.begin, d_a.end, d_a.interval_count,
        d_a.row_offsets, d_a.row_to_y, d_a.row_to_z, d_a.row_count,
        d_b.begin, d_b.end, d_b.interval_count,
        d_b.row_offsets, d_b.row_count,
        &d_r_z_idx, &d_r_y_idx,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections);

    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total_intersections, static_cast<int>(expected.size()));

    std::vector<IntersectionResult> actual;
    if (total_intersections > 0) {
        std::vector<int> h_r_z_idx(total_intersections);
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_r_z_idx.data(), d_r_z_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), d_a_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), d_b_idx, results_bytes, cudaMemcpyDeviceToHost));

        actual.resize(total_intersections);
        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_z_idx[i], h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        std::vector<IntersectionResult> sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    freeIntersectionResults(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}

TEST(VolumeIntersectionSimpleTest, BasicOverlap2D) {
    // Single plane (z=0), two rows in y
    VolumeHost a = buildVolume(1, 2, {
        {{0, 2}, {5, 7}},
        {{10, 12}}
    });

    VolumeHost b = buildVolume(1, 2, {
        {{1, 3}, {6, 9}},
        {{11, 13}}
    });

    std::vector<IntersectionResult> expected = {
        {0, 0, 1, 2, 0, 0},
        {0, 0, 6, 7, 1, 1},
        {0, 1, 11, 12, 2, 2}
    };

    expectIntersections(a, b, expected);
}

TEST(VolumeIntersectionSimpleTest, NoOverlap) {
    VolumeHost a = buildVolume(1, 2, {
        {{0, 2}, {5, 7}},
        {{10, 11}}
    });

    VolumeHost b = buildVolume(1, 2, {
        {{2, 4}, {7, 9}},
        {{11, 13}}
    });

    expectIntersections(a, b, {});
}

TEST(VolumeIntersectionSimpleTest, EmptyRows) {
    VolumeHost a = buildVolume(1, 3, {
        {{0, 4}},
        {},
        {{8, 12}}
    });

    VolumeHost b = buildVolume(1, 3, {
        {{1, 3}},
        {{6, 9}},
        {}
    });

    std::vector<IntersectionResult> expected = {
        {0, 0, 1, 3, 0, 0}
    };

    expectIntersections(a, b, expected);
}

TEST(VolumeIntersectionSimpleTest, ThreeDimensionalMatch) {
    // Two planes in z, each with one row.
    VolumeHost a = buildVolume(2, 1, {
        {{0, 5}},   // z=0
        {{10, 18}}  // z=1
    });

    VolumeHost b = buildVolume(2, 1, {
        {{2, 4}},   // z=0
        {{14, 16}}  // z=1
    });

    std::vector<IntersectionResult> expected = {
        {0, 0, 2, 4, 0, 0},
        {1, 0, 14, 16, 1, 1}
    };

    expectIntersections(a, b, expected);
}

TEST(VolumeIntersectionSimpleTest, MismatchedRowsReturnsError) {
    VolumeHost a = buildVolume(2, 1, {
        {{0, 5}},
        {{8, 12}}
    });

    VolumeHost b = buildVolume(1, 1, {
        {{2, 4}}
    });

    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_intersections = 0;

    cudaError_t err = findVolumeIntersections(
        d_a.begin, d_a.end, d_a.interval_count,
        d_a.row_offsets, d_a.row_to_y, d_a.row_to_z, d_a.row_count,
        d_b.begin, d_b.end, d_b.interval_count,
        d_b.row_offsets, d_b.row_count,
        &d_r_z_idx, &d_r_y_idx,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections);

    EXPECT_EQ(err, cudaErrorInvalidValue);
    EXPECT_EQ(total_intersections, 0);
    EXPECT_EQ(d_r_z_idx, nullptr);
    EXPECT_EQ(d_r_y_idx, nullptr);

    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}
