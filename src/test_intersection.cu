#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"

struct SurfaceHost {
    int y_count = 0;
    std::vector<int> offsets;
    std::vector<int> begin;
    std::vector<int> end;

    int interval_count() const { return static_cast<int>(begin.size()); }
};

SurfaceHost buildSurface(int y_count,
                         const std::vector<std::vector<std::pair<int, int>>>& rows) {
    SurfaceHost surface;
    surface.y_count = y_count;
    if (static_cast<int>(rows.size()) != y_count) {
        throw std::runtime_error("Rows size does not match y_count");
    }

    surface.offsets.resize(y_count + 1, 0);
    int write_idx = 0;
    for (int y = 0; y < y_count; ++y) {
        surface.offsets[y] = write_idx;
        for (const auto& interval : rows[y]) {
            surface.begin.push_back(interval.first);
            surface.end.push_back(interval.second);
            ++write_idx;
        }
    }
    surface.offsets[y_count] = write_idx;
    return surface;
}

struct DeviceSurfaceForTest {
    int* begin = nullptr;
    int* end = nullptr;
    int* offsets = nullptr;
    int interval_count = 0;
};

DeviceSurfaceForTest copySurfaceToDevice(const SurfaceHost& surface) {
    DeviceSurfaceForTest device;
    device.interval_count = surface.interval_count();

    const size_t interval_bytes = static_cast<size_t>(surface.interval_count()) * sizeof(int);
    const size_t offsets_bytes = static_cast<size_t>(surface.y_count + 1) * sizeof(int);

    CUDA_CHECK(cudaMalloc(&device.begin, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.end, interval_bytes));
    CUDA_CHECK(cudaMalloc(&device.offsets, offsets_bytes));

    CUDA_CHECK(cudaMemcpy(device.begin, surface.begin.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.end, surface.end.data(), interval_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.offsets, surface.offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));

    return device;
}

void freeDeviceSurface(DeviceSurfaceForTest& surface) {
    if (surface.begin) CUDA_CHECK(cudaFree(surface.begin));
    if (surface.end) CUDA_CHECK(cudaFree(surface.end));
    if (surface.offsets) CUDA_CHECK(cudaFree(surface.offsets));
    surface = {};
}

struct SurfaceIntersectionResult {
    int y_idx;
    int r_begin;
    int r_end;
    int a_idx;
    int b_idx;

    bool operator==(const SurfaceIntersectionResult& other) const {
        return y_idx == other.y_idx && r_begin == other.r_begin && r_end == other.r_end &&
               a_idx == other.a_idx && b_idx == other.b_idx;
    }

    bool operator<(const SurfaceIntersectionResult& other) const {
        return std::tie(y_idx, a_idx, b_idx, r_begin, r_end) <
               std::tie(other.y_idx, other.a_idx, other.b_idx, other.r_begin, other.r_end);
    }
};

void expectSurfaceIntersections(const SurfaceHost& a,
                                const SurfaceHost& b,
                                const std::vector<SurfaceIntersectionResult>& expected) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_intersections = 0;

    cudaError_t err = findIntervalIntersections(
        d_a.begin, d_a.end, d_a.interval_count,
        d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.interval_count,
        d_b.offsets, b.y_count,
        &d_r_y_idx,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections);

    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total_intersections, static_cast<int>(expected.size()));

    std::vector<SurfaceIntersectionResult> actual;
    if (total_intersections > 0) {
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), d_a_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), d_b_idx, results_bytes, cudaMemcpyDeviceToHost));

        actual.resize(total_intersections);
        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    freeIntervalResults(d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

void expectSurfaceIntersectionsWorkspaceStream(const SurfaceHost& a,
                                               const SurfaceHost& b,
                                               const std::vector<SurfaceIntersectionResult>& expected) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int row_count = a.y_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total_intersections = 0;
    cudaError_t err = computeIntervalIntersectionOffsets(
        d_a.begin, d_a.end, d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.offsets, b.y_count,
        d_counts, d_offsets,
        &total_intersections,
        stream);

    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total_intersections, static_cast<int>(expected.size()));

    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    if (total_intersections > 0) {
        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_a_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_b_idx, results_bytes));

        err = writeIntervalIntersectionsWithOffsets(
            d_a.begin, d_a.end, d_a.offsets, a.y_count,
            d_b.begin, d_b.end, d_b.offsets, b.y_count,
            d_offsets,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            d_a_idx,
            d_b_idx,
            stream);

        ASSERT_EQ(err, cudaSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<SurfaceIntersectionResult> actual(total_intersections);
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), d_a_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), d_b_idx, results_bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaStreamDestroy(stream));

    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(SurfaceIntersectionTest, BasicOverlap) {
    SurfaceHost a = buildSurface(2, {
        {{0, 2}, {5, 7}},
        {{10, 12}}
    });

    SurfaceHost b = buildSurface(2, {
        {{1, 3}, {6, 9}},
        {{11, 13}}
    });

    std::vector<SurfaceIntersectionResult> expected = {
        {0, 1, 2, 0, 0},
        {0, 6, 7, 1, 1},
        {1, 11, 12, 2, 2}
    };

    expectSurfaceIntersections(a, b, expected);
}

TEST(SurfaceIntersectionTest, WorkspaceStreamOverlap) {
    SurfaceHost a = buildSurface(2, {
        {{0, 2}, {5, 7}},
        {{10, 12}}
    });

    SurfaceHost b = buildSurface(2, {
        {{1, 3}, {6, 9}},
        {{11, 13}}
    });

    std::vector<SurfaceIntersectionResult> expected = {
        {0, 1, 2, 0, 0},
        {0, 6, 7, 1, 1},
        {1, 11, 12, 2, 2}
    };

    expectSurfaceIntersectionsWorkspaceStream(a, b, expected);
}

TEST(SurfaceIntersectionTest, NoOverlap) {
    SurfaceHost a = buildSurface(2, {
        {{0, 2}, {5, 7}},
        {{10, 11}}
    });

    SurfaceHost b = buildSurface(2, {
        {{2, 4}, {8, 9}},
        {{11, 13}}
    });

    expectSurfaceIntersections(a, b, {});
}

TEST(SurfaceIntersectionTest, EmptyRow) {
    SurfaceHost a = buildSurface(3, {
        {{0, 4}},
        {},
        {{8, 12}}
    });

    SurfaceHost b = buildSurface(3, {
        {{1, 3}},
        {{6, 9}},
        {}
    });

    std::vector<SurfaceIntersectionResult> expected = {
        {0, 1, 3, 0, 0}
    };

    expectSurfaceIntersections(a, b, expected);
}

TEST(SurfaceIntersectionTest, OneDimensional) {
    SurfaceHost a = buildSurface(1, {
        {{0, 5}}
    });

    SurfaceHost b = buildSurface(1, {
        {{2, 4}}
    });

    std::vector<SurfaceIntersectionResult> expected = {
        {0, 2, 4, 0, 0}
    };

    expectSurfaceIntersections(a, b, expected);
}


TEST(SurfaceIntersectionTest, OneDimensionalComplex) {
    // y_count = 1, several adjacent and overlapping intervals
    SurfaceHost a = buildSurface(1, {{
        {0, 2}, {3, 7}, {8, 10}, {15, 20}
    }});

    SurfaceHost b = buildSurface(1, {{
        {2, 3}, {4, 5}, {6, 9}, {10, 15}, {18, 22}
    }});

    std::vector<SurfaceIntersectionResult> expected = {
        {0, 4, 5, 1, 1},   // [3,7] ∩ [4,5]
        {0, 6, 7, 1, 2},   // [3,7] ∩ [6,9]
        {0, 8, 9, 2, 2},   // [8,10] ∩ [6,9]
        {0, 18, 20, 3, 4}  // [15,20] ∩ [18,22]
    };

    expectSurfaceIntersections(a, b, expected);
}

TEST(SurfaceIntersectionTest, ComplexMultiRow) {
    // 3 rows, varying counts, multiple overlaps per row
    SurfaceHost a = buildSurface(3, {
        {{0, 5}, {6, 10}},
        {{2, 4}, {5, 6}, {8, 12}},
        {}
    });

    SurfaceHost b = buildSurface(3, {
        {{1, 3}, {3, 6}, {7, 8}, {10, 12}},
        {{1, 2}, {4, 9}, {11, 13}},
        {{0, 100}}
    });

    // Global indices (flattened):
    // A: r0 -> 0,1; r1 -> 2,3,4
    // B: r0 -> 0,1,2,3; r1 -> 4,5,6; r2 -> 7
    std::vector<SurfaceIntersectionResult> expected = {
        {0, 3, 5, 0, 1},  // r0: [0,5] ∩ [3,6]
        {0, 1, 3, 0, 0},  // r0: [0,5] ∩ [1,3]
        {0, 7, 8, 1, 2},  // r0: [6,10] ∩ [7,8]
        {1, 5, 6, 3, 5},  // r1: [5,6] ∩ [4,9]
        {1, 8, 9, 4, 5},  // r1: [8,12] ∩ [4,9]
        {1, 11, 12, 4, 6} // r1: [8,12] ∩ [11,13]
    };

    expectSurfaceIntersections(a, b, expected);
}


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

    freeVolumeIntersectionResults(d_r_z_idx, d_r_y_idx, d_r_begin, d_r_end, d_a_idx, d_b_idx);
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}

void expectIntersectionsWorkspaceStream(const VolumeHost& a,
                                        const VolumeHost& b,
                                        const std::vector<IntersectionResult>& expected) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int row_count = d_a.row_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total_intersections = 0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, d_a.row_count,
        d_b.begin, d_b.end, d_b.row_offsets, d_b.row_count,
        d_counts, d_offsets,
        &total_intersections,
        stream);

    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total_intersections, static_cast<int>(expected.size()));

    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    if (total_intersections > 0) {
        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_z_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_a_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_b_idx, results_bytes));

        err = writeVolumeIntersectionsWithOffsets(
            d_a.begin, d_a.end, d_a.row_offsets, d_a.row_count,
            d_b.begin, d_b.end, d_b.row_offsets, d_b.row_count,
            d_a.row_to_y, d_a.row_to_z,
            d_offsets,
            d_r_z_idx,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            d_a_idx,
            d_b_idx,
            stream);

        ASSERT_EQ(err, cudaSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<IntersectionResult> actual(total_intersections);
        std::vector<int> h_r_z_idx(total_intersections);
        std::vector<int> h_r_y_idx(total_intersections);
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        CUDA_CHECK(cudaMemcpy(h_r_z_idx.data(), d_r_z_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), d_r_y_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), d_r_end, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), d_a_idx, results_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), d_b_idx, results_bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_z_idx[i], h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    if (d_r_z_idx) CUDA_CHECK(cudaFree(d_r_z_idx));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaStreamDestroy(stream));

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

TEST(VolumeIntersectionSimpleTest, WorkspaceStreamThreeDimensionalMatch) {
    VolumeHost a = buildVolume(2, 1, {
        {{0, 5}},
        {{10, 18}}
    });

    VolumeHost b = buildVolume(2, 1, {
        {{2, 4}},
        {{14, 16}}
    });

    std::vector<IntersectionResult> expected = {
        {0, 0, 2, 4, 0, 0},
        {1, 0, 14, 16, 1, 1}
    };

    expectIntersectionsWorkspaceStream(a, b, expected);
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

TEST(VolumeIntersectionSimpleTest, ComplexMixedRows3D) {
    // z=2, y=3, mixed densities and overlaps
    // A rows (z,y):
    // (0,0): [0,5],[7,9]; (0,1): -; (0,2): [10,15]
    // (1,0): [2,4],[6,12]; (1,1): [0,3]; (1,2): -
    VolumeHost a = buildVolume(2, 3, {
        {{0, 5}, {7, 9}},   // r0 (z0,y0)
        {},                 // r1 (z0,y1)
        {{10, 15}},         // r2 (z0,y2)
        {{2, 4}, {6, 12}},  // r3 (z1,y0)
        {{0, 3}},           // r4 (z1,y1)
        {}                  // r5 (z1,y2)
    });

    // B rows:
    // (0,0): [1,2],[3,8]; (0,1): [0,100]; (0,2): [5,10],[12,20]
    // (1,0): [0,10],[11,13]; (1,1): [3,5]; (1,2): -
    VolumeHost b = buildVolume(2, 3, {
        {{1, 2}, {3, 8}},   // r0
        {{0, 100}},         // r1
        {{5, 10}, {12, 20}},// r2
        {{0, 10}, {11, 13}},// r3
        {{3, 5}},           // r4
        {}                  // r5
    });

    // A indices: r0 -> 0,1; r1 -> -; r2 -> 2; r3 -> 3,4; r4 -> 5; r5 -> -
    // B indices: r0 -> 0,1; r1 -> 2; r2 -> 3,4; r3 -> 5,6; r4 -> 7; r5 -> -
    std::vector<IntersectionResult> expected = {
        {0, 0, 1, 2, 0, 0},   // (0,0): [0,5] ∩ [1,2]
        {0, 0, 3, 5, 0, 1},   // (0,0): [0,5] ∩ [3,8]
        {0, 0, 7, 8, 1, 1},   // (0,0): [7,9] ∩ [3,8]
        {0, 2, 12, 15, 2, 4}, // (0,2): [10,15] ∩ [12,20]
        {1, 0, 2, 4, 3, 5},   // (1,0): [2,4] ∩ [0,10]
        {1, 0, 6, 10, 4, 5},  // (1,0): [6,12] ∩ [0,10]
        {1, 0, 11, 12, 4, 6}  // (1,0): [6,12] ∩ [11,13]
    };

    expectIntersections(a, b, expected);
}
