#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cub/device/device_scan.cuh>

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

struct DeviceGraphResults {
    int* z_idx = nullptr;
    int* y_idx = nullptr;
    int* begin = nullptr;
    int* end = nullptr;
    int* a_idx = nullptr;
    int* b_idx = nullptr;
};

DeviceGraphResults allocDeviceResults(int count, bool with_z) {
    DeviceGraphResults buffers;
    if (count <= 0) {
        return buffers;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(int);
    if (with_z) {
        CUDA_CHECK(cudaMalloc(&buffers.z_idx, bytes));
    }
    CUDA_CHECK(cudaMalloc(&buffers.y_idx, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.begin, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.end, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.a_idx, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.b_idx, bytes));
    return buffers;
}

void freeDeviceResults(DeviceGraphResults& buffers) {
    if (buffers.z_idx) CUDA_CHECK(cudaFree(buffers.z_idx));
    if (buffers.y_idx) CUDA_CHECK(cudaFree(buffers.y_idx));
    if (buffers.begin) CUDA_CHECK(cudaFree(buffers.begin));
    if (buffers.end) CUDA_CHECK(cudaFree(buffers.end));
    if (buffers.a_idx) CUDA_CHECK(cudaFree(buffers.a_idx));
    if (buffers.b_idx) CUDA_CHECK(cudaFree(buffers.b_idx));
    buffers = {};
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

void expectSurfaceIntersectionsGraph(const SurfaceHost& a,
                                     const SurfaceHost& b,
                                     const std::vector<SurfaceIntersectionResult>& expected) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int row_count = a.y_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (row_count > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 temp_storage_bytes,
                                                 d_counts,
                                                 d_offsets,
                                                 row_count));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    IntervalIntersectionOffsetsGraphConfig offsets_cfg{};
    offsets_cfg.d_a_begin = d_a.begin;
    offsets_cfg.d_a_end = d_a.end;
    offsets_cfg.d_a_y_offsets = d_a.offsets;
    offsets_cfg.a_y_count = a.y_count;
    offsets_cfg.d_b_begin = d_b.begin;
    offsets_cfg.d_b_end = d_b.end;
    offsets_cfg.d_b_y_offsets = d_b.offsets;
    offsets_cfg.b_y_count = b.y_count;
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_temp_storage;
    offsets_cfg.scan_temp_storage_bytes = temp_storage_bytes;
    offsets_cfg.d_total = d_total;

    IntervalIntersectionGraph offsets_graph{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&offsets_graph, offsets_cfg);
    ASSERT_EQ(err, cudaSuccess);

    err = launchIntervalIntersectionGraph(offsets_graph);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(offsets_graph.stream));

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(total, static_cast<int>(expected.size()));

    DeviceGraphResults buffers = allocDeviceResults(total, false);

    IntervalIntersectionWriteGraphConfig write_cfg{};
    write_cfg.d_a_begin = d_a.begin;
    write_cfg.d_a_end = d_a.end;
    write_cfg.d_a_y_offsets = d_a.offsets;
    write_cfg.a_y_count = a.y_count;
    write_cfg.d_b_begin = d_b.begin;
    write_cfg.d_b_end = d_b.end;
    write_cfg.d_b_y_offsets = d_b.offsets;
    write_cfg.b_y_count = b.y_count;
    write_cfg.d_offsets = d_offsets;
    write_cfg.d_r_y_idx = buffers.y_idx;
    write_cfg.d_r_begin = buffers.begin;
    write_cfg.d_r_end = buffers.end;
    write_cfg.d_a_idx = buffers.a_idx;
    write_cfg.d_b_idx = buffers.b_idx;
    write_cfg.total_capacity = total;

    IntervalIntersectionGraph write_graph{};
    err = createIntervalIntersectionWriteGraph(&write_graph, write_cfg);
    ASSERT_EQ(err, cudaSuccess);

    err = launchIntervalIntersectionGraph(write_graph);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(write_graph.stream));

    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        std::vector<int> h_r_y_idx(total);
        std::vector<int> h_r_begin(total);
        std::vector<int> h_r_end(total);
        std::vector<int> h_a_idx(total);
        std::vector<int> h_b_idx(total);

        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), buffers.y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), buffers.begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), buffers.end, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), buffers.a_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), buffers.b_idx, bytes, cudaMemcpyDeviceToHost));

        std::vector<SurfaceIntersectionResult> actual(total);
        for (int i = 0; i < total; ++i) {
            actual[i] = {h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    destroyIntervalIntersectionGraph(&write_graph);
    destroyIntervalIntersectionGraph(&offsets_graph);
    freeDeviceResults(buffers);
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
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

TEST(SurfaceIntersectionTest, GraphOverlap) {
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

    expectSurfaceIntersectionsGraph(a, b, expected);
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

// -----------------------------
// 3D tests (reuse existing helpers below in this file)
// -----------------------------

/* Moved below after 3D helpers */
/*TEST_PLACEHOLDER*/
#if 0
TEST(VolumeIntersectionTest, WorkspaceSimpleOverlapNoRowMaps_DEAD) {
    // z=1, y=2: row0 intersects once, row1 empty
    VolumeHost a = buildVolume(1, 2, {
        {{0, 5}},
        {}
    });
    VolumeHost b = buildVolume(1, 2, {
        {{2, 4}},
        {}
    });
    DeviceVolume d_a = copyToDevice(a, /*copy_row_maps=*/false);
    DeviceVolume d_b = copyToDevice(b, /*copy_row_maps=*/false);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int rows = a.row_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count,
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count,
        d_counts, d_offsets,
        &total,
        stream);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total, 1);

    int* d_r_z_idx = nullptr; // optional, can be nullptr
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    const size_t bytes = static_cast<size_t>(total) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_r_y_idx, bytes));
    CUDA_CHECK(cudaMalloc(&d_r_begin, bytes));
    CUDA_CHECK(cudaMalloc(&d_r_end, bytes));
    CUDA_CHECK(cudaMalloc(&d_a_idx, bytes));
    CUDA_CHECK(cudaMalloc(&d_b_idx, bytes));

    err = writeVolumeIntersectionsWithOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count,
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count,
        /*d_a_row_to_y*/ nullptr, /*d_a_row_to_z*/ nullptr,
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

    int h_y = -1, h_beg = -1, h_end = -1, h_ai = -1, h_bi = -1;
    CUDA_CHECK(cudaMemcpy(&h_y, d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_beg, d_r_begin, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_end, d_r_end, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ai, d_a_idx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bi, d_b_idx, bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_y, 0);
    EXPECT_EQ(h_beg, 2);
    EXPECT_EQ(h_end, 4);
    EXPECT_EQ(h_ai, 0);
    EXPECT_EQ(h_bi, 0);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    CUDA_CHECK(cudaStreamDestroy(stream));
    freeDeviceVolume(d_a);
freeDeviceVolume(d_b);
}
#endif // disabled duplicate early tests

TEST(IntervalGraphInvalidConfig, OffsetsMissingTempStorage_DEAD) {
    // 2D offsets graph with rows>0 must provide CUB temp storage.
    SurfaceHost a = buildSurface(1, {{{0, 1}}});
    SurfaceHost b = buildSurface(1, {{{0, 1}}});
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int rows = a.y_count;
    const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));

    IntervalIntersectionOffsetsGraphConfig cfg{};
    cfg.d_a_begin = d_a.begin;
    cfg.d_a_end = d_a.end;
    cfg.d_a_y_offsets = d_a.offsets;
    cfg.a_y_count = a.y_count;
    cfg.d_b_begin = d_b.begin;
    cfg.d_b_end = d_b.end;
    cfg.d_b_y_offsets = d_b.offsets;
    cfg.b_y_count = b.y_count;
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = nullptr; // invalid
    cfg.scan_temp_storage_bytes = 0;   // invalid

    IntervalIntersectionGraph g{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&g, cfg);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(IntervalGraphInvalidConfig, WriteMissingOutputs_DEAD) {
    SurfaceHost a = buildSurface(1, {{{0, 1}}});
    SurfaceHost b = buildSurface(1, {{{0, 1}}});
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int)));

    IntervalIntersectionWriteGraphConfig w{};
    w.d_a_begin = d_a.begin;
    w.d_a_end = d_a.end;
    w.d_a_y_offsets = d_a.offsets;
    w.a_y_count = a.y_count;
    w.d_b_begin = d_b.begin;
    w.d_b_end = d_b.end;
    w.d_b_y_offsets = d_b.offsets;
    w.b_y_count = b.y_count;
    w.d_offsets = d_offsets;
    w.total_capacity = 1; // but outputs are null -> invalid

    IntervalIntersectionGraph wg{};
    cudaError_t err = createIntervalIntersectionWriteGraph(&wg, w);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(IntervalGraphZeroRows, OffsetsWriteNoop_DEAD) {
    IntervalIntersectionOffsetsGraphConfig ocfg{};
    ocfg.a_y_count = 0;
    ocfg.b_y_count = 0;
    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    ocfg.d_total = d_total;
    IntervalIntersectionGraph og{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&og, ocfg);
    ASSERT_EQ(err, cudaSuccess);
    err = launchIntervalIntersectionGraph(og);
    ASSERT_EQ(err, cudaSuccess);
    int total = -1;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(total, 0);
    destroyIntervalIntersectionGraph(&og);
    if (d_total) CUDA_CHECK(cudaFree(d_total));

    IntervalIntersectionWriteGraphConfig wcfg{};
    wcfg.a_y_count = 0;
    wcfg.b_y_count = 0;
    wcfg.total_capacity = 0; // ok
    IntervalIntersectionGraph wg{};
    err = createIntervalIntersectionWriteGraph(&wg, wcfg);
    EXPECT_EQ(err, cudaSuccess);
    destroyIntervalIntersectionGraph(&wg);
}

TEST(VolumeGraphInvalidConfig, MismatchedRows_DEAD) {
    // a_row_count != b_row_count -> invalid
    VolumeHost a = buildVolume(1, 2, { {{0,5}}, {} }); // rows=2
    VolumeHost b = buildVolume(1, 1, { {{2,4}} });     // rows=1
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * 2));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * 2));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    // Provide a dummy temp size to bypass early nullptr checks if needed
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, 1));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    VolumeIntersectionOffsetsGraphConfig cfg{};
    cfg.d_a_begin = d_a.begin;
    cfg.d_a_end = d_a.end;
    cfg.d_a_row_offsets = d_a.row_offsets;
    cfg.a_row_count = 2;
    cfg.d_b_begin = d_b.begin;
    cfg.d_b_end = d_b.end;
    cfg.d_b_row_offsets = d_b.row_offsets;
    cfg.b_row_count = 1; // mismatch
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = d_temp;
    cfg.scan_temp_storage_bytes = temp_bytes;

    VolumeIntersectionGraph g{};
    cudaError_t err = createVolumeIntersectionOffsetsGraph(&g, cfg);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}

TEST(VolumeGraphSimple, OffsetsAndWriteNoRowMaps_DEAD) {
    VolumeHost a = buildVolume(1, 2, { {{0,5}}, {} });
    VolumeHost b = buildVolume(1, 2, { {{2,4}}, {} });
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);

    const int rows = a.row_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    if (rows > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, rows));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    }

    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    VolumeIntersectionOffsetsGraphConfig ocfg{};
    ocfg.d_a_begin = d_a.begin;
    ocfg.d_a_end = d_a.end;
    ocfg.d_a_row_offsets = d_a.row_offsets;
    ocfg.a_row_count = a.row_count;
    ocfg.d_b_begin = d_b.begin;
    ocfg.d_b_end = d_b.end;
    ocfg.d_b_row_offsets = d_b.row_offsets;
    ocfg.b_row_count = b.row_count;
    ocfg.d_counts = d_counts;
    ocfg.d_offsets = d_offsets;
    ocfg.d_scan_temp_storage = d_temp;
    ocfg.scan_temp_storage_bytes = temp_bytes;
    ocfg.d_total = d_total;

    VolumeIntersectionGraph offsets_graph{};
    cudaError_t err = createVolumeIntersectionOffsetsGraph(&offsets_graph, ocfg);
    ASSERT_EQ(err, cudaSuccess);
    err = launchVolumeIntersectionGraph(offsets_graph);
    ASSERT_EQ(err, cudaSuccess);

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(total, 1);

    // Prepare outputs and write graph
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_r_y_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_begin, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_end, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b_idx, sizeof(int)));

    VolumeIntersectionWriteGraphConfig wcfg{};
    wcfg.d_a_begin = d_a.begin;
    wcfg.d_a_end = d_a.end;
    wcfg.d_a_row_offsets = d_a.row_offsets;
    wcfg.a_row_count = a.row_count;
    wcfg.d_a_row_to_y = nullptr; // default to row index
    wcfg.d_a_row_to_z = nullptr; // default to 0
    wcfg.d_b_begin = d_b.begin;
    wcfg.d_b_end = d_b.end;
    wcfg.d_b_row_offsets = d_b.row_offsets;
    wcfg.b_row_count = b.row_count;
    wcfg.d_offsets = d_offsets;
    wcfg.d_r_z_idx = nullptr; // optional
    wcfg.d_r_y_idx = d_r_y_idx;
    wcfg.d_r_begin = d_r_begin;
    wcfg.d_r_end = d_r_end;
    wcfg.d_a_idx = d_a_idx;
    wcfg.d_b_idx = d_b_idx;
    wcfg.total_capacity = 1;

    VolumeIntersectionGraph write_graph{};
    err = createVolumeIntersectionWriteGraph(&write_graph, wcfg);
    ASSERT_EQ(err, cudaSuccess);
    err = launchVolumeIntersectionGraph(write_graph);
    ASSERT_EQ(err, cudaSuccess);

    int h_y = -1, h_beg = -1, h_end = -1, h_ai = -1, h_bi = -1;
    CUDA_CHECK(cudaMemcpy(&h_y, d_r_y_idx, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_beg, d_r_begin, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_end, d_r_end, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ai, d_a_idx, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bi, d_b_idx, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_y, 0);
    EXPECT_EQ(h_beg, 2);
    EXPECT_EQ(h_end, 4);
    EXPECT_EQ(h_ai, 0);
    EXPECT_EQ(h_bi, 0);

    destroyVolumeIntersectionGraph(&write_graph);
    destroyVolumeIntersectionGraph(&offsets_graph);
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
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

void expectIntersectionsGraph(const VolumeHost& a,
                              const VolumeHost& b,
                              const std::vector<IntersectionResult>& expected) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    const int row_count = d_a.row_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (row_count > 0) {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                 temp_storage_bytes,
                                                 d_counts,
                                                 d_offsets,
                                                 row_count));
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    VolumeIntersectionOffsetsGraphConfig offsets_cfg{};
    offsets_cfg.d_a_begin = d_a.begin;
    offsets_cfg.d_a_end = d_a.end;
    offsets_cfg.d_a_row_offsets = d_a.row_offsets;
    offsets_cfg.a_row_count = d_a.row_count;
    offsets_cfg.d_b_begin = d_b.begin;
    offsets_cfg.d_b_end = d_b.end;
    offsets_cfg.d_b_row_offsets = d_b.row_offsets;
    offsets_cfg.b_row_count = d_b.row_count;
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_temp_storage;
    offsets_cfg.scan_temp_storage_bytes = temp_storage_bytes;
    offsets_cfg.d_total = d_total;

    VolumeIntersectionGraph offsets_graph{};
    cudaError_t err = createVolumeIntersectionOffsetsGraph(&offsets_graph, offsets_cfg);
    ASSERT_EQ(err, cudaSuccess);

    err = launchVolumeIntersectionGraph(offsets_graph);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(offsets_graph.stream));

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(total, static_cast<int>(expected.size()));

    DeviceGraphResults buffers = allocDeviceResults(total, true);

    VolumeIntersectionWriteGraphConfig write_cfg{};
    write_cfg.d_a_begin = d_a.begin;
    write_cfg.d_a_end = d_a.end;
    write_cfg.d_a_row_offsets = d_a.row_offsets;
    write_cfg.a_row_count = d_a.row_count;
    write_cfg.d_a_row_to_y = d_a.row_to_y;
    write_cfg.d_a_row_to_z = d_a.row_to_z;
    write_cfg.d_b_begin = d_b.begin;
    write_cfg.d_b_end = d_b.end;
    write_cfg.d_b_row_offsets = d_b.row_offsets;
    write_cfg.b_row_count = d_b.row_count;
    write_cfg.d_offsets = d_offsets;
    write_cfg.d_r_z_idx = buffers.z_idx;
    write_cfg.d_r_y_idx = buffers.y_idx;
    write_cfg.d_r_begin = buffers.begin;
    write_cfg.d_r_end = buffers.end;
    write_cfg.d_a_idx = buffers.a_idx;
    write_cfg.d_b_idx = buffers.b_idx;
    write_cfg.total_capacity = total;

    VolumeIntersectionGraph write_graph{};
    err = createVolumeIntersectionWriteGraph(&write_graph, write_cfg);
    ASSERT_EQ(err, cudaSuccess);

    err = launchVolumeIntersectionGraph(write_graph);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(write_graph.stream));

    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        std::vector<int> h_r_z_idx(total);
        std::vector<int> h_r_y_idx(total);
        std::vector<int> h_r_begin(total);
        std::vector<int> h_r_end(total);
        std::vector<int> h_a_idx(total);
        std::vector<int> h_b_idx(total);

        CUDA_CHECK(cudaMemcpy(h_r_z_idx.data(), buffers.z_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_y_idx.data(), buffers.y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_begin.data(), buffers.begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r_end.data(), buffers.end, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a_idx.data(), buffers.a_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_idx.data(), buffers.b_idx, bytes, cudaMemcpyDeviceToHost));

        std::vector<IntersectionResult> actual(total);
        for (int i = 0; i < total; ++i) {
            actual[i] = {h_r_z_idx[i], h_r_y_idx[i], h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        EXPECT_EQ(actual, sorted_expected);
    }

    destroyVolumeIntersectionGraph(&write_graph);
    destroyVolumeIntersectionGraph(&offsets_graph);
    freeDeviceResults(buffers);
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_temp_storage) CUDA_CHECK(cudaFree(d_temp_storage));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
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

TEST(VolumeIntersectionSimpleTest, GraphThreeDimensionalMatch) {
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

    expectIntersectionsGraph(a, b, expected);
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

// Additional 2D/3D graph and workspace tests (invalid configs, zero-rows, simple overlaps)

TEST(IntervalGraphInvalidConfig, OffsetsMissingTempStorage) {
    SurfaceHost a = buildSurface(1, {{{0, 1}}});
    SurfaceHost b = buildSurface(1, {{{0, 1}}});
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int rows = a.y_count;
    const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));

    IntervalIntersectionOffsetsGraphConfig cfg{};
    cfg.d_a_begin = d_a.begin;
    cfg.d_a_end = d_a.end;
    cfg.d_a_y_offsets = d_a.offsets;
    cfg.a_y_count = a.y_count;
    cfg.d_b_begin = d_b.begin;
    cfg.d_b_end = d_b.end;
    cfg.d_b_y_offsets = d_b.offsets;
    cfg.b_y_count = b.y_count;
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = nullptr; // invalid in capture when rows>0
    cfg.scan_temp_storage_bytes = 0;

    IntervalIntersectionGraph g{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&g, cfg);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(IntervalGraphInvalidConfig, WriteMissingOutputs) {
    SurfaceHost a = buildSurface(1, {{{0, 1}}});
    SurfaceHost b = buildSurface(1, {{{0, 1}}});
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int)));

    IntervalIntersectionWriteGraphConfig w{};
    w.d_a_begin = d_a.begin;
    w.d_a_end = d_a.end;
    w.d_a_y_offsets = d_a.offsets;
    w.a_y_count = a.y_count;
    w.d_b_begin = d_b.begin;
    w.d_b_end = d_b.end;
    w.d_b_y_offsets = d_b.offsets;
    w.b_y_count = b.y_count;
    w.d_offsets = d_offsets;
    w.total_capacity = 1; // outputs null -> invalid

    IntervalIntersectionGraph wg{};
    cudaError_t err = createIntervalIntersectionWriteGraph(&wg, w);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(IntervalGraphZeroRows, OffsetsWriteNoop) {
    IntervalIntersectionOffsetsGraphConfig ocfg{};
    ocfg.a_y_count = 0;
    ocfg.b_y_count = 0;
    int* d_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    ocfg.d_total = d_total;
    IntervalIntersectionGraph og{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&og, ocfg);
    ASSERT_EQ(err, cudaSuccess);
    err = launchIntervalIntersectionGraph(og);
    ASSERT_EQ(err, cudaSuccess);
    int total = -1;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(total, 0);
    destroyIntervalIntersectionGraph(&og);
    if (d_total) CUDA_CHECK(cudaFree(d_total));

    IntervalIntersectionWriteGraphConfig wcfg{};
    wcfg.a_y_count = 0;
    wcfg.b_y_count = 0;
    wcfg.total_capacity = 0; // ok
    IntervalIntersectionGraph wg{};
    err = createIntervalIntersectionWriteGraph(&wg, wcfg);
    EXPECT_EQ(err, cudaSuccess);
    destroyIntervalIntersectionGraph(&wg);
}

TEST(VolumeGraphInvalidConfig, MismatchedRows) {
    VolumeHost a = buildVolume(1, 2, { {{0,5}}, {} }); // rows=2
    VolumeHost b = buildVolume(1, 1, { {{2,4}} });     // rows=1
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * 2));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * 2));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, 1));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    VolumeIntersectionOffsetsGraphConfig cfg{};
    cfg.d_a_begin = d_a.begin;
    cfg.d_a_end = d_a.end;
    cfg.d_a_row_offsets = d_a.row_offsets;
    cfg.a_row_count = 2;
    cfg.d_b_begin = d_b.begin;
    cfg.d_b_end = d_b.end;
    cfg.d_b_row_offsets = d_b.row_offsets;
    cfg.b_row_count = 1; // mismatch
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = d_temp;
    cfg.scan_temp_storage_bytes = temp_bytes;

    VolumeIntersectionGraph g{};
    cudaError_t err = createVolumeIntersectionOffsetsGraph(&g, cfg);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}

TEST(VolumeIntersectionTest, WorkspaceSimpleOverlapNoRowMaps) {
    VolumeHost a = buildVolume(1, 2, { {{0,5}}, {} });
    VolumeHost b = buildVolume(1, 2, { {{2,4}}, {} });
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int rows = a.row_count();
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, rows,
        d_b.begin, d_b.end, d_b.row_offsets, rows,
        d_counts, d_offsets,
        &total,
        stream);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(total, 1);

    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    const size_t bytes = static_cast<size_t>(total) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_r_y_idx, bytes));
    CUDA_CHECK(cudaMalloc(&d_r_begin, bytes));
    CUDA_CHECK(cudaMalloc(&d_r_end, bytes));
    CUDA_CHECK(cudaMalloc(&d_a_idx, bytes));
    CUDA_CHECK(cudaMalloc(&d_b_idx, bytes));

    err = writeVolumeIntersectionsWithOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, rows,
        d_b.begin, d_b.end, d_b.row_offsets, rows,
        /*d_a_row_to_y*/ nullptr, /*d_a_row_to_z*/ nullptr,
        d_offsets,
        /*d_r_z_idx*/ nullptr,
        d_r_y_idx,
        d_r_begin,
        d_r_end,
        d_a_idx,
        d_b_idx,
        stream);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_y = -1, h_beg = -1, h_end = -1, h_ai = -1, h_bi = -1;
    CUDA_CHECK(cudaMemcpy(&h_y, d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_beg, d_r_begin, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_end, d_r_end, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_ai, d_a_idx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bi, d_b_idx, bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_y, 0);
    EXPECT_EQ(h_beg, 2);
    EXPECT_EQ(h_end, 4);
    EXPECT_EQ(h_ai, 0);
    EXPECT_EQ(h_bi, 0);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_r_y_idx) CUDA_CHECK(cudaFree(d_r_y_idx));
    if (d_r_begin) CUDA_CHECK(cudaFree(d_r_begin));
    if (d_r_end) CUDA_CHECK(cudaFree(d_r_end));
    if (d_a_idx) CUDA_CHECK(cudaFree(d_a_idx));
    if (d_b_idx) CUDA_CHECK(cudaFree(d_b_idx));
    CUDA_CHECK(cudaStreamDestroy(stream));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}
