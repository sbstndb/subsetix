#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cub/device/device_scan.cuh>

#include "interval_intersection.cuh"
#include "operation_chain.cuh"
#include "surface_chain_executor.cuh"
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

class SurfaceOperationChainTest : public ::testing::Test {
protected:
    static SurfaceHost makeSurface(int rows, int base) {
        std::vector<std::vector<std::pair<int, int>>> intervals(rows);
        for (int y = 0; y < rows; ++y) {
            intervals[y].push_back({base + y, base + y + 1});
        }
        return buildSurface(rows, intervals);
    }
};

TEST_F(SurfaceOperationChainTest, LinearSequenceBuildsSuccessfully)
{
    constexpr int rows = 2;
    SurfaceHost host_a = makeSurface(rows, 0);
    SurfaceHost host_b = makeSurface(rows, 10);
    SurfaceHost host_c = makeSurface(rows, 20);

    DeviceSurfaceForTest dev_a = copySurfaceToDevice(host_a);
    DeviceSurfaceForTest dev_b = copySurfaceToDevice(host_b);
    DeviceSurfaceForTest dev_c = copySurfaceToDevice(host_c);

    subsetix::SurfaceOperationChain chain;
    auto a = chain.add_input({dev_a.begin, dev_a.end, dev_a.offsets, rows});
    auto b = chain.add_input({dev_b.begin, dev_b.end, dev_b.offsets, rows});
    auto diff = chain.add_operation(subsetix::SurfaceOpType::Difference, a, b);
    auto c = chain.add_input({dev_c.begin, dev_c.end, dev_c.offsets, rows});
    auto final_node = chain.add_operation(subsetix::SurfaceOpType::Intersection, diff, c);

    EXPECT_EQ(chain.row_count(), rows);
    ASSERT_EQ(chain.nodes().size(), 2u);
    EXPECT_TRUE(diff.is_node);
    EXPECT_TRUE(final_node.is_node);
    EXPECT_EQ(chain.nodes()[0].type, subsetix::SurfaceOpType::Difference);
    EXPECT_EQ(chain.nodes()[1].type, subsetix::SurfaceOpType::Intersection);

    freeDeviceSurface(dev_a);
    freeDeviceSurface(dev_b);
    freeDeviceSurface(dev_c);
}

TEST_F(SurfaceOperationChainTest, ExecutorPreparesLinearPlan)
{
    constexpr int rows = 3;
    SurfaceHost host_a = makeSurface(rows, 0);
    SurfaceHost host_b = makeSurface(rows, 5);
    SurfaceHost host_c = makeSurface(rows, 10);

    DeviceSurfaceForTest dev_a = copySurfaceToDevice(host_a);
    DeviceSurfaceForTest dev_b = copySurfaceToDevice(host_b);
    DeviceSurfaceForTest dev_c = copySurfaceToDevice(host_c);

    subsetix::SurfaceOperationChain chain;
    auto a = chain.add_input({dev_a.begin, dev_a.end, dev_a.offsets, rows});
    auto b = chain.add_input({dev_b.begin, dev_b.end, dev_b.offsets, rows});
    auto u = chain.add_operation(subsetix::SurfaceOpType::Union, a, b);
    auto c = chain.add_input({dev_c.begin, dev_c.end, dev_c.offsets, rows});
    chain.add_operation(subsetix::SurfaceOpType::Difference, u, c);

    subsetix::SurfaceChainExecutor exec;
    cudaError_t err = exec.prepare(chain);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(exec.row_count(), rows);

    const auto& plan = exec.plan();
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].type, subsetix::SurfaceOpType::Union);
    EXPECT_FALSE(plan[0].lhs.from_node);
    EXPECT_EQ(plan[0].lhs.index, a.index);
    EXPECT_FALSE(plan[0].rhs.from_node);
    EXPECT_EQ(plan[0].rhs.index, b.index);

    EXPECT_EQ(plan[1].type, subsetix::SurfaceOpType::Difference);
    EXPECT_TRUE(plan[1].lhs.from_node);
    EXPECT_EQ(plan[1].lhs.index, 0);
    EXPECT_FALSE(plan[1].rhs.from_node);
    EXPECT_EQ(plan[1].rhs.index, c.index);

    err = exec.plan_resources();
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(exec.counts_stride(), static_cast<size_t>(rows));
    EXPECT_EQ(exec.offsets_stride(), static_cast<size_t>(rows) + 1);
    EXPECT_EQ(exec.nodes_requiring_materialization(), 2u);
    const auto& flags = exec.materialization_flags();
    ASSERT_EQ(flags.size(), 2u);
    EXPECT_TRUE(flags[0]);
    EXPECT_TRUE(flags[1]);

    freeDeviceSurface(dev_a);
    freeDeviceSurface(dev_b);
    freeDeviceSurface(dev_c);
}

TEST_F(SurfaceOperationChainTest, RejectsMismatchedRowCounts)
{
    SurfaceHost host_a = makeSurface(2, 0);
    SurfaceHost host_b = makeSurface(3, 5);

    DeviceSurfaceForTest dev_a = copySurfaceToDevice(host_a);
    DeviceSurfaceForTest dev_b = copySurfaceToDevice(host_b);

    subsetix::SurfaceOperationChain chain;
    auto a = chain.add_input({dev_a.begin, dev_a.end, dev_a.offsets, 2});
    EXPECT_EQ(chain.row_count(), 2);

    EXPECT_THROW(chain.add_input({dev_b.begin, dev_b.end, dev_b.offsets, 3}), std::invalid_argument);

    subsetix::SurfaceHandle invalid_node{0, true};
    EXPECT_THROW(chain.add_operation(subsetix::SurfaceOpType::Union, a, invalid_node), std::invalid_argument);

    freeDeviceSurface(dev_a);
    freeDeviceSurface(dev_b);
}

TEST(SurfaceChainExecutorTest, EmptyChainRejected)
{
    subsetix::SurfaceOperationChain chain;
    subsetix::SurfaceChainExecutor exec;
    cudaError_t err = exec.prepare(chain);
    EXPECT_EQ(err, cudaErrorInvalidValue);
    EXPECT_TRUE(exec.plan().empty());
}

TEST(SurfaceChainExecutorTest, PlanResourcesAllowsZeroRows)
{
    int* d_dummy = nullptr;
    ASSERT_EQ(cudaMalloc(&d_dummy, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_dummy, 0, sizeof(int)), cudaSuccess);

    subsetix::SurfaceOperationChain chain;
    auto a = chain.add_input({d_dummy, d_dummy, d_dummy, 0});
    auto b = chain.add_input({d_dummy, d_dummy, d_dummy, 0});
    chain.add_operation(subsetix::SurfaceOpType::Union, a, b);

    subsetix::SurfaceChainExecutor exec;
    ASSERT_EQ(exec.prepare(chain), cudaSuccess);
    EXPECT_EQ(exec.row_count(), 0);
    ASSERT_EQ(exec.plan().size(), 1u);
    ASSERT_EQ(exec.plan_resources(), cudaSuccess);

    EXPECT_EQ(exec.counts_stride(), 1u);
    EXPECT_EQ(exec.offsets_stride(), 1u);
    EXPECT_EQ(exec.nodes_requiring_materialization(), 1u);

    CUDA_CHECK(cudaFree(d_dummy));
}

TEST(SurfaceChainExecutorTest, OffsetsGraphEvaluatesChain)
{
    constexpr int rows = 1;
    SurfaceHost host_a = buildSurface(rows, {{{0, 4}}});
    SurfaceHost host_b = buildSurface(rows, {{{4, 6}}});
    SurfaceHost host_c = buildSurface(rows, {{{1, 3}}});

    DeviceSurfaceForTest dev_a = copySurfaceToDevice(host_a);
    DeviceSurfaceForTest dev_b = copySurfaceToDevice(host_b);
    DeviceSurfaceForTest dev_c = copySurfaceToDevice(host_c);

    subsetix::SurfaceOperationChain chain;
    auto a = chain.add_input({dev_a.begin, dev_a.end, dev_a.offsets, rows});
    auto b = chain.add_input({dev_b.begin, dev_b.end, dev_b.offsets, rows});
    auto c = chain.add_input({dev_c.begin, dev_c.end, dev_c.offsets, rows});
    auto u = chain.add_operation(subsetix::SurfaceOpType::Union, a, b);
    chain.add_operation(subsetix::SurfaceOpType::Difference, u, c);

    subsetix::SurfaceChainExecutor exec;
    ASSERT_EQ(exec.prepare(chain), cudaSuccess);
    ASSERT_EQ(exec.plan_resources(), cudaSuccess);

    const size_t node_count = exec.node_count();
    ASSERT_EQ(node_count, 2u);

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    int* d_totals = nullptr;
    void* d_scan_temp = nullptr;
    size_t counts_elements = exec.counts_stride() * node_count;
    size_t offsets_elements = exec.offsets_stride() * node_count;

    CUDA_CHECK(cudaMalloc(&d_counts, counts_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_totals, node_count * sizeof(int)));

    size_t scan_temp_bytes = 0;
    if (exec.row_count() > 0) {
        int* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, exec.row_count() * sizeof(int)));
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, tmp, tmp, exec.row_count());
        CUDA_CHECK(cudaFree(tmp));
    }
    if (scan_temp_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
    }

    subsetix::SurfaceWorkspaceView workspace{};
    workspace.d_counts = d_counts;
    workspace.d_offsets = d_offsets;
    workspace.counts_stride = exec.counts_stride();
    workspace.offsets_stride = exec.offsets_stride();
    workspace.d_scan_temp = d_scan_temp;
    workspace.scan_temp_bytes = scan_temp_bytes;
    workspace.d_totals = d_totals;

    const size_t union_capacity = static_cast<size_t>(host_a.interval_count() + host_b.interval_count());
    int* d_union_begin = nullptr;
    int* d_union_end = nullptr;
    int* d_union_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_union_begin, union_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_union_end, union_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_union_y, union_capacity * sizeof(int)));

    std::vector<subsetix::SurfaceArenaSlice> slices(node_count);
    slices[0] = {d_union_begin, d_union_end, d_union_y, nullptr, nullptr, nullptr, union_capacity};
    slices[1] = {};

    subsetix::SurfaceArenaView arena{ slices.data(), slices.size() };

    subsetix::SurfaceChainGraph offsets_graph;
    ASSERT_EQ(exec.capture_offsets_graph(workspace, arena, &offsets_graph), cudaSuccess);
    ASSERT_EQ(subsetix::launch_surface_chain_graph(offsets_graph), cudaSuccess);
    ASSERT_EQ(CUDA_CHECK(cudaStreamSynchronize(offsets_graph.stream ? offsets_graph.stream : 0)), cudaSuccess);

    std::vector<int> totals(node_count, -1);
    CUDA_CHECK(cudaMemcpy(totals.data(), d_totals, node_count * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(totals[0], 1);
    EXPECT_EQ(totals[1], 2);

    const size_t diff_capacity = static_cast<size_t>(totals[1]);
    int* d_diff_begin = nullptr;
    int* d_diff_end = nullptr;
    int* d_diff_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diff_begin, diff_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diff_end, diff_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diff_y, diff_capacity * sizeof(int)));
    slices[1] = {d_diff_begin, d_diff_end, d_diff_y, nullptr, nullptr, nullptr, diff_capacity};

    subsetix::SurfaceArenaView write_arena{ slices.data(), slices.size() };

    subsetix::SurfaceChainGraph write_graph;
    ASSERT_EQ(exec.capture_write_graph(workspace, write_arena, &write_graph), cudaSuccess);
    ASSERT_EQ(subsetix::launch_surface_chain_graph(write_graph), cudaSuccess);
    ASSERT_EQ(CUDA_CHECK(cudaStreamSynchronize(write_graph.stream ? write_graph.stream : 0)), cudaSuccess);

    std::vector<int> diff_begin(diff_capacity);
    std::vector<int> diff_end(diff_capacity);
    CUDA_CHECK(cudaMemcpy(diff_begin.data(), d_diff_begin, diff_capacity * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(diff_end.data(), d_diff_end, diff_capacity * sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(diff_begin[0], 0);
    EXPECT_EQ(diff_end[0], 1);
    EXPECT_EQ(diff_begin[1], 3);
    EXPECT_EQ(diff_end[1], 6);

    subsetix::destroy_surface_chain_graph(&write_graph);
    subsetix::destroy_surface_chain_graph(&offsets_graph);

    CUDA_CHECK(cudaFree(d_union_begin));
    CUDA_CHECK(cudaFree(d_union_end));
    CUDA_CHECK(cudaFree(d_union_y));
    CUDA_CHECK(cudaFree(d_diff_begin));
    CUDA_CHECK(cudaFree(d_diff_end));
    CUDA_CHECK(cudaFree(d_diff_y));
    if (d_scan_temp) CUDA_CHECK(cudaFree(d_scan_temp));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_totals));

    freeDeviceSurface(dev_a);
    freeDeviceSurface(dev_b);
    freeDeviceSurface(dev_c);
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

std::vector<SurfaceIntersectionResult> runSurfaceIntersections(const SurfaceHost& a,
                                                               const SurfaceHost& b,
                                                               cudaStream_t stream = nullptr) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int row_count = a.y_count;

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* d_total = nullptr;
    DeviceGraphResults d_results{};

    IntervalIntersectionGraph offsets_graph{};
    IntervalIntersectionGraph write_graph{};
    IntervalIntersectionOffsetsGraphConfig cfg{};

    std::vector<SurfaceIntersectionResult> host_results;
    cudaError_t err = cudaSuccess;
    int total_intersections = 0;

    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
        if (err != cudaSuccess) {
            goto cleanup;
        }

        err = cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_temp_bytes,
            d_counts,
            d_offsets,
            row_count,
            stream ? stream : nullptr);
        if (err != cudaSuccess) {
            goto cleanup;
        }
        if (scan_temp_bytes > 0) {
            err = CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
            if (err != cudaSuccess) {
                goto cleanup;
            }
        }

        err = CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

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
    cfg.d_scan_temp_storage = d_scan_temp;
    cfg.scan_temp_storage_bytes = scan_temp_bytes;
    cfg.d_total = d_total;

    err = createIntervalIntersectionOffsetsGraph(&offsets_graph, cfg, stream);
    if (err != cudaSuccess) {
        ADD_FAILURE() << "createIntervalIntersectionOffsetsGraph failed: " << cudaGetErrorString(err);
        goto cleanup;
    }

    err = launchIntervalIntersectionGraph(offsets_graph, stream);
    if (err != cudaSuccess) {
        ADD_FAILURE() << "launchIntervalIntersectionGraph (offsets) failed: " << cudaGetErrorString(err);
        goto cleanup;
    }

    if (row_count > 0) {
        cudaStream_t offsets_stream = stream ? stream : offsets_graph.stream;
        err = cudaStreamSynchronize(offsets_stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize (offsets) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        err = CUDA_CHECK(cudaMemcpy(&total_intersections, d_total, sizeof(int), cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    if (total_intersections > 0) {
        d_results = allocDeviceResults(total_intersections, false);

        IntervalIntersectionWriteGraphConfig cfg{};
        cfg.d_a_begin = d_a.begin;
        cfg.d_a_end = d_a.end;
        cfg.d_a_y_offsets = d_a.offsets;
        cfg.a_y_count = a.y_count;
        cfg.d_b_begin = d_b.begin;
        cfg.d_b_end = d_b.end;
        cfg.d_b_y_offsets = d_b.offsets;
        cfg.b_y_count = b.y_count;
        cfg.d_offsets = d_offsets;
        cfg.d_r_y_idx = d_results.y_idx;
        cfg.d_r_begin = d_results.begin;
        cfg.d_r_end = d_results.end;
        cfg.d_a_idx = d_results.a_idx;
        cfg.d_b_idx = d_results.b_idx;
        cfg.total_capacity = total_intersections;

        err = createIntervalIntersectionWriteGraph(&write_graph, cfg, stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "createIntervalIntersectionWriteGraph failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        err = launchIntervalIntersectionGraph(write_graph, stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "launchIntervalIntersectionGraph (write) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        cudaStream_t write_stream = stream ? stream : write_graph.stream;
        err = cudaStreamSynchronize(write_stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize (write) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        std::vector<int> h_y(total_intersections);
        std::vector<int> h_begin(total_intersections);
        std::vector<int> h_end(total_intersections);
        std::vector<int> h_a(total_intersections);
        std::vector<int> h_b(total_intersections);

        err = CUDA_CHECK(cudaMemcpy(h_y.data(), d_results.y_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_begin.data(), d_results.begin, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_end.data(), d_results.end, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_a.data(), d_results.a_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_b.data(), d_results.b_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }

        host_results.reserve(static_cast<size_t>(total_intersections));
        for (int i = 0; i < total_intersections; ++i) {
            host_results.push_back({h_y[i], h_begin[i], h_end[i], h_a[i], h_b[i]});
        }
    }

cleanup:
    destroyIntervalIntersectionGraph(&write_graph);
    destroyIntervalIntersectionGraph(&offsets_graph);

    freeDeviceResults(d_results);
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_scan_temp) CUDA_CHECK(cudaFree(d_scan_temp));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);

    return host_results;
}

void expectSurfaceIntersectionsWorkspace(const SurfaceHost& a,
                                         const SurfaceHost& b,
                                         const std::vector<SurfaceIntersectionResult>& expected,
                                         bool use_stream) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    auto actual = runSurfaceIntersections(a, b, stream);

    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
}

struct SurfaceSpan {
    int y_idx;
    int r_begin;
    int r_end;

    bool operator==(const SurfaceSpan& other) const {
        return y_idx == other.y_idx && r_begin == other.r_begin && r_end == other.r_end;
    }

    bool operator<(const SurfaceSpan& other) const {
        return std::tie(y_idx, r_begin, r_end) <
               std::tie(other.y_idx, other.r_begin, other.r_end);
    }
};

std::vector<SurfaceSpan> runSurfaceUnion(const SurfaceHost& a,
                                         const SurfaceHost& b,
                                         cudaStream_t stream = nullptr) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    const int row_count = a.y_count;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeIntervalUnionOffsets(
        d_a.begin, d_a.end, d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.offsets, b.y_count,
        d_counts, d_offsets,
        &total,
        stream);

    if (err != cudaSuccess) {
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        freeDeviceSurface(d_a);
        freeDeviceSurface(d_b);
        ADD_FAILURE() << "computeIntervalUnionOffsets failed: " << cudaGetErrorString(err);
        return {};
    }

    std::vector<SurfaceSpan> host_results;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;

    if (total > 0) {
        const size_t results_bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));

        err = writeIntervalUnionWithOffsets(
            d_a.begin, d_a.end, d_a.offsets, a.y_count,
            d_b.begin, d_b.end, d_b.offsets, b.y_count,
            d_offsets,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            stream);

        if (err != cudaSuccess) {
            CUDA_CHECK(cudaFree(d_r_y_idx));
            CUDA_CHECK(cudaFree(d_r_begin));
            CUDA_CHECK(cudaFree(d_r_end));
            if (d_counts) CUDA_CHECK(cudaFree(d_counts));
            if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
            freeDeviceSurface(d_a);
            freeDeviceSurface(d_b);
            ADD_FAILURE() << "writeIntervalUnionWithOffsets failed: " << cudaGetErrorString(err);
            return {};
        }

        host_results.resize(total);
        std::vector<int> h_y(total);
        std::vector<int> h_begin(total);
        std::vector<int> h_end(total);
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_begin.data(), d_r_begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), d_r_end, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < total; ++i) {
            host_results[i] = {h_y[i], h_begin[i], h_end[i]};
        }

        CUDA_CHECK(cudaFree(d_r_y_idx));
        CUDA_CHECK(cudaFree(d_r_begin));
        CUDA_CHECK(cudaFree(d_r_end));
    }

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
    return host_results;
}

std::vector<SurfaceSpan> runSurfaceDifference(const SurfaceHost& a,
                                              const SurfaceHost& b,
                                              cudaStream_t stream = nullptr) {
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    const int row_count = a.y_count;
    if (row_count > 0) {
        const size_t row_bytes = static_cast<size_t>(row_count) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeIntervalDifferenceOffsets(
        d_a.begin, d_a.end, d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.offsets, b.y_count,
        d_counts, d_offsets,
        &total,
        stream);

    if (err != cudaSuccess) {
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        freeDeviceSurface(d_a);
        freeDeviceSurface(d_b);
        ADD_FAILURE() << "computeIntervalDifferenceOffsets failed: " << cudaGetErrorString(err);
        return {};
    }

    std::vector<SurfaceSpan> host_results;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;

    if (total > 0) {
        const size_t results_bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));

        err = writeIntervalDifferenceWithOffsets(
            d_a.begin, d_a.end, d_a.offsets, a.y_count,
            d_b.begin, d_b.end, d_b.offsets, b.y_count,
            d_offsets,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            stream);

        if (err != cudaSuccess) {
            CUDA_CHECK(cudaFree(d_r_y_idx));
            CUDA_CHECK(cudaFree(d_r_begin));
            CUDA_CHECK(cudaFree(d_r_end));
            if (d_counts) CUDA_CHECK(cudaFree(d_counts));
            if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
            freeDeviceSurface(d_a);
            freeDeviceSurface(d_b);
            ADD_FAILURE() << "writeIntervalDifferenceWithOffsets failed: " << cudaGetErrorString(err);
            return {};
        }

        host_results.resize(total);
        std::vector<int> h_y(total);
        std::vector<int> h_begin(total);
        std::vector<int> h_end(total);
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_begin.data(), d_r_begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), d_r_end, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < total; ++i) {
            host_results[i] = {h_y[i], h_begin[i], h_end[i]};
        }

        CUDA_CHECK(cudaFree(d_r_y_idx));
        CUDA_CHECK(cudaFree(d_r_begin));
        CUDA_CHECK(cudaFree(d_r_end));
    }

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
    return host_results;
}

void expectSurfaceUnionWorkspace(const SurfaceHost& a,
                                 const SurfaceHost& b,
                                 const std::vector<SurfaceSpan>& expected,
                                 bool use_stream = false) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    auto actual = runSurfaceUnion(a, b, stream);
    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
}

void expectSurfaceDifferenceWorkspace(const SurfaceHost& a,
                                      const SurfaceHost& b,
                                      const std::vector<SurfaceSpan>& expected,
                                      bool use_stream = false) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    auto actual = runSurfaceDifference(a, b, stream);
    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/true);
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

    expectSurfaceIntersectionsWorkspace(a, b, {}, /*use_stream=*/false);
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
}

TEST(SurfaceSetOpsTest, UnionMergesIntervals) {
    SurfaceHost a = buildSurface(2, {
        {{0, 4}, {6, 8}},
        {}
    });

    SurfaceHost b = buildSurface(2, {
        {{2, 5}, {8, 10}},
        {{1, 3}}
    });

    std::vector<SurfaceSpan> expected = {
        {0, 0, 5},
        {0, 6, 10},
        {1, 1, 3}
    };

    expectSurfaceUnionWorkspace(a, b, expected);
}

TEST(SurfaceSetOpsTest, DifferenceSlicesIntervals) {
    SurfaceHost a = buildSurface(2, {
        {{0, 5}, {7, 10}},
        {{1, 4}}
    });

    SurfaceHost b = buildSurface(2, {
        {{1, 2}, {3, 8}},
        {}
    });

    std::vector<SurfaceSpan> expected = {
        {0, 0, 1},
        {0, 2, 3},
        {0, 8, 10},
        {1, 1, 4}
    };

    expectSurfaceDifferenceWorkspace(a, b, expected);
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

#if 0
#if 0
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
#endif // disable duplicate early 3D/2D tests

#endif
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

    expectSurfaceIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
}

// -----------------------------
// Additional negative/edge tests (2D/3D)
// -----------------------------

TEST(SurfaceWorkspaceInvalidInput, MismatchedRows) {
    SurfaceHost a = buildSurface(2, { {{0,1}}, {{2,3}} });
    SurfaceHost b = buildSurface(1, { {{0,1}} });
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int rows = a.y_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    cudaError_t err = computeIntervalIntersectionOffsets(
        d_a.begin, d_a.end, d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.offsets, b.y_count,
        d_counts, d_offsets,
        nullptr,
        nullptr);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(SurfaceGraphInvalidConfig, MismatchedRows) {
    SurfaceHost a = buildSurface(2, { {{0,1}}, {{2,3}} });
    SurfaceHost b = buildSurface(1, { {{0,1}} });
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int rows = a.y_count; // 2
    const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, rows));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    IntervalIntersectionOffsetsGraphConfig cfg{};
    cfg.d_a_begin = d_a.begin;
    cfg.d_a_end = d_a.end;
    cfg.d_a_y_offsets = d_a.offsets;
    cfg.a_y_count = a.y_count;   // 2
    cfg.d_b_begin = d_b.begin;
    cfg.d_b_end = d_b.end;
    cfg.d_b_y_offsets = d_b.offsets;
    cfg.b_y_count = b.y_count;   // 1 -> mismatch
    cfg.d_counts = d_counts;
    cfg.d_offsets = d_offsets;
    cfg.d_scan_temp_storage = d_temp;
    cfg.scan_temp_storage_bytes = temp_bytes;

    IntervalIntersectionGraph g{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&g, cfg);
    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

TEST(SurfaceWorkspaceInvalid, NullCountsOffsets) {
    SurfaceHost a = buildSurface(1, { {{0,1}} });
    SurfaceHost b = buildSurface(1, { {{0,1}} });
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    int total = 0;
    cudaError_t err = computeIntervalIntersectionOffsets(
        d_a.begin, d_a.end, d_a.offsets, a.y_count,
        d_b.begin, d_b.end, d_b.offsets, b.y_count,
        /*d_counts*/ nullptr, /*d_offsets*/ nullptr,
        &total,
        nullptr);
    EXPECT_EQ(err, cudaErrorInvalidValue);
    freeDeviceSurface(d_a);
    freeDeviceSurface(d_b);
}

#if 0
TEST(VolumeWorkspaceInvalid, NullCountsOffsets) {
    VolumeHost a = buildVolume(1, 1, { {{0,1}} });
    VolumeHost b = buildVolume(1, 1, { {{0,1}} });
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);
    int total = 0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
        /*d_counts*/ nullptr, /*d_offsets*/ nullptr,
        &total,
        nullptr);
    EXPECT_EQ(err, cudaErrorInvalidValue);
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
}

TEST(SurfaceGraphRowsGtZeroButTotalZero, OffsetsThenWriteZeroCapacity) {
    // Disjoint intervals per row, rows>0 => total==0
    SurfaceHost a = buildSurface(2, { {{0,1}}, {{2,3}} });
    SurfaceHost b = buildSurface(2, { {{1,2}}, {{3,4}} });
    DeviceSurfaceForTest d_a = copySurfaceToDevice(a);
    DeviceSurfaceForTest d_b = copySurfaceToDevice(b);

    const int rows = a.y_count;
    const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));

    void* d_temp = nullptr; size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, rows));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    int* d_total = nullptr; CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    IntervalIntersectionOffsetsGraphConfig ocfg{};
    ocfg.d_a_begin = d_a.begin; ocfg.d_a_end = d_a.end; ocfg.d_a_y_offsets = d_a.offsets; ocfg.a_y_count = a.y_count;
    ocfg.d_b_begin = d_b.begin; ocfg.d_b_end = d_b.end; ocfg.d_b_y_offsets = d_b.offsets; ocfg.b_y_count = b.y_count;
    ocfg.d_counts = d_counts; ocfg.d_offsets = d_offsets;
    ocfg.d_scan_temp_storage = d_temp; ocfg.scan_temp_storage_bytes = temp_bytes; ocfg.d_total = d_total;

    IntervalIntersectionGraph og{};
    cudaError_t err = createIntervalIntersectionOffsetsGraph(&og, ocfg);
    ASSERT_EQ(err, cudaSuccess);
    err = launchIntervalIntersectionGraph(og); ASSERT_EQ(err, cudaSuccess);
    int total = 42; CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(total, 0);

    IntervalIntersectionWriteGraphConfig wcfg{};
    wcfg.d_a_begin = d_a.begin; wcfg.d_a_end = d_a.end; wcfg.d_a_y_offsets = d_a.offsets; wcfg.a_y_count = a.y_count;
    wcfg.d_b_begin = d_b.begin; wcfg.d_b_end = d_b.end; wcfg.d_b_y_offsets = d_b.offsets; wcfg.b_y_count = b.y_count;
    wcfg.d_offsets = d_offsets;
    wcfg.total_capacity = 0; // zero-capacity write graph
    IntervalIntersectionGraph wg{};
    err = createIntervalIntersectionWriteGraph(&wg, wcfg);
    EXPECT_EQ(err, cudaSuccess);
    err = launchIntervalIntersectionGraph(wg); // empty graph
    EXPECT_EQ(err, cudaSuccess);

    destroyIntervalIntersectionGraph(&wg);
    destroyIntervalIntersectionGraph(&og);
    if (d_counts) CUDA_CHECK(cudaFree(d_counts)); if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_temp) CUDA_CHECK(cudaFree(d_temp)); if (d_total) CUDA_CHECK(cudaFree(d_total));
    freeDeviceSurface(d_a); freeDeviceSurface(d_b);
}

TEST(VolumeGraphRowsGtZeroButTotalZero, OffsetsThenWriteZeroCapacity) {
    VolumeHost a = buildVolume(1, 2, { {{0,1}}, {{2,3}} });
    VolumeHost b = buildVolume(1, 2, { {{1,2}}, {{3,4}} });
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);

    const int rows = a.row_count();
    int* d_counts = nullptr; int* d_offsets = nullptr;
    if (rows>0) { const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes)); CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes)); }
    void* d_temp = nullptr; size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_offsets, rows));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    int* d_total=nullptr; CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));

    VolumeIntersectionOffsetsGraphConfig ocfg{};
    ocfg.d_a_begin=d_a.begin; ocfg.d_a_end=d_a.end; ocfg.d_a_row_offsets=d_a.row_offsets; ocfg.a_row_count=rows;
    ocfg.d_b_begin=d_b.begin; ocfg.d_b_end=d_b.end; ocfg.d_b_row_offsets=d_b.row_offsets; ocfg.b_row_count=rows;
    ocfg.d_counts=d_counts; ocfg.d_offsets=d_offsets; ocfg.d_scan_temp_storage=d_temp; ocfg.scan_temp_storage_bytes=temp_bytes; ocfg.d_total=d_total;

    VolumeIntersectionGraph og{}; cudaError_t err = createVolumeIntersectionOffsetsGraph(&og, ocfg);
    ASSERT_EQ(err, cudaSuccess); err = launchVolumeIntersectionGraph(og); ASSERT_EQ(err, cudaSuccess);
    int total=123; CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost)); EXPECT_EQ(total, 0);

    VolumeIntersectionWriteGraphConfig wcfg{};
    wcfg.d_a_begin=d_a.begin; wcfg.d_a_end=d_a.end; wcfg.d_a_row_offsets=d_a.row_offsets; wcfg.a_row_count=rows;
    wcfg.d_b_begin=d_b.begin; wcfg.d_b_end=d_b.end; wcfg.d_b_row_offsets=d_b.row_offsets; wcfg.b_row_count=rows;
    wcfg.d_offsets=d_offsets; wcfg.total_capacity=0;
    VolumeIntersectionGraph wg{}; err = createVolumeIntersectionWriteGraph(&wg, wcfg);
    EXPECT_EQ(err, cudaSuccess); err = launchVolumeIntersectionGraph(wg); EXPECT_EQ(err, cudaSuccess);

    destroyVolumeIntersectionGraph(&wg); destroyVolumeIntersectionGraph(&og);
    if (d_counts) CUDA_CHECK(cudaFree(d_counts)); if (d_offsets) CUDA_CHECK(cudaFree(d_offsets)); if (d_temp) CUDA_CHECK(cudaFree(d_temp)); if (d_total) CUDA_CHECK(cudaFree(d_total));
    freeDeviceVolume(d_a); freeDeviceVolume(d_b);
}

TEST(VolumeWorkspaceInvalid, MismatchedRows) {
    VolumeHost a = buildVolume(1, 2, { {{0,1}}, {} });
    VolumeHost b = buildVolume(1, 1, { {{0,1}} });
    DeviceVolume d_a = copyToDevice(a, false);
    DeviceVolume d_b = copyToDevice(b, false);
    int total=0;
    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
        /*d_counts*/ nullptr, /*d_offsets*/ nullptr,
        &total,
        nullptr);
    EXPECT_EQ(err, cudaErrorInvalidValue);
    freeDeviceVolume(d_a); freeDeviceVolume(d_b);
}
#endif // disabled early volume tests; duplicates exist after 3D helpers


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

std::vector<IntersectionResult> runVolumeIntersections(const VolumeHost& a,
                                                       const VolumeHost& b,
                                                       cudaStream_t stream = nullptr) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    const int rows = a.row_count();

    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* d_total = nullptr;
    DeviceGraphResults d_results{};

    VolumeIntersectionGraph offsets_graph{};
    VolumeIntersectionGraph write_graph{};
    VolumeIntersectionOffsetsGraphConfig offsets_cfg{};
    VolumeIntersectionWriteGraphConfig write_cfg{};

    std::vector<IntersectionResult> host_results;
    cudaError_t err = cudaSuccess;
    int total_intersections = 0;

    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        err = CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
        if (err != cudaSuccess) {
            goto cleanup;
        }

        err = cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_temp_bytes,
            d_counts,
            d_offsets,
            rows,
            stream ? stream : nullptr);
        if (err != cudaSuccess) {
            goto cleanup;
        }
        if (scan_temp_bytes > 0) {
            err = CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
            if (err != cudaSuccess) {
                goto cleanup;
            }
        }

        err = CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    offsets_cfg.d_a_begin = d_a.begin;
    offsets_cfg.d_a_end = d_a.end;
    offsets_cfg.d_a_row_offsets = d_a.row_offsets;
    offsets_cfg.a_row_count = a.row_count();
    offsets_cfg.d_b_begin = d_b.begin;
    offsets_cfg.d_b_end = d_b.end;
    offsets_cfg.d_b_row_offsets = d_b.row_offsets;
    offsets_cfg.b_row_count = b.row_count();
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_scan_temp;
    offsets_cfg.scan_temp_storage_bytes = scan_temp_bytes;
    offsets_cfg.d_total = d_total;

    err = createVolumeIntersectionOffsetsGraph(&offsets_graph, offsets_cfg, stream);
    if (err != cudaSuccess) {
        ADD_FAILURE() << "createVolumeIntersectionOffsetsGraph failed: " << cudaGetErrorString(err);
        goto cleanup;
    }

    err = launchVolumeIntersectionGraph(offsets_graph, stream);
    if (err != cudaSuccess) {
        ADD_FAILURE() << "launchVolumeIntersectionGraph (offsets) failed: " << cudaGetErrorString(err);
        goto cleanup;
    }
    if (rows > 0) {
        cudaStream_t offsets_stream = stream ? stream : offsets_graph.stream;
        err = cudaStreamSynchronize(offsets_stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize (offsets) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        err = CUDA_CHECK(cudaMemcpy(&total_intersections, d_total, sizeof(int), cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
    }

    if (total_intersections > 0) {
        d_results = allocDeviceResults(total_intersections, true);

        write_cfg.d_a_begin = d_a.begin;
        write_cfg.d_a_end = d_a.end;
        write_cfg.d_a_row_offsets = d_a.row_offsets;
        write_cfg.a_row_count = a.row_count();
        write_cfg.d_a_row_to_y = d_a.row_to_y;
        write_cfg.d_a_row_to_z = d_a.row_to_z;
        write_cfg.d_b_begin = d_b.begin;
        write_cfg.d_b_end = d_b.end;
        write_cfg.d_b_row_offsets = d_b.row_offsets;
        write_cfg.b_row_count = b.row_count();
        write_cfg.d_offsets = d_offsets;
        write_cfg.d_r_z_idx = d_results.z_idx;
        write_cfg.d_r_y_idx = d_results.y_idx;
        write_cfg.d_r_begin = d_results.begin;
        write_cfg.d_r_end = d_results.end;
        write_cfg.d_a_idx = d_results.a_idx;
        write_cfg.d_b_idx = d_results.b_idx;
        write_cfg.total_capacity = total_intersections;

        err = createVolumeIntersectionWriteGraph(&write_graph, write_cfg, stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "createVolumeIntersectionWriteGraph failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        err = launchVolumeIntersectionGraph(write_graph, stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "launchVolumeIntersectionGraph (write) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        cudaStream_t write_stream = stream ? stream : write_graph.stream;
        err = cudaStreamSynchronize(write_stream);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaStreamSynchronize (write) failed: " << cudaGetErrorString(err);
            goto cleanup;
        }

        const size_t results_bytes = static_cast<size_t>(total_intersections) * sizeof(int);
        std::vector<int> h_z(total_intersections);
        std::vector<int> h_y(total_intersections);
        std::vector<int> h_begin(total_intersections);
        std::vector<int> h_end(total_intersections);
        std::vector<int> h_a(total_intersections);
        std::vector<int> h_b(total_intersections);

        err = CUDA_CHECK(cudaMemcpy(h_z.data(), d_results.z_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_y.data(), d_results.y_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_begin.data(), d_results.begin, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_end.data(), d_results.end, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_a.data(), d_results.a_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }
        err = CUDA_CHECK(cudaMemcpy(h_b.data(), d_results.b_idx, results_bytes, cudaMemcpyDeviceToHost));
        if (err != cudaSuccess) {
            goto cleanup;
        }

        host_results.reserve(static_cast<size_t>(total_intersections));
        for (int i = 0; i < total_intersections; ++i) {
            host_results.push_back({h_z[i], h_y[i], h_begin[i], h_end[i], h_a[i], h_b[i]});
        }
    }

cleanup:
    destroyVolumeIntersectionGraph(&write_graph);
    destroyVolumeIntersectionGraph(&offsets_graph);

    freeDeviceResults(d_results);
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_scan_temp) CUDA_CHECK(cudaFree(d_scan_temp));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
    return host_results;
}

void expectVolumeIntersectionsWorkspace(const VolumeHost& a,
                                        const VolumeHost& b,
                                        const std::vector<IntersectionResult>& expected,
                                        bool use_stream) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    auto actual = runVolumeIntersections(a, b, stream);
    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
}

struct VolumeSpan {
    int z_idx;
    int y_idx;
    int r_begin;
    int r_end;

    bool operator==(const VolumeSpan& other) const {
        return z_idx == other.z_idx && y_idx == other.y_idx &&
               r_begin == other.r_begin && r_end == other.r_end;
    }

    bool operator<(const VolumeSpan& other) const {
        return std::tie(z_idx, y_idx, r_begin, r_end) <
               std::tie(other.z_idx, other.y_idx, other.r_begin, other.r_end);
    }
};

std::vector<VolumeSpan> runVolumeUnion(const VolumeHost& a,
                                       const VolumeHost& b,
                                       cudaStream_t stream = nullptr) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    const int rows = a.row_count();
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeVolumeUnionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
        d_counts, d_offsets,
        &total,
        stream);

    if (err != cudaSuccess) {
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        freeDeviceVolume(d_a);
        freeDeviceVolume(d_b);
        ADD_FAILURE() << "computeVolumeUnionOffsets failed: " << cudaGetErrorString(err);
        return {};
    }

    std::vector<VolumeSpan> host_results;
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;

    if (total > 0) {
        const size_t results_bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_z_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));

        err = writeVolumeUnionWithOffsets(
            d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
            d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
            d_a.row_to_y, d_a.row_to_z,
            d_offsets,
            d_r_z_idx,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            stream);

        if (err != cudaSuccess) {
            CUDA_CHECK(cudaFree(d_r_z_idx));
            CUDA_CHECK(cudaFree(d_r_y_idx));
            CUDA_CHECK(cudaFree(d_r_begin));
            CUDA_CHECK(cudaFree(d_r_end));
            if (d_counts) CUDA_CHECK(cudaFree(d_counts));
            if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
            freeDeviceVolume(d_a);
            freeDeviceVolume(d_b);
            ADD_FAILURE() << "writeVolumeUnionWithOffsets failed: " << cudaGetErrorString(err);
            return {};
        }

        host_results.resize(total);
        std::vector<int> h_z(total);
        std::vector<int> h_y(total);
        std::vector<int> h_begin(total);
        std::vector<int> h_end(total);
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_z.data(), d_r_z_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_begin.data(), d_r_begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), d_r_end, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < total; ++i) {
            host_results[i] = {h_z[i], h_y[i], h_begin[i], h_end[i]};
        }

        CUDA_CHECK(cudaFree(d_r_z_idx));
        CUDA_CHECK(cudaFree(d_r_y_idx));
        CUDA_CHECK(cudaFree(d_r_begin));
        CUDA_CHECK(cudaFree(d_r_end));
    }

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
    return host_results;
}

std::vector<VolumeSpan> runVolumeDifference(const VolumeHost& a,
                                            const VolumeHost& b,
                                            cudaStream_t stream = nullptr) {
    DeviceVolume d_a = copyToDevice(a, true);
    DeviceVolume d_b = copyToDevice(b, false);

    const int rows = a.row_count();
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    int total = 0;
    cudaError_t err = computeVolumeDifferenceOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
        d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
        d_counts, d_offsets,
        &total,
        stream);

    if (err != cudaSuccess) {
        if (d_counts) CUDA_CHECK(cudaFree(d_counts));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        freeDeviceVolume(d_a);
        freeDeviceVolume(d_b);
        ADD_FAILURE() << "computeVolumeDifferenceOffsets failed: " << cudaGetErrorString(err);
        return {};
    }

    std::vector<VolumeSpan> host_results;
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;

    if (total > 0) {
        const size_t results_bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_z_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, results_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, results_bytes));

        err = writeVolumeDifferenceWithOffsets(
            d_a.begin, d_a.end, d_a.row_offsets, a.row_count(),
            d_b.begin, d_b.end, d_b.row_offsets, b.row_count(),
            d_a.row_to_y, d_a.row_to_z,
            d_offsets,
            d_r_z_idx,
            d_r_y_idx,
            d_r_begin,
            d_r_end,
            stream);

        if (err != cudaSuccess) {
            CUDA_CHECK(cudaFree(d_r_z_idx));
            CUDA_CHECK(cudaFree(d_r_y_idx));
            CUDA_CHECK(cudaFree(d_r_begin));
            CUDA_CHECK(cudaFree(d_r_end));
            if (d_counts) CUDA_CHECK(cudaFree(d_counts));
            if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
            freeDeviceVolume(d_a);
            freeDeviceVolume(d_b);
            ADD_FAILURE() << "writeVolumeDifferenceWithOffsets failed: " << cudaGetErrorString(err);
            return {};
        }

        host_results.resize(total);
        std::vector<int> h_z(total);
        std::vector<int> h_y(total);
        std::vector<int> h_begin(total);
        std::vector<int> h_end(total);
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(h_z.data(), d_r_z_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_begin.data(), d_r_begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), d_r_end, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < total; ++i) {
            host_results[i] = {h_z[i], h_y[i], h_begin[i], h_end[i]};
        }

        CUDA_CHECK(cudaFree(d_r_z_idx));
        CUDA_CHECK(cudaFree(d_r_y_idx));
        CUDA_CHECK(cudaFree(d_r_begin));
        CUDA_CHECK(cudaFree(d_r_end));
    }

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    freeDeviceVolume(d_a);
    freeDeviceVolume(d_b);
    return host_results;
}

void expectVolumeUnionWorkspace(const VolumeHost& a,
                                const VolumeHost& b,
                                const std::vector<VolumeSpan>& expected,
                                bool use_stream = false) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    auto actual = runVolumeUnion(a, b, stream);
    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
}

void expectVolumeDifferenceWorkspace(const VolumeHost& a,
                                     const VolumeHost& b,
                                     const std::vector<VolumeSpan>& expected,
                                     bool use_stream = false) {
    cudaStream_t stream = nullptr;
    if (use_stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    auto actual = runVolumeDifference(a, b, stream);
    if (use_stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    ASSERT_EQ(actual.size(), expected.size());
    auto sorted_actual = actual;
    auto sorted_expected = expected;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
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

    expectVolumeIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectVolumeIntersectionsWorkspace(a, b, {}, /*use_stream=*/false);
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

    expectVolumeIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectVolumeIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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

    expectVolumeIntersectionsWorkspace(a, b, expected, /*use_stream=*/true);
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

TEST(VolumeSetOpsTest, UnionMergesRows) {
    VolumeHost a = buildVolume(2, 1, {
        {{0, 4}},
        {{4, 6}}
    });

    VolumeHost b = buildVolume(2, 1, {
        {{2, 5}},
        {{1, 3}, {6, 8}}
    });

    std::vector<VolumeSpan> expected = {
        {0, 0, 0, 5},
        {1, 0, 1, 3},
        {1, 0, 4, 8}
    };

    expectVolumeUnionWorkspace(a, b, expected);
}

TEST(VolumeSetOpsTest, DifferenceSlicesRows) {
    VolumeHost a = buildVolume(2, 1, {
        {{0, 10}},
        {{2, 9}}
    });

    VolumeHost b = buildVolume(2, 1, {
        {{3, 7}},
        {{1, 3}, {5, 7}}
    });

    std::vector<VolumeSpan> expected = {
        {0, 0, 0, 3},
        {0, 0, 7, 10},
        {1, 0, 3, 5},
        {1, 0, 7, 9}
    };

    expectVolumeDifferenceWorkspace(a, b, expected);
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

    const int rows = d_a.row_count;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));
    }

    cudaError_t err = computeVolumeIntersectionOffsets(
        d_a.begin, d_a.end, d_a.row_offsets, d_a.row_count,
        d_b.begin, d_b.end, d_b.row_offsets, d_b.row_count,
        d_counts, d_offsets,
        nullptr,
        nullptr);

    EXPECT_EQ(err, cudaErrorInvalidValue);

    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
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

    expectVolumeIntersectionsWorkspace(a, b, expected, /*use_stream=*/false);
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
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* d_total = nullptr;
    VolumeIntersectionGraph offsets_graph{};
    VolumeIntersectionGraph write_graph{};

    if (rows > 0) {
        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_counts, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_offsets, row_bytes));

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_temp_bytes,
            d_counts,
            d_offsets,
            rows,
            stream));
        if (scan_temp_bytes == 0) {
            scan_temp_bytes = sizeof(int);
        }
        CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
        CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    }

    VolumeIntersectionOffsetsGraphConfig offsets_cfg{};
    offsets_cfg.d_a_begin = d_a.begin;
    offsets_cfg.d_a_end = d_a.end;
    offsets_cfg.d_a_row_offsets = d_a.row_offsets;
    offsets_cfg.a_row_count = rows;
    offsets_cfg.d_b_begin = d_b.begin;
    offsets_cfg.d_b_end = d_b.end;
    offsets_cfg.d_b_row_offsets = d_b.row_offsets;
    offsets_cfg.b_row_count = rows;
    offsets_cfg.d_counts = d_counts;
    offsets_cfg.d_offsets = d_offsets;
    offsets_cfg.d_scan_temp_storage = d_scan_temp;
    offsets_cfg.scan_temp_storage_bytes = scan_temp_bytes;
    offsets_cfg.d_total = d_total;

    cudaError_t err = createVolumeIntersectionOffsetsGraph(&offsets_graph, offsets_cfg, stream);
    ASSERT_EQ(err, cudaSuccess);
    err = launchVolumeIntersectionGraph(offsets_graph, stream);
    ASSERT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int total = 0;
    if (rows > 0) {
        CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    }
    ASSERT_EQ(total, 1);

    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;

    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_r_y_idx, bytes));
        CUDA_CHECK(cudaMalloc(&d_r_begin, bytes));
        CUDA_CHECK(cudaMalloc(&d_r_end, bytes));
        CUDA_CHECK(cudaMalloc(&d_a_idx, bytes));
        CUDA_CHECK(cudaMalloc(&d_b_idx, bytes));

        VolumeIntersectionWriteGraphConfig write_cfg{};
        write_cfg.d_a_begin = d_a.begin;
        write_cfg.d_a_end = d_a.end;
        write_cfg.d_a_row_offsets = d_a.row_offsets;
        write_cfg.a_row_count = rows;
        write_cfg.d_a_row_to_y = nullptr;
        write_cfg.d_a_row_to_z = nullptr;
        write_cfg.d_b_begin = d_b.begin;
        write_cfg.d_b_end = d_b.end;
        write_cfg.d_b_row_offsets = d_b.row_offsets;
        write_cfg.b_row_count = rows;
        write_cfg.d_offsets = d_offsets;
        write_cfg.d_r_z_idx = nullptr;
        write_cfg.d_r_y_idx = d_r_y_idx;
        write_cfg.d_r_begin = d_r_begin;
        write_cfg.d_r_end = d_r_end;
        write_cfg.d_a_idx = d_a_idx;
        write_cfg.d_b_idx = d_b_idx;
        write_cfg.total_capacity = total;

        err = createVolumeIntersectionWriteGraph(&write_graph, write_cfg, stream);
        ASSERT_EQ(err, cudaSuccess);
        err = launchVolumeIntersectionGraph(write_graph, stream);
        ASSERT_EQ(err, cudaSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int h_y = -1, h_beg = -1, h_end = -1, h_ai = -1, h_bi = -1;
    if (total > 0) {
        const size_t bytes = static_cast<size_t>(total) * sizeof(int);
        CUDA_CHECK(cudaMemcpy(&h_y, d_r_y_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_beg, d_r_begin, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_end, d_r_end, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_ai, d_a_idx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_bi, d_b_idx, bytes, cudaMemcpyDeviceToHost));
    }
    EXPECT_EQ(h_y, 0);
    EXPECT_EQ(h_beg, 2);
    EXPECT_EQ(h_end, 4);
    EXPECT_EQ(h_ai, 0);
    EXPECT_EQ(h_bi, 0);

    destroyVolumeIntersectionGraph(&write_graph);
    destroyVolumeIntersectionGraph(&offsets_graph);
    if (d_total) CUDA_CHECK(cudaFree(d_total));
    if (d_scan_temp) CUDA_CHECK(cudaFree(d_scan_temp));
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
