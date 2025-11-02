#include "surface_chain_executor.cuh"
#include "interval_intersection.cuh"

#include <algorithm>
#include <vector>

namespace subsetix
{
    namespace
    {
        bool is_operand_available(const SurfaceOperandRef& operand, int node_index, size_t input_count)
        {
            if (operand.from_node) {
                return operand.index >= 0 && operand.index < node_index;
            }
            return operand.index >= 0 && static_cast<size_t>(operand.index) < input_count;
        }

        __global__ void finalize_surface_offsets_kernel(const int* d_counts,
                                                        int* d_offsets,
                                                        int row_count,
                                                        int* d_total)
        {
            if (threadIdx.x > 0) {
                return;
            }

            int total = 0;
            if (row_count > 0 && d_counts && d_offsets) {
                const int last_index = row_count - 1;
                total = d_offsets[last_index] + d_counts[last_index];
            }

            if (d_offsets) {
                d_offsets[row_count] = total;
            }

            if (d_total) {
                d_total[0] = total;
            }
        }
    } // namespace

    void SurfaceChainExecutor::reset()
    {
        row_count_ = -1;
        inputs_.clear();
        plan_.clear();
        counts_stride_ = 0;
        offsets_stride_ = 0;
        required_arena_capacity_ = 0;
        nodes_requiring_materialization_ = 0;
        needs_materialization_.clear();
        produce_in_offsets_.clear();
    }

    cudaError_t SurfaceChainExecutor::prepare(const SurfaceOperationChain& chain)
    {
        reset();
        if (chain.empty()) {
            return cudaErrorInvalidValue;
        }
        if (chain.row_count() < 0) {
            return cudaErrorInvalidValue;
        }

        row_count_ = chain.row_count();
        inputs_ = chain.inputs();
        plan_.reserve(chain.nodes().size());

        for (size_t node_idx = 0; node_idx < chain.nodes().size(); ++node_idx) {
            const auto& node = chain.nodes()[node_idx];
            SurfaceNodePlan plan_node;
            plan_node.type = node.type;
            plan_node.lhs = SurfaceOperandRef{node.lhs.is_node, node.lhs.index};
            plan_node.rhs = SurfaceOperandRef{node.rhs.is_node, node.rhs.index};

            if (!is_operand_available(plan_node.lhs, static_cast<int>(node_idx), inputs_.size()) ||
                !is_operand_available(plan_node.rhs, static_cast<int>(node_idx), inputs_.size())) {
                reset();
                return cudaErrorInvalidValue;
            }

            plan_.push_back(plan_node);
        }
        return cudaSuccess;
    }

    cudaError_t SurfaceChainExecutor::plan_resources()
    {
        if (plan_.empty() || row_count_ < 0) {
            return cudaErrorInvalidValue;
        }

        counts_stride_ = static_cast<size_t>(row_count_ > 0 ? row_count_ : 1);
        offsets_stride_ = static_cast<size_t>(row_count_) + 1;

        needs_materialization_.assign(plan_.size(), false);
        produce_in_offsets_.assign(plan_.size(), false);
        required_arena_capacity_ = 0;
        nodes_requiring_materialization_ = 0;

        std::vector<int> usage_counts(plan_.size(), 0);
        for (size_t node_idx = 0; node_idx < plan_.size(); ++node_idx) {
            const auto& node = plan_[node_idx];
            if (node.lhs.from_node && static_cast<size_t>(node.lhs.index) < plan_.size()) {
                ++usage_counts[node.lhs.index];
            }
            if (node.rhs.from_node && static_cast<size_t>(node.rhs.index) < plan_.size()) {
                ++usage_counts[node.rhs.index];
            }
        }

        for (size_t node_idx = 0; node_idx < plan_.size(); ++node_idx) {
            const bool consumed = usage_counts[node_idx] > 0;
            const bool final_output = (node_idx == plan_.size() - 1);
            needs_materialization_[node_idx] = consumed || final_output;
            produce_in_offsets_[node_idx] = consumed;
            if (needs_materialization_[node_idx]) {
                ++nodes_requiring_materialization_;
            }
        }

        return cudaSuccess;
    }

    cudaError_t SurfaceChainExecutor::capture_offsets_graph(const SurfaceWorkspaceView& workspace,
                                                            const SurfaceArenaView& arena,
                                                            SurfaceChainGraph* graph,
                                                            cudaStream_t stream)
    {
        if (!graph) {
            return cudaErrorInvalidValue;
        }

        destroy_surface_chain_graph(graph);

        if (plan_.empty() || row_count_ < 0) {
            return cudaErrorInvalidValue;
        }

        if (needs_materialization_.size() != plan_.size()) {
            return cudaErrorInvalidValue;
        }

        if (!workspace.d_counts || !workspace.d_offsets || !workspace.d_totals) {
            return cudaErrorInvalidValue;
        }
        if (workspace.scan_temp_bytes > 0 && !workspace.d_scan_temp) {
            return cudaErrorInvalidValue;
        }
        if (workspace.counts_stride != counts_stride_ || workspace.offsets_stride != offsets_stride_) {
            return cudaErrorInvalidValue;
        }

        if (!plan_.empty() && arena.count > 0 && arena.count < plan_.size()) {
            return cudaErrorInvalidValue;
        }

        for (size_t idx = 0; idx < plan_.size(); ++idx) {
            if (!produce_in_offsets_[idx]) {
                continue;
            }
            if (!arena.slices || static_cast<size_t>(arena.count) <= idx) {
                return cudaErrorInvalidValue;
            }
            const auto& slice = arena.slices[idx];
            if (row_count_ > 0) {
                if (!slice.d_begin || !slice.d_end || !slice.d_y_idx) {
                    return cudaErrorInvalidValue;
                }
                if (plan_[idx].type == SurfaceOpType::Intersection &&
                    (!slice.d_a_idx || !slice.d_b_idx)) {
                    return cudaErrorInvalidValue;
                }
            }
        }

        auto resolve_operand = [&](const SurfaceOperandRef& operand) -> SurfaceView {
            if (operand.from_node) {
                const size_t source_idx = static_cast<size_t>(operand.index);
                if (source_idx >= plan_.size()) {
                    return SurfaceView{};
                }
                if (!produce_in_offsets_[source_idx]) {
                    return SurfaceView{};
                }
                const SurfaceArenaSlice& slice = arena.slices ? arena.slices[source_idx] : SurfaceArenaSlice{};
                SurfaceView view{};
                view.d_begin = slice.d_begin;
                view.d_end = slice.d_end;
                view.d_row_offsets = workspace.d_offsets + source_idx * offsets_stride_;
                view.row_count = row_count_;
                return view;
            }
            const size_t input_idx = static_cast<size_t>(operand.index);
            if (input_idx >= inputs_.size()) {
                return SurfaceView{};
            }
            return inputs_[input_idx];
        };

        cudaStream_t capture_stream = stream;
        bool owns_stream = false;
        if (!capture_stream) {
            cudaError_t err_create = cudaStreamCreate(&capture_stream);
            if (err_create != cudaSuccess) {
                return err_create;
            }
            owns_stream = true;
        }

        cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        auto abort_capture = [&](cudaError_t status) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return status;
        };

        for (size_t node_idx = 0; node_idx < plan_.size(); ++node_idx) {
            int* node_counts = workspace.d_counts + node_idx * counts_stride_;
            int* node_offsets = workspace.d_offsets + node_idx * offsets_stride_;
            int* node_total = workspace.d_totals + node_idx;

            const SurfaceNodePlan& node = plan_[node_idx];

            SurfaceView lhs{};
            SurfaceView rhs{};

            if (row_count_ > 0) {
                lhs = resolve_operand(node.lhs);
                rhs = resolve_operand(node.rhs);
                if (!lhs.d_begin || !lhs.d_end || !lhs.d_row_offsets ||
                    !rhs.d_begin || !rhs.d_end || !rhs.d_row_offsets) {
                    return abort_capture(cudaErrorInvalidValue);
                }

                switch (node.type) {
                    case SurfaceOpType::Difference:
                        err = enqueueIntervalDifferenceOffsets(lhs.d_begin, lhs.d_end, lhs.d_row_offsets, lhs.row_count,
                                                                rhs.d_begin, rhs.d_end, rhs.d_row_offsets, rhs.row_count,
                                                                node_counts,
                                                                node_offsets,
                                                                capture_stream,
                                                                workspace.d_scan_temp,
                                                                workspace.scan_temp_bytes);
                        break;
                    case SurfaceOpType::Intersection:
                        err = enqueueIntervalIntersectionOffsets(lhs.d_begin, lhs.d_end, lhs.d_row_offsets, lhs.row_count,
                                                                 rhs.d_begin, rhs.d_end, rhs.d_row_offsets, rhs.row_count,
                                                                 node_counts,
                                                                 node_offsets,
                                                                 capture_stream,
                                                                 workspace.d_scan_temp,
                                                                 workspace.scan_temp_bytes);
                        break;
                    case SurfaceOpType::Union:
                        err = enqueueIntervalUnionOffsets(lhs.d_begin, lhs.d_end, lhs.d_row_offsets, lhs.row_count,
                                                          rhs.d_begin, rhs.d_end, rhs.d_row_offsets, rhs.row_count,
                                                          node_counts,
                                                          node_offsets,
                                                          capture_stream,
                                                          workspace.d_scan_temp,
                                                          workspace.scan_temp_bytes);
                        break;
                    default:
                        err = cudaErrorInvalidValue;
                        break;
                }

                if (err != cudaSuccess) {
                    return abort_capture(err);
                }
            }

            finalize_surface_offsets_kernel<<<1, 1, 0, capture_stream>>>(node_counts,
                                                                         node_offsets,
                                                                         row_count_,
                                                                         node_total);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return abort_capture(err);
            }

            if (produce_in_offsets_[node_idx] && row_count_ > 0) {
                const auto& slice = arena.slices[node_idx];
                switch (node.type) {
                    case SurfaceOpType::Difference:
                        err = enqueueIntervalDifferenceWrite(lhs.d_begin,
                                                              lhs.d_end,
                                                              lhs.d_row_offsets,
                                                              row_count_,
                                                              rhs.d_begin,
                                                              rhs.d_end,
                                                              rhs.d_row_offsets,
                                                              row_count_,
                                                              node_offsets,
                                                              slice.d_y_idx,
                                                              slice.d_begin,
                                                              slice.d_end,
                                                              capture_stream);
                        break;
                    case SurfaceOpType::Intersection:
                        err = enqueueIntervalIntersectionWrite(lhs.d_begin,
                                                               lhs.d_end,
                                                               lhs.d_row_offsets,
                                                               row_count_,
                                                               rhs.d_begin,
                                                               rhs.d_end,
                                                               rhs.d_row_offsets,
                                                               row_count_,
                                                               node_offsets,
                                                               slice.d_y_idx,
                                                               slice.d_begin,
                                                               slice.d_end,
                                                               slice.d_a_idx,
                                                               slice.d_b_idx,
                                                               capture_stream);
                        break;
                    case SurfaceOpType::Union:
                        err = enqueueIntervalUnionWrite(lhs.d_begin,
                                                        lhs.d_end,
                                                        lhs.d_row_offsets,
                                                        row_count_,
                                                        rhs.d_begin,
                                                        rhs.d_end,
                                                        rhs.d_row_offsets,
                                                        row_count_,
                                                        node_offsets,
                                                        slice.d_y_idx,
                                                        slice.d_begin,
                                                        slice.d_end,
                                                        capture_stream);
                        break;
                    default:
                        err = cudaErrorInvalidValue;
                        break;
                }

                if (err != cudaSuccess) {
                    return abort_capture(err);
                }
            }

        }

        cudaGraph_t captured_graph = nullptr;
        err = cudaStreamEndCapture(capture_stream, &captured_graph);
        if (err != cudaSuccess) {
            if (captured_graph) {
                cudaGraphDestroy(captured_graph);
            }
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        cudaGraphExec_t exec = nullptr;
        err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(captured_graph);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        graph->graph = captured_graph;
        graph->exec = exec;
        graph->stream = capture_stream;
        graph->owns_stream = owns_stream;
        return cudaSuccess;
    }

    cudaError_t SurfaceChainExecutor::capture_write_graph(const SurfaceWorkspaceView& workspace,
                                                          const SurfaceArenaView& arena,
                                                          SurfaceChainGraph* graph,
                                                          cudaStream_t stream)
    {
        if (!graph) {
            return cudaErrorInvalidValue;
        }

        destroy_surface_chain_graph(graph);

        if (plan_.empty() || row_count_ < 0) {
            return cudaErrorInvalidValue;
        }

        if (needs_materialization_.size() != plan_.size()) {
            return cudaErrorInvalidValue;
        }

        if (!workspace.d_offsets) {
            return cudaErrorInvalidValue;
        }

        if (!plan_.empty() && arena.count > 0 && arena.count < plan_.size()) {
            return cudaErrorInvalidValue;
        }

        for (size_t idx = 0; idx < plan_.size(); ++idx) {
            if (!needs_materialization_[idx] || produce_in_offsets_[idx]) {
                continue;
            }
            if (!arena.slices || static_cast<size_t>(arena.count) <= idx) {
                return cudaErrorInvalidValue;
            }
            const auto& slice = arena.slices[idx];
            if (row_count_ > 0) {
                if (!slice.d_begin || !slice.d_end || !slice.d_y_idx) {
                    return cudaErrorInvalidValue;
                }
                if (plan_[idx].type == SurfaceOpType::Intersection &&
                    (!slice.d_a_idx || !slice.d_b_idx)) {
                    return cudaErrorInvalidValue;
                }
            }
        }

        auto resolve_operand = [&](const SurfaceOperandRef& operand) -> SurfaceView {
            if (operand.from_node) {
                const size_t source_idx = static_cast<size_t>(operand.index);
                if (source_idx >= plan_.size() || !needs_materialization_[source_idx]) {
                    return SurfaceView{};
                }
                const SurfaceArenaSlice& slice = arena.slices ? arena.slices[source_idx] : SurfaceArenaSlice{};
                SurfaceView view{};
                view.d_begin = slice.d_begin;
                view.d_end = slice.d_end;
                view.d_row_offsets = workspace.d_offsets + source_idx * offsets_stride_;
                view.row_count = row_count_;
                return view;
            }
            const size_t input_idx = static_cast<size_t>(operand.index);
            if (input_idx >= inputs_.size()) {
                return SurfaceView{};
            }
            return inputs_[input_idx];
        };

        cudaStream_t capture_stream = stream;
        bool owns_stream = false;
        if (!capture_stream) {
            cudaError_t err_create = cudaStreamCreate(&capture_stream);
            if (err_create != cudaSuccess) {
                return err_create;
            }
            owns_stream = true;
        }

        cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        auto abort_capture = [&](cudaError_t status) {
            cudaStreamEndCapture(capture_stream, nullptr);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return status;
        };

        for (size_t node_idx = 0; node_idx < plan_.size(); ++node_idx) {
            if (!needs_materialization_[node_idx] || produce_in_offsets_[node_idx]) {
                continue;
            }

            int* node_offsets = workspace.d_offsets + node_idx * offsets_stride_;
            const auto& slice = arena.slices[node_idx];
            const SurfaceNodePlan& node = plan_[node_idx];

            if (row_count_ == 0) {
                continue;
            }

            SurfaceView lhs = resolve_operand(node.lhs);
            SurfaceView rhs = resolve_operand(node.rhs);
            if (!lhs.d_begin || !lhs.d_end || !lhs.d_row_offsets ||
                !rhs.d_begin || !rhs.d_end || !rhs.d_row_offsets) {
                return abort_capture(cudaErrorInvalidValue);
            }

            switch (node.type) {
                case SurfaceOpType::Difference:
                    err = enqueueIntervalDifferenceWrite(lhs.d_begin,
                                                          lhs.d_end,
                                                          lhs.d_row_offsets,
                                                          row_count_,
                                                          rhs.d_begin,
                                                          rhs.d_end,
                                                          rhs.d_row_offsets,
                                                          row_count_,
                                                          node_offsets,
                                                          slice.d_y_idx,
                                                          slice.d_begin,
                                                          slice.d_end,
                                                          capture_stream);
                    break;
                case SurfaceOpType::Intersection:
                    err = enqueueIntervalIntersectionWrite(lhs.d_begin,
                                                           lhs.d_end,
                                                           lhs.d_row_offsets,
                                                           row_count_,
                                                           rhs.d_begin,
                                                           rhs.d_end,
                                                           rhs.d_row_offsets,
                                                           row_count_,
                                                           node_offsets,
                                                           slice.d_y_idx,
                                                           slice.d_begin,
                                                           slice.d_end,
                                                           slice.d_a_idx,
                                                           slice.d_b_idx,
                                                           capture_stream);
                    break;
                case SurfaceOpType::Union:
                    err = enqueueIntervalUnionWrite(lhs.d_begin,
                                                    lhs.d_end,
                                                    lhs.d_row_offsets,
                                                    row_count_,
                                                    rhs.d_begin,
                                                    rhs.d_end,
                                                    rhs.d_row_offsets,
                                                    row_count_,
                                                    node_offsets,
                                                    slice.d_y_idx,
                                                    slice.d_begin,
                                                    slice.d_end,
                                                    capture_stream);
                    break;
                default:
                    err = cudaErrorInvalidValue;
                    break;
            }

            if (err != cudaSuccess) {
                return abort_capture(err);
            }
        }

        cudaGraph_t captured_graph = nullptr;
        err = cudaStreamEndCapture(capture_stream, &captured_graph);
        if (err != cudaSuccess) {
            if (captured_graph) {
                cudaGraphDestroy(captured_graph);
            }
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        cudaGraphExec_t exec = nullptr;
        err = cudaGraphInstantiate(&exec, captured_graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(captured_graph);
            if (owns_stream) {
                cudaStreamDestroy(capture_stream);
            }
            return err;
        }

        graph->graph = captured_graph;
        graph->exec = exec;
        graph->stream = capture_stream;
        graph->owns_stream = owns_stream;
        return cudaSuccess;
    }

    cudaError_t launch_surface_chain_graph(const SurfaceChainGraph& graph, cudaStream_t stream)
    {
        if (!graph.exec) {
            return cudaErrorInvalidValue;
        }
        cudaStream_t launch_stream = stream ? stream : graph.stream;
        return cudaGraphLaunch(graph.exec, launch_stream);
    }

    void destroy_surface_chain_graph(SurfaceChainGraph* graph)
    {
        if (!graph) {
            return;
        }
        if (graph->exec) {
            cudaGraphExecDestroy(graph->exec);
        }
        if (graph->graph) {
            cudaGraphDestroy(graph->graph);
        }
        if (graph->owns_stream && graph->stream) {
            cudaStreamDestroy(graph->stream);
        }
        graph->graph = nullptr;
        graph->exec = nullptr;
        graph->stream = nullptr;
        graph->owns_stream = false;
    }

} // namespace subsetix
