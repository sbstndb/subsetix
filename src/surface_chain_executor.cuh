#ifndef SUBSETIX_SURFACE_CHAIN_EXECUTOR_CUH
#define SUBSETIX_SURFACE_CHAIN_EXECUTOR_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "operation_chain.cuh"

namespace subsetix
{
    struct SurfaceOperandRef
    {
        bool from_node = false;
        int index = -1;

        SurfaceOperandRef() = default;
        SurfaceOperandRef(bool fromNode, int idx)
            : from_node(fromNode)
            , index(idx)
        {}
    };

    struct SurfaceNodePlan
    {
        SurfaceOpType type = SurfaceOpType::Intersection;
        SurfaceOperandRef lhs{};
        SurfaceOperandRef rhs{};
    };

    struct SurfaceWorkspaceView
    {
        int* d_counts = nullptr;
        int* d_offsets = nullptr;
        size_t counts_stride = 0;
        size_t offsets_stride = 0;
        void* d_scan_temp = nullptr;
        size_t scan_temp_bytes = 0;
        int* d_totals = nullptr;
    };

    struct SurfaceArenaSlice
    {
        int* d_begin = nullptr;
        int* d_end = nullptr;
        int* d_y_idx = nullptr;
        int* d_z_idx = nullptr;
        int* d_a_idx = nullptr;
        int* d_b_idx = nullptr;
        size_t capacity = 0;
    };

    struct SurfaceArenaView
    {
        const SurfaceArenaSlice* slices = nullptr;
        size_t count = 0;
    };

    struct SurfaceChainGraph
    {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStream_t stream = nullptr;
        bool owns_stream = false;
    };

    class SurfaceChainExecutor
    {
    public:
        SurfaceChainExecutor() = default;

        cudaError_t prepare(const SurfaceOperationChain& chain);

        void reset();

        int row_count() const { return row_count_; }
        const std::vector<SurfaceView>& inputs() const { return inputs_; }
        const std::vector<SurfaceNodePlan>& plan() const { return plan_; }

        bool empty() const { return plan_.empty(); }

        size_t node_count() const { return plan_.size(); }

        size_t counts_stride() const { return counts_stride_; }
        size_t offsets_stride() const { return offsets_stride_; }
        size_t workspace_counts_elements() const { return counts_stride_ * plan_.size(); }

        size_t required_arena_capacity() const { return required_arena_capacity_; }
        size_t nodes_requiring_materialization() const { return nodes_requiring_materialization_; }
        const std::vector<bool>& materialization_flags() const { return needs_materialization_; }
        const std::vector<bool>& offsets_materialization_flags() const { return produce_in_offsets_; }

        cudaError_t plan_resources();
        cudaError_t capture_offsets_graph(const SurfaceWorkspaceView& workspace,
                                          const SurfaceArenaView& arena,
                                          SurfaceChainGraph* graph,
                                          cudaStream_t stream = nullptr);
        cudaError_t capture_write_graph(const SurfaceWorkspaceView& workspace,
                                        const SurfaceArenaView& arena,
                                        SurfaceChainGraph* graph,
                                        cudaStream_t stream = nullptr);

    private:
        int row_count_ = -1;
        std::vector<SurfaceView> inputs_;
        std::vector<SurfaceNodePlan> plan_;

        size_t counts_stride_ = 0;
        size_t offsets_stride_ = 0;
        size_t required_arena_capacity_ = 0;
        size_t nodes_requiring_materialization_ = 0;
        std::vector<bool> needs_materialization_;
        std::vector<bool> produce_in_offsets_;
    };

    cudaError_t launch_surface_chain_graph(const SurfaceChainGraph& graph, cudaStream_t stream = nullptr);
    void destroy_surface_chain_graph(SurfaceChainGraph* graph);

} // namespace subsetix

#endif // SUBSETIX_SURFACE_CHAIN_EXECUTOR_CUH
