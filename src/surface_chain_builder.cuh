#ifndef SUBSETIX_SURFACE_CHAIN_BUILDER_CUH
#define SUBSETIX_SURFACE_CHAIN_BUILDER_CUH

#include <cuda_runtime.h>

#include <vector>

#include "surface_chain_executor.cuh"

namespace subsetix
{
    struct SurfaceDescriptor
    {
        SurfaceView view{};
        int interval_count = 0;
    };

    class SurfaceChainBuilder
    {
    public:
        SurfaceChainBuilder() = default;

        SurfaceHandle add_input(const SurfaceDescriptor& descriptor);
        SurfaceHandle add_union(SurfaceHandle lhs, SurfaceHandle rhs);
        SurfaceHandle add_difference(SurfaceHandle lhs, SurfaceHandle rhs);
        SurfaceHandle add_intersection(SurfaceHandle lhs, SurfaceHandle rhs);

        const SurfaceOperationChain& chain() const { return chain_; }
        const std::vector<SurfaceDescriptor>& descriptors() const { return descriptors_; }

    private:
        SurfaceOperationChain chain_{};
        std::vector<SurfaceDescriptor> descriptors_;
    };

    struct SurfaceChainRunResult
    {
        int total = 0;
        int* d_begin = nullptr;
        int* d_end = nullptr;
        int* d_y_idx = nullptr;
        int* d_a_idx = nullptr;
        int* d_b_idx = nullptr;
    };

    class SurfaceChainRunner
    {
    public:
        explicit SurfaceChainRunner(const SurfaceChainBuilder& builder);
        SurfaceChainRunner(const SurfaceChainRunner&) = delete;
        SurfaceChainRunner& operator=(const SurfaceChainRunner&) = delete;
        SurfaceChainRunner(SurfaceChainRunner&&) = delete;
        SurfaceChainRunner& operator=(SurfaceChainRunner&&) = delete;
        ~SurfaceChainRunner();

        cudaError_t prepare(cudaStream_t stream = nullptr);
        cudaError_t run(cudaStream_t stream = nullptr);

        const SurfaceChainRunResult& result() const { return result_; }

    private:
        void cleanup();
        cudaError_t allocate_workspace();
        cudaError_t allocate_node_buffers();
        void compute_node_capacities();

        SurfaceOperationChain chain_{};
        std::vector<SurfaceDescriptor> descriptors_;
        SurfaceChainExecutor executor_{};

        SurfaceWorkspaceView workspace_{};
        SurfaceChainGraph offsets_graph_{};
        SurfaceChainGraph write_graph_{};

        std::vector<size_t> node_capacities_;
        std::vector<SurfaceArenaSlice> slices_;
        SurfaceArenaView offsets_arena_{};
        SurfaceArenaView write_arena_{};

        int* d_counts_ = nullptr;
        int* d_offsets_ = nullptr;
        int* d_totals_ = nullptr;
        void* d_scan_temp_ = nullptr;

        std::vector<int*> node_begin_;
        std::vector<int*> node_end_;
        std::vector<int*> node_y_;
        std::vector<int*> node_a_idx_;
        std::vector<int*> node_b_idx_;

        std::vector<int> host_totals_;
        SurfaceChainRunResult result_{};
    };

} // namespace subsetix

#endif // SUBSETIX_SURFACE_CHAIN_BUILDER_CUH
