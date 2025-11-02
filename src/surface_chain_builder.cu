#include "surface_chain_builder.cuh"

#include <algorithm>

namespace subsetix
{
    SurfaceHandle SurfaceChainBuilder::add_input(const SurfaceDescriptor& descriptor)
    {
        SurfaceHandle handle = chain_.add_input(descriptor.view);
        descriptors_.push_back(descriptor);
        return handle;
    }

    SurfaceHandle SurfaceChainBuilder::add_union(SurfaceHandle lhs, SurfaceHandle rhs)
    {
        return chain_.add_operation(SurfaceOpType::Union, lhs, rhs);
    }

    SurfaceHandle SurfaceChainBuilder::add_difference(SurfaceHandle lhs, SurfaceHandle rhs)
    {
        return chain_.add_operation(SurfaceOpType::Difference, lhs, rhs);
    }

    SurfaceHandle SurfaceChainBuilder::add_intersection(SurfaceHandle lhs, SurfaceHandle rhs)
    {
        return chain_.add_operation(SurfaceOpType::Intersection, lhs, rhs);
    }

    SurfaceChainRunner::SurfaceChainRunner(const SurfaceChainBuilder& builder)
        : chain_(builder.chain())
        , descriptors_(builder.descriptors())
    {
    }

    SurfaceChainRunner::~SurfaceChainRunner()
    {
        cleanup();
    }

    void SurfaceChainRunner::cleanup()
    {
        destroy_surface_chain_graph(&write_graph_);
        destroy_surface_chain_graph(&offsets_graph_);

        for (int* ptr : node_begin_) {
            if (ptr) cudaFree(ptr);
        }
        for (int* ptr : node_end_) {
            if (ptr) cudaFree(ptr);
        }
        for (int* ptr : node_y_) {
            if (ptr) cudaFree(ptr);
        }
        for (int* ptr : node_a_idx_) {
            if (ptr) cudaFree(ptr);
        }
        for (int* ptr : node_b_idx_) {
            if (ptr) cudaFree(ptr);
        }
        node_begin_.clear();
        node_end_.clear();
        node_y_.clear();
        node_a_idx_.clear();
        node_b_idx_.clear();

        if (d_counts_) cudaFree(d_counts_);
        if (d_offsets_) cudaFree(d_offsets_);
        if (d_totals_) cudaFree(d_totals_);
        if (d_scan_temp_) cudaFree(d_scan_temp_);
        d_counts_ = nullptr;
        d_offsets_ = nullptr;
        d_totals_ = nullptr;
        d_scan_temp_ = nullptr;

        slices_.clear();
        node_capacities_.clear();
        host_totals_.clear();
        result_ = {};
        workspace_ = {};
        offsets_arena_ = {};
        write_arena_ = {};
    }

    cudaError_t SurfaceChainRunner::prepare(cudaStream_t stream)
    {
        cleanup();

        cudaError_t err = executor_.prepare(chain_);
        if (err != cudaSuccess) {
            return err;
        }

        err = executor_.plan_resources();
        if (err != cudaSuccess) {
            return err;
        }

        compute_node_capacities();

        err = allocate_workspace();
        if (err != cudaSuccess) {
            return err;
        }

        err = allocate_node_buffers();
        if (err != cudaSuccess) {
            return err;
        }

        workspace_.d_counts = d_counts_;
        workspace_.d_offsets = d_offsets_;
        workspace_.counts_stride = executor_.counts_stride();
        workspace_.offsets_stride = executor_.offsets_stride();
        workspace_.d_scan_temp = d_scan_temp_;
        if (executor_.row_count() == 0) {
            workspace_.scan_temp_bytes = 0;
        }
        workspace_.d_totals = d_totals_;

        offsets_arena_.slices = slices_.data();
        offsets_arena_.count = slices_.size();
        write_arena_ = offsets_arena_;

        err = executor_.capture_offsets_graph(workspace_, offsets_arena_, &offsets_graph_, stream);
        if (err != cudaSuccess) {
            return err;
        }

        err = executor_.capture_write_graph(workspace_, write_arena_, &write_graph_, stream);
        if (err != cudaSuccess) {
            return err;
        }

        result_.d_begin = node_begin_.empty() ? nullptr : node_begin_.back();
        result_.d_end = node_end_.empty() ? nullptr : node_end_.back();
        result_.d_y_idx = node_y_.empty() ? nullptr : node_y_.back();
        result_.d_a_idx = node_a_idx_.empty() ? nullptr : node_a_idx_.back();
        result_.d_b_idx = node_b_idx_.empty() ? nullptr : node_b_idx_.back();
        result_.total = 0;

        return cudaSuccess;
    }

    cudaError_t SurfaceChainRunner::run(cudaStream_t stream)
    {
        cudaStream_t launch_stream = stream ? stream : offsets_graph_.stream;
        cudaError_t err = subsetix::launch_surface_chain_graph(offsets_graph_, stream);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaStreamSynchronize(launch_stream);
        if (err != cudaSuccess) {
            return err;
        }

        if (!host_totals_.empty()) {
            err = CUDA_CHECK(cudaMemcpy(host_totals_.data(),
                                        d_totals_,
                                        host_totals_.size() * sizeof(int),
                                        cudaMemcpyDeviceToHost));
            if (err != cudaSuccess) {
                return err;
            }
        }

        launch_stream = stream ? stream : write_graph_.stream;
        err = subsetix::launch_surface_chain_graph(write_graph_, stream);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaStreamSynchronize(launch_stream);
        if (err != cudaSuccess) {
            return err;
        }

        if (!host_totals_.empty()) {
            result_.total = host_totals_.back();
        }
        return cudaSuccess;
    }

    cudaError_t SurfaceChainRunner::allocate_workspace()
    {
        const size_t counts_elements = executor_.counts_stride() * executor_.node_count();
        const size_t offsets_elements = executor_.offsets_stride() * executor_.node_count();
        host_totals_.assign(executor_.node_count(), 0);

        if (counts_elements > 0) {
            CUDA_CHECK(cudaMalloc(&d_counts_, counts_elements * sizeof(int)));
        }
        if (offsets_elements > 0) {
            CUDA_CHECK(cudaMalloc(&d_offsets_, offsets_elements * sizeof(int)));
        }
        if (!host_totals_.empty()) {
            CUDA_CHECK(cudaMalloc(&d_totals_, host_totals_.size() * sizeof(int)));
        }

        size_t temp_bytes = 0;
        if (executor_.row_count() > 0) {
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,
                                                     temp_bytes,
                                                     d_counts_,
                                                     d_offsets_,
                                                     executor_.row_count()));
            if (temp_bytes > 0) {
                CUDA_CHECK(cudaMalloc(&d_scan_temp_, temp_bytes));
            }
        }
        workspace_.scan_temp_bytes = temp_bytes;
        return cudaSuccess;
    }

    void SurfaceChainRunner::compute_node_capacities()
    {
        const auto& plan = executor_.plan();
        node_capacities_.assign(plan.size(), 0);
        auto capacity_for_operand = [&](const SurfaceOperandRef& operand, size_t current_idx) {
            if (operand.from_node) {
                const size_t idx = static_cast<size_t>(operand.index);
                if (idx < current_idx) {
                    return node_capacities_[idx];
                }
                return node_capacities_[idx];
            }
            return static_cast<size_t>(descriptors_[operand.index].interval_count);
        };

        for (size_t idx = 0; idx < plan.size(); ++idx) {
            const auto& node = plan[idx];
            const size_t lhs_cap = capacity_for_operand(node.lhs, idx);
            const size_t rhs_cap = capacity_for_operand(node.rhs, idx);
            size_t cap = 0;
            switch (node.type) {
                case SurfaceOpType::Union:
                    cap = lhs_cap + rhs_cap;
                    break;
                case SurfaceOpType::Difference:
                    cap = lhs_cap;
                    break;
                case SurfaceOpType::Intersection:
                    cap = std::min(lhs_cap, rhs_cap);
                    break;
            }
            node_capacities_[idx] = cap;
        }
    }

    cudaError_t SurfaceChainRunner::allocate_node_buffers()
    {
        const auto& materialize = executor_.materialization_flags();
        const auto& offsets_materialize = executor_.offsets_materialization_flags();
        const auto& plan = executor_.plan();

        slices_.assign(plan.size(), SurfaceArenaSlice{});
        node_begin_.assign(plan.size(), nullptr);
        node_end_.assign(plan.size(), nullptr);
        node_y_.assign(plan.size(), nullptr);
        node_a_idx_.assign(plan.size(), nullptr);
        node_b_idx_.assign(plan.size(), nullptr);

        for (size_t idx = 0; idx < plan.size(); ++idx) {
            if (!materialize[idx] && !offsets_materialize[idx]) {
                continue;
            }

            const size_t capacity = std::max<size_t>(1, node_capacities_[idx]);

            CUDA_CHECK(cudaMalloc(&node_begin_[idx], capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&node_end_[idx], capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&node_y_[idx], capacity * sizeof(int)));

            slices_[idx].d_begin = node_begin_[idx];
            slices_[idx].d_end = node_end_[idx];
            slices_[idx].d_y_idx = node_y_[idx];
            slices_[idx].capacity = capacity;

            if (plan[idx].type == SurfaceOpType::Intersection) {
                CUDA_CHECK(cudaMalloc(&node_a_idx_[idx], capacity * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&node_b_idx_[idx], capacity * sizeof(int)));
                slices_[idx].d_a_idx = node_a_idx_[idx];
                slices_[idx].d_b_idx = node_b_idx_[idx];
            }
        }

        return cudaSuccess;
    }

} // namespace subsetix
