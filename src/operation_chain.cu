#include "operation_chain.cuh"

namespace subsetix
{
    namespace
    {
        void validate_surface_view(const SurfaceView& view)
        {
            if (!view.d_begin || !view.d_end || !view.d_row_offsets) {
                throw std::invalid_argument("SurfaceView requires non-null device pointers");
            }
            if (view.row_count < 0) {
                throw std::invalid_argument("SurfaceView row_count must be non-negative");
            }
        }

        int view_row_count(const SurfaceOperationChain& chain, const SurfaceHandle& handle)
        {
            if (handle.is_node) {
                const auto& node = chain.node(handle);
                const SurfaceHandle& any_operand = node.lhs.is_node ? node.lhs : node.rhs;
                return view_row_count(chain, any_operand);
            }
            return chain.input(handle).row_count;
        }
    } // namespace

    SurfaceHandle SurfaceOperationChain::add_input(const SurfaceView& view)
    {
        validate_surface_view(view);
        ensure_same_row_count(view.row_count);
        inputs_.push_back(view);
        return SurfaceHandle(static_cast<int>(inputs_.size() - 1), false);
    }

    SurfaceHandle SurfaceOperationChain::add_operation(SurfaceOpType type,
                                                       SurfaceHandle lhs,
                                                       SurfaceHandle rhs)
    {
        const bool lhs_is_node = lhs.is_node;
        const bool rhs_is_node = rhs.is_node;

        if (lhs_is_node) {
            ensure_valid_node_handle(lhs);
        } else {
            ensure_valid_input_handle(lhs, inputs_.size());
        }
        if (rhs_is_node) {
            ensure_valid_node_handle(rhs);
        } else {
            ensure_valid_input_handle(rhs, inputs_.size());
        }

        const int lhs_rows = lhs_is_node ? view_row_count(*this, lhs) : inputs_[lhs.index].row_count;
        const int rhs_rows = rhs_is_node ? view_row_count(*this, rhs) : inputs_[rhs.index].row_count;

        if (lhs_rows != rhs_rows) {
            throw std::invalid_argument("Surface operands must share identical row_count");
        }
        ensure_same_row_count(lhs_rows);

        nodes_.emplace_back(type, lhs, rhs);
        return SurfaceHandle(static_cast<int>(nodes_.size() - 1), true);
    }

    const SurfaceView& SurfaceOperationChain::input(SurfaceHandle handle) const
    {
        ensure_valid_input_handle(handle, inputs_.size());
        return inputs_[handle.index];
    }

    const SurfaceOperationChain::Node& SurfaceOperationChain::node(SurfaceHandle handle) const
    {
        ensure_valid_node_handle(handle);
        return nodes_[handle.index];
    }

    void SurfaceOperationChain::clear()
    {
        row_count_ = -1;
        inputs_.clear();
        nodes_.clear();
    }

    void SurfaceOperationChain::ensure_valid_input_handle(const SurfaceHandle& handle, size_t input_count)
    {
        if (handle.index < 0 || handle.is_node) {
            throw std::invalid_argument("SurfaceHandle does not reference an input");
        }
        if (static_cast<size_t>(handle.index) >= input_count) {
            throw std::out_of_range("SurfaceHandle input index exceeds registered inputs");
        }
    }

    void SurfaceOperationChain::ensure_valid_node_handle(const SurfaceHandle& handle) const
    {
        if (handle.index < 0 || !handle.is_node || handle.index >= static_cast<int>(nodes_.size())) {
            throw std::invalid_argument("SurfaceHandle does not reference an operation node");
        }
    }

    void SurfaceOperationChain::ensure_same_row_count(int candidate_row_count)
    {
        if (row_count_ < 0) {
            row_count_ = candidate_row_count;
            return;
        }
        if (candidate_row_count != row_count_) {
            throw std::invalid_argument("SurfaceOperationChain enforces consistent row_count");
        }
    }

} // namespace subsetix
