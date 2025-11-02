#ifndef SUBSETIX_OPERATION_CHAIN_CUH
#define SUBSETIX_OPERATION_CHAIN_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace subsetix
{
    enum class SurfaceOpType
    {
        Difference,
        Intersection,
        Union
    };

    struct SurfaceView
    {
        const int* d_begin = nullptr;
        const int* d_end = nullptr;
        const int* d_row_offsets = nullptr;
        int row_count = 0;
    };

    struct SurfaceResultView
    {
        int* d_begin = nullptr;
        int* d_end = nullptr;
        int* d_y_idx = nullptr;
        int* d_z_idx = nullptr;
        int* d_a_idx = nullptr;
        int* d_b_idx = nullptr;
    };

    struct SurfaceHandle
    {
        int index;
        bool is_node;

        SurfaceHandle() : index(-1), is_node(false) {}
        SurfaceHandle(int idx, bool isNode) : index(idx), is_node(isNode) {}

        bool valid() const { return index >= 0; }
    };

    class SurfaceOperationChain
    {
    public:
        struct Node
        {
            SurfaceOpType type;
            SurfaceHandle lhs;
            SurfaceHandle rhs;

            Node() : type(SurfaceOpType::Intersection), lhs(), rhs() {}
            Node(SurfaceOpType operation,
                 SurfaceHandle left,
                 SurfaceHandle right)
                : type(operation)
                , lhs(left)
                , rhs(right)
            {}
        };

        SurfaceOperationChain() = default;

        SurfaceHandle add_input(const SurfaceView& view);

        SurfaceHandle add_operation(SurfaceOpType type,
                                    SurfaceHandle lhs,
                                    SurfaceHandle rhs);

        int row_count() const { return row_count_; }

        const std::vector<SurfaceView>& inputs() const { return inputs_; }

        const std::vector<Node>& nodes() const { return nodes_; }

        const SurfaceView& input(SurfaceHandle handle) const;

        const Node& node(SurfaceHandle handle) const;

        bool empty() const { return nodes_.empty(); }

        void clear();

    private:
        int row_count_ = -1;
        std::vector<SurfaceView> inputs_;
        std::vector<Node> nodes_;

        static void ensure_valid_input_handle(const SurfaceHandle& handle, size_t input_count);
        void ensure_valid_node_handle(const SurfaceHandle& handle) const;
        void ensure_same_row_count(int candidate_row_count);
    };

} // namespace subsetix

#endif // SUBSETIX_OPERATION_CHAIN_CUH
