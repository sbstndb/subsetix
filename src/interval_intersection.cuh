#ifndef INTERVAL_INTERSECTION_CUH
#define INTERVAL_INTERSECTION_CUH

#include <cuda_runtime.h>

cudaError_t enqueueIntervalIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueIntervalIntersectionWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx,
    cudaStream_t stream = nullptr);

cudaError_t computeIntervalIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    int* total_intersections_count,
    cudaStream_t stream = nullptr);

cudaError_t writeIntervalIntersectionsWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx,
    cudaStream_t stream = nullptr);

cudaError_t enqueueVolumeIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueVolumeIntersectionWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y,
    const int* d_a_row_to_z,
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx,
    cudaStream_t stream = nullptr);

cudaError_t computeVolumeIntersectionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    int* total_intersections_count,
    cudaStream_t stream = nullptr);

cudaError_t writeVolumeIntersectionsWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y, // may be nullptr -> defaults to row index
    const int* d_a_row_to_z, // may be nullptr -> defaults to 0
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    int* d_a_idx,
    int* d_b_idx,
    cudaStream_t stream = nullptr);

cudaError_t enqueueIntervalUnionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueIntervalUnionWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t computeIntervalUnionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    int* total_intervals_count,
    cudaStream_t stream = nullptr);

cudaError_t writeIntervalUnionWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t enqueueIntervalDifferenceOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueIntervalDifferenceWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t computeIntervalDifferenceOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    int* d_counts,
    int* d_offsets,
    int* total_intervals_count,
    cudaStream_t stream = nullptr);

cudaError_t writeIntervalDifferenceWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_y_offsets,
    int b_y_count,
    const int* d_offsets,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t enqueueVolumeUnionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueVolumeUnionWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y,
    const int* d_a_row_to_z,
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t computeVolumeUnionOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    int* total_intervals_count,
    cudaStream_t stream = nullptr);

cudaError_t writeVolumeUnionWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y,
    const int* d_a_row_to_z,
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t enqueueVolumeDifferenceOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    cudaStream_t stream = nullptr,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0);

cudaError_t enqueueVolumeDifferenceWrite(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y,
    const int* d_a_row_to_z,
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);

cudaError_t computeVolumeDifferenceOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    int* d_counts,
    int* d_offsets,
    int* total_intervals_count,
    cudaStream_t stream = nullptr);

cudaError_t writeVolumeDifferenceWithOffsets(
    const int* d_a_begin,
    const int* d_a_end,
    const int* d_a_row_offsets,
    int a_row_count,
    const int* d_b_begin,
    const int* d_b_end,
    const int* d_b_row_offsets,
    int b_row_count,
    const int* d_a_row_to_y,
    const int* d_a_row_to_z,
    const int* d_offsets,
    int* d_r_z_idx,
    int* d_r_y_idx,
    int* d_r_begin,
    int* d_r_end,
    cudaStream_t stream = nullptr);


struct IntervalIntersectionGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
};

struct IntervalIntersectionOffsetsGraphConfig {
    const int* d_a_begin = nullptr;
    const int* d_a_end = nullptr;
    const int* d_a_y_offsets = nullptr;
    int a_y_count = 0;
    const int* d_b_begin = nullptr;
    const int* d_b_end = nullptr;
    const int* d_b_y_offsets = nullptr;
    int b_y_count = 0;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    int* d_total = nullptr; // optional device pointer storing total intersections
};

struct IntervalIntersectionWriteGraphConfig {
    const int* d_a_begin = nullptr;
    const int* d_a_end = nullptr;
    const int* d_a_y_offsets = nullptr;
    int a_y_count = 0;
    const int* d_b_begin = nullptr;
    const int* d_b_end = nullptr;
    const int* d_b_y_offsets = nullptr;
    int b_y_count = 0;
    const int* d_offsets = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_capacity = 0; // number of entries the outputs can hold
};

cudaError_t createIntervalIntersectionOffsetsGraph(IntervalIntersectionGraph* graph,
                                                   const IntervalIntersectionOffsetsGraphConfig& config,
                                                   cudaStream_t stream = nullptr);
cudaError_t createIntervalIntersectionWriteGraph(IntervalIntersectionGraph* graph,
                                                 const IntervalIntersectionWriteGraphConfig& config,
                                                 cudaStream_t stream = nullptr);
cudaError_t launchIntervalIntersectionGraph(const IntervalIntersectionGraph& graph,
                                            cudaStream_t stream = nullptr);
void destroyIntervalIntersectionGraph(IntervalIntersectionGraph* graph);


struct VolumeIntersectionGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
};

struct VolumeIntersectionOffsetsGraphConfig {
    const int* d_a_begin = nullptr;
    const int* d_a_end = nullptr;
    const int* d_a_row_offsets = nullptr;
    int a_row_count = 0;
    const int* d_b_begin = nullptr;
    const int* d_b_end = nullptr;
    const int* d_b_row_offsets = nullptr;
    int b_row_count = 0;
    int* d_counts = nullptr;
    int* d_offsets = nullptr;
    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    int* d_total = nullptr; // optional device pointer storing total intersections
};

struct VolumeIntersectionWriteGraphConfig {
    const int* d_a_begin = nullptr;
    const int* d_a_end = nullptr;
    const int* d_a_row_offsets = nullptr;
    int a_row_count = 0;
    const int* d_a_row_to_y = nullptr;
    const int* d_a_row_to_z = nullptr;
    const int* d_b_begin = nullptr;
    const int* d_b_end = nullptr;
    const int* d_b_row_offsets = nullptr;
    int b_row_count = 0;
    const int* d_offsets = nullptr;
    int* d_r_z_idx = nullptr;
    int* d_r_y_idx = nullptr;
    int* d_r_begin = nullptr;
    int* d_r_end = nullptr;
    int* d_a_idx = nullptr;
    int* d_b_idx = nullptr;
    int total_capacity = 0;
};

cudaError_t createVolumeIntersectionOffsetsGraph(VolumeIntersectionGraph* graph,
                                                 const VolumeIntersectionOffsetsGraphConfig& config,
                                                 cudaStream_t stream = nullptr);
cudaError_t createVolumeIntersectionWriteGraph(VolumeIntersectionGraph* graph,
                                               const VolumeIntersectionWriteGraphConfig& config,
                                               cudaStream_t stream = nullptr);
cudaError_t launchVolumeIntersectionGraph(const VolumeIntersectionGraph& graph,
                                          cudaStream_t stream = nullptr);
void destroyVolumeIntersectionGraph(VolumeIntersectionGraph* graph);


#endif // INTERVAL_INTERSECTION_CUH
