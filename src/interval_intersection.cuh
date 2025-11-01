#ifndef INTERVAL_INTERSECTION_CUH
#define INTERVAL_INTERSECTION_CUH

#include <cuda_runtime.h>

cudaError_t findIntervalIntersections(
    // Inputs
    const int* d_a_begin,
    const int* d_a_end,
    int a_interval_count,
    const int* d_a_y_offsets,
    int a_y_count,
    const int* d_b_begin,
    const int* d_b_end,
    int b_interval_count,
    const int* d_b_y_offsets,
    int b_y_count,
    // Outputs
    int** d_r_y_idx,
    int** d_r_begin,
    int** d_r_end,
    int** d_a_idx,
    int** d_b_idx,
    int* total_intersections_count);

cudaError_t findVolumeIntersections(
    // Inputs for set A
    const int* d_a_begin,
    const int* d_a_end,
    int a_interval_count,
    const int* d_a_row_offsets,
    const int* d_a_row_to_y, // may be nullptr -> defaults to row index
    const int* d_a_row_to_z, // may be nullptr -> defaults to 0
    int a_row_count,
    // Inputs for set B
    const int* d_b_begin,
    const int* d_b_end,
    int b_interval_count,
    const int* d_b_row_offsets,
    int b_row_count,
    // Outputs (allocated by the function, caller must free)
    int** d_r_z_idx,
    int** d_r_y_idx,
    int** d_r_begin,
    int** d_r_end,
    int** d_a_idx,
    int** d_b_idx,
    int* total_intersections_count);

void freeIntervalResults(int* d_r_y_idx,
                         int* d_r_begin,
                         int* d_r_end,
                         int* d_a_idx,
                         int* d_b_idx);

void freeVolumeIntersectionResults(int* d_r_z_idx,
                                   int* d_r_y_idx,
                                   int* d_r_begin,
                                   int* d_r_end,
                                   int* d_a_idx,
                                   int* d_b_idx);


#endif // INTERVAL_INTERSECTION_CUH
