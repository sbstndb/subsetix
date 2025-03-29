#ifndef INTERVAL_INTERSECTION_CUH
#define INTERVAL_INTERSECTION_CUH

#include <cuda_runtime.h> 
cudaError_t findIntervalIntersections(
    // Inputs
    const int* d_a_begin,
    const int* d_a_end,
    int a_size,
    const int* d_b_begin,
    const int* d_b_end,
    int b_size,
    // Outputs (allocated by the function, caller must free)
    int** d_r_begin,
    int** d_r_end,
    int** d_a_idx,
    int** d_b_idx,
    int* total_intersections_count
);

void freeIntersectionResults(int* d_r_begin, int* d_r_end, int* d_a_idx, int* d_b_idx);


#endif // INTERVAL_INTERSECTION_CUH
