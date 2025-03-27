// File: interval_intersection.cuh
#ifndef INTERVAL_INTERSECTION_CUH
#define INTERVAL_INTERSECTION_CUH

#include <cuda_runtime.h> // For cudaError_t

/**
 * @brief Finds intersections between two sets of intervals (A and B) on the GPU.
 *
 * Assumes B intervals (d_b_begin, d_b_end) are sorted by their begin points.
 * This function allocates device memory for the results (*d_r_begin, *d_r_end,
 * *d_a_idx, *d_b_idx). The caller is responsible for freeing this memory
 * using cudaFree() after use.
 *
 * @param d_a_begin Pointer to device memory containing the start points of intervals A.
 * @param d_a_end   Pointer to device memory containing the end points of intervals A.
 * @param a_size    The number of intervals in A.
 * @param d_b_begin Pointer to device memory containing the start points of intervals B (sorted).
 * @param d_b_end   Pointer to device memory containing the end points of intervals B (sorted).
 * @param b_size    The number of intervals in B.
 * @param d_r_begin Output: Pointer to a pointer for the allocated device memory storing the start points of intersection intervals.
 * @param d_r_end   Output: Pointer to a pointer for the allocated device memory storing the end points of intersection intervals.
 * @param d_a_idx   Output: Pointer to a pointer for the allocated device memory storing the original index from A for each intersection.
 * @param d_b_idx   Output: Pointer to a pointer for the allocated device memory storing the original index from B for each intersection.
 * @param total_intersections_count Output: Pointer to an integer where the total number of intersections found will be stored.
 *
 * @return cudaSuccess on success, or a CUDA error code on failure.
 */
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


/**
 * @brief Helper function to free the memory allocated by findIntervalIntersections.
 *
 * Provides a convenient way to free all four result arrays. Handles null pointers.
 *
 * @param d_r_begin Pointer to device memory for intersection start points (may be null).
 * @param d_r_end   Pointer to device memory for intersection end points (may be null).
 * @param d_a_idx   Pointer to device memory for original A indices (may be null).
 * @param d_b_idx   Pointer to device memory for original B indices (may be null).
 */
void freeIntersectionResults(int* d_r_begin, int* d_r_end, int* d_a_idx, int* d_b_idx);


#endif // INTERVAL_INTERSECTION_CUH
