#include <stdio.h>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>

#include "interval_intersection.cuh"
#include "cuda_utils.cuh"   

int main() {
    // You can modify this value !
    int n = 10000000 ;  
    size_t size_bytes = (size_t)n * sizeof(int);

    printf("Simplified Example: N = %d\n", n);

    // Host side
    // there are two sets, a and b.
    // we store begin and end in a SOA way. 
    std::vector<int> h_a_begin(n);
    std::vector<int> h_a_end(n);
    std::vector<int> h_b_begin(n);
    std::vector<int> h_b_end(n);

    printf("Initializing host data...\n");

    // Here, interval sets are defined to be like this : 
    // a : [----)....[----)....
    // b : .[----)....[----)...
    // i : .[---).....[---)....
    for (int i = 0; i < n; ++i) {
        h_a_begin[i] = 4 * i;
        h_a_end[i]   = 4 * i + 2;
        h_b_begin[i] = 4 * i + 1;
        h_b_end[i]   = 4 * i + 3;
    }

    // Device side
    printf("Allocating and copying inputs to device...\n");
    int *d_a_begin = nullptr; 
    int *d_a_end = nullptr;
    int *d_b_begin = nullptr; 
    int *d_b_end = nullptr;

    // Allocate sets in gpu
    cudaMalloc(&d_a_begin, size_bytes);
    cudaMalloc(&d_a_end,   size_bytes);
    cudaMalloc(&d_b_begin, size_bytes);
    cudaMalloc(&d_b_end,   size_bytes);

    // Copy Host to device
    cudaMemcpy(d_a_begin, h_a_begin.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end,   h_a_end.data(),   size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end,   h_b_end.data(),   size_bytes, cudaMemcpyHostToDevice);

    // Intersection arrays, device side
    printf("Calling library function...\n");
    int *d_r_begin = nullptr;
    int  *d_r_end = nullptr;
    int   *d_a_idx = nullptr;
    int *d_b_idx = nullptr;
    int  total_intersections = 0;

    // Events are for for timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    //cudaError_t err = 
    findIntervalIntersections(
        d_a_begin, d_a_end, n,
        d_b_begin, d_b_end, n,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections
    );
//    CUDA_CHECK(err); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken by findIntervalIntersections: %.5f ms\n", milliseconds);
    printf("Time taken by findIntervalIntersections per n: %.5f ns\n", 1000*1000*milliseconds/n);

    printf("Total intersections found: %d\n", total_intersections);

    // Copy result to host
    if (total_intersections > 0) {
        printf("Copying results Device -> Host...\n");
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int>   h_a_idx(total_intersections);
        std::vector<int>   h_b_idx(total_intersections);

        size_t results_size_bytes = (size_t)total_intersections * sizeof(int);
        size_t indices_size_bytes = (size_t)total_intersections * sizeof(int);

        cudaMemcpy(h_r_begin.data(), d_r_begin, results_size_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_end.data(),   d_r_end,   results_size_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_a_idx.data(),   d_a_idx,   indices_size_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_idx.data(),   d_b_idx,   indices_size_bytes, cudaMemcpyDeviceToHost);
        // Display first results
        int display_count = (total_intersections < 5) ? total_intersections : 5;
        printf("First %d intersections:\n", display_count);
        for (int i = 0; i < display_count; ++i) {
             printf(" -- Intersection %d: A[%d] intersects B[%d] -> Result: [%.1d, %.1d)\n",
                   i, h_a_idx[i], h_b_idx[i], h_r_begin[i], h_r_end[i]);
        }
    }

    // Free
    printf("Freeing arrays...\n");
    freeIntersectionResults(d_r_begin, d_r_end, d_a_idx, d_b_idx);
    cudaFree(d_a_begin);                                
    cudaFree(d_a_end);
    cudaFree(d_b_begin);
    cudaFree(d_b_end);

//    return EXIT_SUCCESS;
}
