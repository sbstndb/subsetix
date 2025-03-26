#include <stdio.h>
#include <cuda_runtime.h>

// Binary search to find the first j such that B_end[j] > value
__device__ int lower_bound_end(float* B_end, int n, float value) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (B_end[mid] <= value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Binary search to find the first j such that B_begin[j] >= value
__device__ int lower_bound_begin(float* B_begin, int n, float value) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (B_begin[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Optimized kernel: each thread processes one A[i] and finds its intersections with B
__global__ void intersection_optimized(
    float* d_a_begin, float* d_a_end, int a_size,
    float* d_b_begin, float* d_b_end, int b_size,
    float* d_r_begin, float* d_r_end, int* d_flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];

        // Define max and min as device functions
        auto max = [](float a, float b) { return (a > b) ? a : b; };
        auto min = [](float a, float b) { return (a < b) ? a : b; };

        // Find the intersection range using binary search
        int j_min = lower_bound_end(d_b_end, b_size, a_begin);
        int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

        // Compute intersections for j in [j_min, j_max[
        for (int j = j_min; j < j_max && j < b_size; j++) {
            float b_begin = d_b_begin[j];
            float b_end = d_b_end[j];
            int idx = i * b_size + j;
            d_flags[idx] = 1;
            d_r_begin[idx] = max(a_begin, b_begin);
            d_r_end[idx] = min(a_end, b_end);
        }
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);

    // Host memory allocation
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end = (float*)malloc(size);
    float *h_r_begin = (float*)malloc(size * n);
    float *h_r_end = (float*)malloc(size * n);
    int *h_flags = (int*)malloc(n * n * sizeof(int));

    // Initialize intervals
    for (int i = 0; i < n; i++) {
        h_a_begin[i] = 4.0f * i;
        h_a_end[i] = 4.0f * i + 2.0f;
        h_b_begin[i] = 4.0f * i + 1.0f - 4.0f;
        h_b_end[i] = 4.0f * i + 3.0f - 4.0f;
        for (int j = 0; j < n; j++) {
            h_r_begin[i * n + j] = 0.0f;
            h_r_end[i * n + j] = 0.0f;
            h_flags[i * n + j] = 0;
        }
    }

    // Device memory allocation
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end, *d_r_begin, *d_r_end;
    int *d_flags;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end, size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end, size);
    cudaMalloc(&d_r_begin, size * n);
    cudaMalloc(&d_r_end, size * n);
    cudaMalloc(&d_flags, n * n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_begin, h_r_begin, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_end, h_r_end, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optimized configuration: 1D grid with 256 threads per block
//    int threadsPerBlock = 256;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Measure execution time
    cudaEventRecord(start);
    int NUM_RUNS = 100 ; 
    for (int i = 0 ; i < NUM_RUNS ; i++){
	    intersection_optimized<<<blocksPerGrid, threadsPerBlock>>>(
	        d_a_begin, d_a_end, n,
	        d_b_begin, d_b_end, n,
	        d_r_begin, d_r_end, d_flags
	    );
	     cudaDeviceSynchronize();
	 }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Optimized kernel execution time: %.3f ms\n", milliseconds/NUM_RUNS);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(h_flags, d_flags, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_begin, d_r_begin, size * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end, d_r_end, size * n, cudaMemcpyDeviceToHost);
/**
    // Display intersections by batch (per A[i])
    printf("\nIntersections by batch (per A[i]):\n");
    for (int i = 0; i < n; i++) {
        int has_intersection = 0;
        // Check if A[i] has any intersections
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (h_flags[idx]) {
                has_intersection = 1;
                break;
            }
        }
        if (has_intersection) {
            printf("A[%d] = [%.1f, %.1f[ intersects with:\n", i, h_a_begin[i], h_a_end[i]);
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                if (h_flags[idx]) {
                    printf("  B[%d] = [%.1f, %.1f[ -> Intersection: [%.1f, %.1f[\n",
                           j, h_b_begin[j], h_b_end[j], h_r_begin[idx], h_r_end[idx]);
                }
            }
        }
    }

    // Display final intersections
    printf("\nFinal intersections:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (h_flags[idx]) {
                printf("[%.1f, %.1f[ (from A[%d] and B[%d])\n", 
                       h_r_begin[idx], h_r_end[idx], i, j);
            }
        }
    }

    // Count non-empty intersections for verification
    int intersection_count = 0;
    for (int i = 0; i < n * n; i++) {
        if (h_flags[i]) intersection_count++;
    }
    printf("\nNumber of non-empty intersections: %d\n", intersection_count);
**/
    // Free resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a_begin); free(h_a_end); free(h_b_begin); free(h_b_end);
    free(h_r_begin); free(h_r_end); free(h_flags);
    cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
    cudaFree(d_r_begin); cudaFree(d_r_end); cudaFree(d_flags);

    return 0;
}
