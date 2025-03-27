#include <stdio.h>
#include <stdlib.h> // For malloc/free
#include <vector>   // For host-side dynamic arrays (optional)

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h> // For device execution policy

// --- Error Checking Macro ---
// Simplified version for brevity. A production version should be more robust.
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d - %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Launch Error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
     err = cudaDeviceSynchronize(); \
     if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Synchronization Error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// --- Device Functions (Unchanged) ---
__device__ int lower_bound_end(const float* B_end, int n, float value) {
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

__device__ int lower_bound_begin(const float* B_begin, int n, float value) {
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

__device__ float max_device(float a, float b) {
    return (a > b) ? a : b;
}

__device__ float min_device(float a, float b) {
    return (a < b) ? a : b;
}

// --- Kernel 1: Count Intersections per A interval ---
__global__ void intersection_count_kernel(
    const float* d_a_begin, const float* d_a_end, int a_size,
    const float* d_b_begin, const float* d_b_end, int b_size,
    int* d_per_thread_counts) // Output: count for each A[i]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];
        int local_count = 0;

        // ASSUMPTION: B_begin and B_end are sorted.
        int j_min = lower_bound_end(d_b_end, b_size, a_begin);
        int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

        // Iterate through potentially intersecting B intervals
        for (int j = j_min; j < j_max && j < b_size; j++) {
            // Read B interval bounds only if needed
            float b_begin = d_b_begin[j];
            // Optimization: Check if B starts after A ends (redundant due to j_max, but safe)
            // if (b_begin >= a_end) break; // Can't happen if lower_bound_begin is correct

            // Optimization: Check if B ends before A starts (redundant due to j_min, but safe)
            // float b_end = d_b_end[j]; // Only read if b_begin < a_end
            // if (b_end <= a_begin) continue; // Can't happen if lower_bound_end is correct

             // Calculate intersection only if bounds suggest overlap might exist
            float b_end = d_b_end[j]; // Read b_end here
            float inter_begin = max_device(a_begin, b_begin);
            float inter_end   = min_device(a_end, b_end);

            // Check if the intersection is valid (non-empty interval)
            if (inter_begin < inter_end) {
                local_count++;
            }
        }
        d_per_thread_counts[i] = local_count;
    }
}


// --- Kernel 2: Write Intersections using Offsets ---
__global__ void intersection_write_kernel(
    const float* d_a_begin, const float* d_a_end, int a_size,
    const float* d_b_begin, const float* d_b_end, int b_size,
    const int* d_output_offsets, // Input: starting offset for each A[i]
    float* d_r_begin, float* d_r_end, // Output: final results
    int* d_a_idx, int* d_b_idx)       // Output: final indices
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];
        int write_offset = d_output_offsets[i]; // Base offset for this thread's results
        int local_write_idx = 0;                // Counter for intersections found by *this* thread

        // ASSUMPTION: B_begin and B_end are sorted.
        int j_min = lower_bound_end(d_b_end, b_size, a_begin);
        int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

        for (int j = j_min; j < j_max && j < b_size; j++) {
            float b_begin = d_b_begin[j];
            float b_end = d_b_end[j];

            float inter_begin = max_device(a_begin, b_begin);
            float inter_end   = min_device(a_end, b_end);

            if (inter_begin < inter_end) {
                int pos = write_offset + local_write_idx; // Calculate global position
                d_r_begin[pos] = inter_begin;
                d_r_end[pos] = inter_end;
                d_a_idx[pos] = i;
                d_b_idx[pos] = j;
                local_write_idx++; // Increment local counter *after* writing
            }
        }
    }
}


int main() {
    int n = 10000000; // Increase n for more meaningful benchmark
    if (n <= 0) return 1; // Basic sanity check
    size_t size_bytes = (size_t)n * sizeof(float);
    size_t size_bytes_int = (size_t)n * sizeof(int);

    printf("Setting up data for N = %d\n", n);

    // --- Host Data Allocation ---
    // Using vectors for easier management, but raw malloc is also fine
    std::vector<float> h_a_begin(n);
    std::vector<float> h_a_end(n);
    std::vector<float> h_b_begin(n);
    std::vector<float> h_b_end(n);

    // --- Initialization (Same as before, ensures B is sorted) ---
    printf("Initializing host data...\n");
    for (int i = 0; i < n; ++i) {
        h_a_begin[i] = 4.0f * i;
        h_a_end[i]   = 4.0f * i + 2.0f;
//        h_b_begin[i] = 4.0f * i;
//        h_b_end[i]   = 4.0f * i + 2.0f;	
//        h_b_begin[i] = 4.0f * i + 1.0f - 4.0f; // = 4.0f * (i-1) + 1.0f
//        h_b_end[i]   = 4.0f * i + 3.0f - 4.0f; // = 4.0f * (i-1) + 3.0f
//        h_b_begin[i] = -4.0f * i + 1.0f - 4.0f; // = 4.0f * (i-1) + 1.0f
//        h_b_end[i]   = -4.0f * i + 3.0f - 4.0f; // = 4.0f * (i-1) + 3.0f

        h_b_begin[i] = 8.0f * i + 1.0f - 4.0f; // = 4.0f * (i-1) + 1.0f
        h_b_end[i]   = 8.0f * i + 5.0f - 4.0f; // = 4.0f * (i-1) + 3.0f


    }
    // B arrays are sorted by construction in this example.
    // If they weren't, you'd need a thrust::sort call here on B arrays.

    // --- Device Allocation (Inputs A and B) ---
    printf("Allocating device memory for inputs...\n");
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end;
    CHECK_CUDA(cudaMalloc(&d_a_begin, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_a_end,   size_bytes));
    CHECK_CUDA(cudaMalloc(&d_b_begin, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_b_end,   size_bytes));

    // --- Device Allocation (Intermediate Count/Offset Arrays) ---
    // Using Thrust vectors for convenience with thrust::scan
    printf("Allocating device memory for intermediate counts/offsets...\n");
    thrust::device_vector<int> d_per_thread_counts(n);
    thrust::device_vector<int> d_output_offsets(n);

    // --- Copy Inputs H->D ---
    printf("Copying inputs Host -> Device...\n");
    CHECK_CUDA(cudaMemcpy(d_a_begin, h_a_begin.data(), size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a_end,   h_a_end.data(),   size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_begin, h_b_begin.data(), size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_end,   h_b_end.data(),   size_bytes, cudaMemcpyHostToDevice));

    // --- Kernel Configuration ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // --- CUDA Events for Timing ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // === PASS 1: Count Intersections ===
    printf("Launching Count Kernel...\n");
    CHECK_CUDA(cudaEventRecord(start, 0));

    intersection_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end, n,
        d_b_begin, d_b_end, n,
        thrust::raw_pointer_cast(d_per_thread_counts.data())
    );
    CHECK_KERNEL(); // Check for kernel launch/sync errors after the kernel

    // === Prefix Sum (Scan) using Thrust ===
    printf("Performing Prefix Sum (Thrust Scan)...\n");
    thrust::exclusive_scan(thrust::device, // Execute on device
                           d_per_thread_counts.begin(),
                           d_per_thread_counts.end(),
                           d_output_offsets.begin());
    // Need to synchronize explicitly if timing scan separately or relying on its completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // === Determine Total Count and Allocate Final Output ===
    printf("Determining total count and allocating final output...\n");
    int h_total_count = 0;
    if (n > 0) {
        // Total count = last offset + last count
        int last_offset = d_output_offsets.back(); // Thrust handles D->H copy for .back()
        int last_count = d_per_thread_counts.back();
        h_total_count = last_offset + last_count;
    }
    printf("Total intersections found: %d\n", h_total_count);

    float *d_r_begin = nullptr, *d_r_end = nullptr;
    int   *d_a_idx = nullptr,   *d_b_idx = nullptr;
    std::vector<float> h_r_begin, h_r_end;
    std::vector<int>   h_a_idx,   h_b_idx;

    if (h_total_count > 0) {
        size_t results_size_bytes = (size_t)h_total_count * sizeof(float);
        size_t indices_size_bytes = (size_t)h_total_count * sizeof(int);

        printf("Allocating device memory for %d results...\n", h_total_count);
        CHECK_CUDA(cudaMalloc(&d_r_begin, results_size_bytes));
        CHECK_CUDA(cudaMalloc(&d_r_end,   results_size_bytes));
        CHECK_CUDA(cudaMalloc(&d_a_idx,   indices_size_bytes));
        CHECK_CUDA(cudaMalloc(&d_b_idx,   indices_size_bytes));

        // Allocate host memory for results
        try {
            h_r_begin.resize(h_total_count);
            h_r_end.resize(h_total_count);
            h_a_idx.resize(h_total_count);
            h_b_idx.resize(h_total_count);
        } catch (const std::bad_alloc& e) {
            fprintf(stderr, "Host memory allocation failed for results: %s\n", e.what());
            // Clean up already allocated GPU memory before exiting
            cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
            // thrust vectors handle their own memory
            cudaFree(d_r_begin); cudaFree(d_r_end); cudaFree(d_a_idx); cudaFree(d_b_idx);
            cudaEventDestroy(start); cudaEventDestroy(stop);
            exit(EXIT_FAILURE);
        }


        // === PASS 2: Write Intersections ===
        printf("Launching Write Kernel...\n");
        intersection_write_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_a_begin, d_a_end, n,
            d_b_begin, d_b_end, n,
            thrust::raw_pointer_cast(d_output_offsets.data()),
            d_r_begin, d_r_end,
            d_a_idx, d_b_idx
        );
        CHECK_KERNEL(); // Check for kernel launch/sync errors after the kernel

    } // End if (h_total_count > 0)

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop)); // Wait for all GPU work to finish

    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Total Kernel Execution Time (Count + Scan + Write): %.3f ms\n", elapsedTime);


    // === Copy Results D->H (only if intersections were found) ===
    if (h_total_count > 0) {
        printf("Copying results Device -> Host...\n");
        size_t results_size_bytes = (size_t)h_total_count * sizeof(float);
        size_t indices_size_bytes = (size_t)h_total_count * sizeof(int);
        CHECK_CUDA(cudaMemcpy(h_r_begin.data(), d_r_begin, results_size_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_r_end.data(),   d_r_end,   results_size_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_a_idx.data(),   d_a_idx,   indices_size_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_b_idx.data(),   d_b_idx,   indices_size_bytes, cudaMemcpyDeviceToHost));
    }

    /* // Optional: Display some results
    if (h_total_count > 0) {
        int display_count = (h_total_count < 20) ? h_total_count : 20;
        printf("Displaying first %d intersections:\n", display_count);
        for (int i = 0; i < display_count; ++i) {
            // Fetch original A/B for display (only possible if host data is still valid)
            int aidx = h_a_idx[i];
            int bidx = h_b_idx[i];
            printf("Intersection %d: A[%d] [%.1f, %.1f) intersects B[%d] [%.1f, %.1f) -> Result: [%.1f, %.1f)\n",
                   i, aidx, (aidx < n ? h_a_begin[aidx] : -1.0f), (aidx < n ? h_a_end[aidx] : -1.0f),
                   bidx, (bidx < n ? h_b_begin[bidx] : -1.0f), (bidx < n ? h_b_end[bidx] : -1.0f),
                   h_r_begin[i], h_r_end[i]);
        }
    }
    */

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    CHECK_CUDA(cudaFree(d_a_begin));
    CHECK_CUDA(cudaFree(d_a_end));
    CHECK_CUDA(cudaFree(d_b_begin));
    CHECK_CUDA(cudaFree(d_b_end));
    // Thrust vectors d_per_thread_counts, d_output_offsets are automatically freed when they go out of scope
    if (d_r_begin) CHECK_CUDA(cudaFree(d_r_begin));
    if (d_r_end)   CHECK_CUDA(cudaFree(d_r_end));
    if (d_a_idx)   CHECK_CUDA(cudaFree(d_a_idx));
    if (d_b_idx)   CHECK_CUDA(cudaFree(d_b_idx));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("Done.\n");
    return 0;
}
