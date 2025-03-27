#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> 
#include <tuple>    

#include "interval_intersection.cuh" 
#include "cuda_utils.cuh"          

struct IntersectionResult {
    int r_begin;
    int r_end;
    int a_idx;
    int b_idx;

    bool operator==(const IntersectionResult& other) const {
        return (std::abs(r_begin - other.r_begin) == 0 &&
                std::abs(r_end - other.r_end) == 0 &&
                a_idx == other.a_idx &&
                b_idx == other.b_idx);
    }

    bool operator<(const IntersectionResult& other) const {
        return std::tie(a_idx, b_idx, r_begin, r_end) <
               std::tie(other.a_idx, other.b_idx, other.r_begin, other.r_end);
    }

    // Friend function for Google Test to print the struct nicely on failure
    friend void PrintTo(const IntersectionResult& r, std::ostream* os) {
        *os << "IntersectionResult(r_begin=" << r.r_begin << ", r_end=" << r.r_end
            << ", a_idx=" << r.a_idx << ", b_idx=" << r.b_idx << ")";
    }
};


// --- Simple Test Case ---

TEST(IntervalIntersectionSimpleTest, BasicOverlap) {
    cudaError_t err;

    // 1. Host Data Setup
    // A: [0, 2), [5, 7)
    std::vector<int> h_a_begin = {0, 5}; 
    std::vector<int> h_a_end   = {2, 7};
    // B: [1, 3), [6, 9)
    std::vector<int> h_b_begin = {1, 6}; 
    std::vector<int> h_b_end   = {3, 9};
    int a_size = h_a_begin.size();
    int b_size = h_b_begin.size();
    size_t a_bytes = a_size * sizeof(int);
    size_t b_bytes = b_size * sizeof(int);

    std::vector<IntersectionResult> expected = {
        {1, 2, 0, 0},
        {6, 7, 1, 1}
    };

    // 2. Device Allocation & Input Copy
    int *d_a_begin = nullptr;
    int *d_a_end = nullptr;
    int *d_b_begin = nullptr;
    int *d_b_end = nullptr;

    cudaMalloc(&d_a_begin, a_bytes);
    cudaMalloc(&d_a_end,   a_bytes);
    cudaMalloc(&d_b_begin, b_bytes);
    cudaMalloc(&d_b_end,   b_bytes);

    cudaMemcpy(d_a_begin, h_a_begin.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end,   h_a_end.data(),   a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end,   h_b_end.data(),   b_bytes, cudaMemcpyHostToDevice);

    int *d_r_begin = nullptr;
    int *d_r_end = nullptr;
    int   *d_a_idx = nullptr;
    int *d_b_idx = nullptr;
    int   total_intersections = 0;

    findIntervalIntersections(
        d_a_begin, d_a_end, a_size,
        d_b_begin, d_b_end, b_size,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections
    );

    ASSERT_EQ(total_intersections, expected.size()); // Check count first

    std::vector<IntersectionResult> actual;
    if (total_intersections > 0) { 
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        size_t results_bytes = total_intersections * sizeof(int);
        size_t indices_bytes = total_intersections * sizeof(int);
        cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_end.data(),   d_r_end,   results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_a_idx.data(),   d_a_idx,   indices_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_idx.data(),   d_b_idx,   indices_bytes, cudaMemcpyDeviceToHost);

        actual.resize(total_intersections);
        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        EXPECT_EQ(actual, expected);
    }


    freeIntersectionResults(d_r_begin, d_r_end, d_a_idx, d_b_idx);
    if(d_a_begin) cudaFree(d_a_begin);
    if(d_a_end)   cudaFree(d_a_end);
    if(d_b_begin) cudaFree(d_b_begin);
    if(d_b_end)   cudaFree(d_b_end);
}


TEST(IntervalIntersectionBigOverlap, BigOverlap) {
    cudaError_t err;

    // 1. Host Data Setup
    // A: [0, 2), [5, 7)
    std::vector<int> h_a_begin = {0};
    std::vector<int> h_a_end   = {10};
    // B: [1, 3), [6, 9)
    std::vector<int> h_b_begin = {-2, 1, 4, 11};
    std::vector<int> h_b_end   = {-1, 3, 5, 12};
    int a_size = h_a_begin.size();
    int b_size = h_b_begin.size();
    size_t a_bytes = a_size * sizeof(int);
    size_t b_bytes = b_size * sizeof(int);

    std::vector<IntersectionResult> expected = {
        {1, 3, 0, 1},
        {4, 5, 0, 2}
    };

    // 2. Device Allocation & Input Copy
    int *d_a_begin = nullptr;
    int *d_a_end = nullptr;
    int *d_b_begin = nullptr;
    int *d_b_end = nullptr;

    cudaMalloc(&d_a_begin, a_bytes);
    cudaMalloc(&d_a_end,   a_bytes);
    cudaMalloc(&d_b_begin, b_bytes);
    cudaMalloc(&d_b_end,   b_bytes);

    cudaMemcpy(d_a_begin, h_a_begin.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end,   h_a_end.data(),   a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end,   h_b_end.data(),   b_bytes, cudaMemcpyHostToDevice);

    int *d_r_begin = nullptr;
    int *d_r_end = nullptr;
    int   *d_a_idx = nullptr;
    int *d_b_idx = nullptr;
    int   total_intersections = 0;

    findIntervalIntersections(
        d_a_begin, d_a_end, a_size,
        d_b_begin, d_b_end, b_size,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections
    );

    ASSERT_EQ(total_intersections, expected.size()); // Check count first

    std::vector<IntersectionResult> actual;
    if (total_intersections > 0) {
        std::vector<int> h_r_begin(total_intersections);
        std::vector<int> h_r_end(total_intersections);
        std::vector<int> h_a_idx(total_intersections);
        std::vector<int> h_b_idx(total_intersections);

        size_t results_bytes = total_intersections * sizeof(int);
        size_t indices_bytes = total_intersections * sizeof(int);
        cudaMemcpy(h_r_begin.data(), d_r_begin, results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_r_end.data(),   d_r_end,   results_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_a_idx.data(),   d_a_idx,   indices_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b_idx.data(),   d_b_idx,   indices_bytes, cudaMemcpyDeviceToHost);

        actual.resize(total_intersections);
        for (int i = 0; i < total_intersections; ++i) {
            actual[i] = {h_r_begin[i], h_r_end[i], h_a_idx[i], h_b_idx[i]};
        }

        EXPECT_EQ(actual, expected);
    }


    freeIntersectionResults(d_r_begin, d_r_end, d_a_idx, d_b_idx);
    if(d_a_begin) cudaFree(d_a_begin);
    if(d_a_end)   cudaFree(d_a_end);
    if(d_b_begin) cudaFree(d_b_begin);
    if(d_b_end)   cudaFree(d_b_end);
}


// Add more TEST(...) cases here for empty inputs, no overlaps, etc. following the same pattern.

// Example: Test for no overlap
TEST(IntervalIntersectionSimpleTest, NoOverlap) {
     cudaError_t err;

    // 1. Host Data Setup
    std::vector<int> h_a_begin = {0, 5};
    std::vector<int> h_a_end   = {2, 7};
    std::vector<int> h_b_begin = {2, 8}; // No overlap with A
    std::vector<int> h_b_end   = {4, 10};
    int a_size = h_a_begin.size();
    int b_size = h_b_begin.size();
    size_t a_bytes = a_size * sizeof(int);
    size_t b_bytes = b_size * sizeof(int);

    // 2. Device Allocation & Input Copy
    int *d_a_begin = nullptr, *d_a_end = nullptr;
    int *d_b_begin = nullptr, *d_b_end = nullptr;

    err = cudaMalloc(&d_a_begin, a_bytes); ASSERT_EQ(err, cudaSuccess);
    err = cudaMalloc(&d_a_end,   a_bytes); ASSERT_EQ(err, cudaSuccess);
    err = cudaMalloc(&d_b_begin, b_bytes); ASSERT_EQ(err, cudaSuccess);
    err = cudaMalloc(&d_b_end,   b_bytes); ASSERT_EQ(err, cudaSuccess);
    err = cudaMemcpy(d_a_begin, h_a_begin.data(), a_bytes, cudaMemcpyHostToDevice); ASSERT_EQ(err, cudaSuccess);
    err = cudaMemcpy(d_a_end,   h_a_end.data(),   a_bytes, cudaMemcpyHostToDevice); ASSERT_EQ(err, cudaSuccess);
    err = cudaMemcpy(d_b_begin, h_b_begin.data(), b_bytes, cudaMemcpyHostToDevice); ASSERT_EQ(err, cudaSuccess);
    err = cudaMemcpy(d_b_end,   h_b_end.data(),   b_bytes, cudaMemcpyHostToDevice); ASSERT_EQ(err, cudaSuccess);

    // 3. Execute the Library Function
    int *d_r_begin = nullptr, *d_r_end = nullptr;
    int   *d_a_idx = nullptr,   *d_b_idx = nullptr;
    int   total_intersections = 0;

    err = findIntervalIntersections(
        d_a_begin, d_a_end, a_size,
        d_b_begin, d_b_end, b_size,
        &d_r_begin, &d_r_end,
        &d_a_idx, &d_b_idx,
        &total_intersections
    );
    EXPECT_EQ(err, cudaSuccess); // Check function return

    // 4. Verify Results
    EXPECT_EQ(total_intersections, 0);
    // Pointers d_r_begin etc should be nullptr if count is 0
    EXPECT_EQ(d_r_begin, nullptr);

    // 5. Cleanup
    freeIntersectionResults(d_r_begin, d_r_end, d_a_idx, d_b_idx);
    if(d_a_begin) cudaFree(d_a_begin);
    if(d_a_end)   cudaFree(d_a_end);
    if(d_b_begin) cudaFree(d_b_begin);
    if(d_b_end)   cudaFree(d_b_end);
}

// Remember to link with gtest_main or provide a main function:
// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();

