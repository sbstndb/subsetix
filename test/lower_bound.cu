
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// Définition de la fonction device (extrait de votre code)
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

// Kernel de test
__global__ void test_lower_bound_end(float* B_end, int n, float value, int* result) {
    *result = lower_bound_end(B_end, n, value);
}

// Test Google Test
TEST(LowerBoundEndTest, FindsCorrectIndex) {
    const int n = 5;
    float h_B_end[n] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f}; // Tableau trié
    float value = 4.0f; // Valeur cible
    int h_result;       // Résultat sur l'hôte
    float* d_B_end;     // Tableau sur le device
    int* d_result;      // Résultat sur le device

    // Allocation mémoire sur le device
    cudaMalloc(&d_B_end, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(int));

    // Copier le tableau vers le device
    cudaMemcpy(d_B_end, h_B_end, n * sizeof(float), cudaMemcpyHostToDevice);

    test_lower_bound_end<<<1, 1>>>(d_B_end, n, -1.0f, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_result, 0);


    test_lower_bound_end<<<1, 1>>>(d_B_end, n, 4.0f, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_result, 2);

    test_lower_bound_end<<<1, 1>>>(d_B_end, n, 5.0f, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_result, 3);

    test_lower_bound_end<<<1, 1>>>(d_B_end, n, 9.0f, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_result, 5);


    // Libérer la mémoire
    cudaFree(d_B_end);
    cudaFree(d_result);
}
