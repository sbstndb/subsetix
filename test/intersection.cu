// Définition du kernel (extrait de votre code, avec dépendances)
__device__ float max_device(float a, float b) { return (a > b) ? a : b; }
__device__ float min_device(float a, float b) { return (a < b) ? a : b; }
__device__ int lower_bound_end(float* B_end, int n, float value) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (B_end[mid] <= value) left = mid + 1;
        else right = mid;
    }
    return left;
}
__device__ int lower_bound_begin(float* B_begin, int n, float value) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (B_begin[mid] < value) left = mid + 1;
        else right = mid;
    }
    return left;
}

__global__ void intersection_atomic(
    float* d_a_begin, float* d_a_end, int a_size,
    float* d_b_begin, float* d_b_end, int b_size,
    float* d_r_begin, float* d_r_end,
    int* d_a_idx, int* d_b_idx,
    int* d_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];
        int j_min = lower_bound_end(d_b_end, b_size, a_begin);
        int j_max = lower_bound_begin(d_b_begin, b_size, a_end);
        for (int j = j_min; j < j_max && j < b_size; j++) {
            float b_begin = d_b_begin[j];
            float b_end = d_b_end[j];
            float inter_begin = max_device(a_begin, b_begin);
            float inter_end = min_device(a_end, b_end);
            int pos = atomicAdd(d_counter, 1);
            d_r_begin[pos] = inter_begin;
            d_r_end[pos] = inter_end;
            d_a_idx[pos] = i;
            d_b_idx[pos] = j;
        }
    }
}

// Test Google Test
TEST(IntersectionAtomicTest, ComputesIntersectionsCorrectly) {
    // Petits ensembles de données
    const int a_size = 2;
    const int b_size = 2;
    float h_a_begin[a_size] = {0.0f, 2.0f};
    float h_a_end[a_size] = {2.0f, 4.0f};
    float h_b_begin[b_size] = {1.0f, 3.0f};
    float h_b_end[b_size] = {3.0f, 5.0f};

    // Allocation sur le device
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end;
    cudaMalloc(&d_a_begin, a_size * sizeof(float));
    cudaMalloc(&d_a_end, a_size * sizeof(float));
    cudaMalloc(&d_b_begin, b_size * sizeof(float));
    cudaMalloc(&d_b_end, b_size * sizeof(float));

    // Copier les données vers le device
    cudaMemcpy(d_a_begin, h_a_begin, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, b_size * sizeof(float), cudaMemcpyHostToDevice);

    // Allocation pour les résultats
    int max_intersections = a_size * b_size; // 4 au maximum
    float *d_r_begin, *d_r_end;
    int *d_a_idx, *d_b_idx, *d_counter;
    cudaMalloc(&d_r_begin, max_intersections * sizeof(float));
    cudaMalloc(&d_r_end, max_intersections * sizeof(float));
    cudaMalloc(&d_a_idx, max_intersections * sizeof(int));
    cudaMalloc(&d_b_idx, max_intersections * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Lancer le kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (a_size + threadsPerBlock - 1) / threadsPerBlock;
    intersection_atomic<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end, a_size,
        d_b_begin, d_b_end, b_size,
        d_r_begin, d_r_end,
        d_a_idx, d_b_idx,
        d_counter
    );
    cudaDeviceSynchronize();

    // Récupérer le nombre d'intersections
    int h_counter;
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_counter, 2); // Attendu: 2 intersections

    // Récupérer les résultats
    float h_r_begin[max_intersections];
    float h_r_end[max_intersections];
    int h_a_idx[max_intersections];
    int h_b_idx[max_intersections];
    cudaMemcpy(h_r_begin, d_r_begin, h_counter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end, d_r_end, h_counter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_idx, d_a_idx, h_counter * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_idx, d_b_idx, h_counter * sizeof(int), cudaMemcpyDeviceToHost);

    // Vérifier les intersections
    // Intersection 1: A[0] et B[0] -> [1.0, 2.0[
    EXPECT_FLOAT_EQ(h_r_begin[0], 1.0f);
    EXPECT_FLOAT_EQ(h_r_end[0], 2.0f);
    EXPECT_EQ(h_a_idx[0], 0);
    EXPECT_EQ(h_b_idx[0], 0);

    // Intersection 2: A[1] et B[1] -> [3.0, 4.0[
    EXPECT_FLOAT_EQ(h_r_begin[1], 3.0f);
    EXPECT_FLOAT_EQ(h_r_end[1], 4.0f);
    EXPECT_EQ(h_a_idx[1], 1);
    EXPECT_EQ(h_b_idx[1], 1);

    // Libérer la mémoire
    cudaFree(d_a_begin); cudaFree(d_a_end);
    cudaFree(d_b_begin); cudaFree(d_b_end);
    cudaFree(d_r_begin); cudaFree(d_r_end);
    cudaFree(d_a_idx); cudaFree(d_b_idx);
    cudaFree(d_counter);
}
