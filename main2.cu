#include <stdio.h>
#include <cuda_runtime.h>

// Structure pour un intervalle (facultatif, mais utile pour la clarté)
struct Interval {
    float begin;
    float end;
};

// Noyau CUDA pour calculer l'intersection d'intervalles
__global__ void intersection(
    float* d_a_begin, float* d_a_end, int a_size,
    float* d_b_begin, float* d_b_end, int b_size,
    float* d_r_begin, float* d_r_end, int* d_flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Indice dans a

    if (i >= a_size) return;

    float a_begin = d_a_begin[i];
    float a_end = d_a_end[i];
    int result_idx = i * b_size; // Chaque intervalle de a peut intersecter tous les b

    // Boucle sur les intervalles de b (pas idéal pour éviter les branchements, voir note)
    for (int j = 0; j < b_size; j++) {
        float b_begin = d_b_begin[j];
        float b_end = d_b_end[j];

        // Vérification de chevauchement
        int overlaps = (b_begin < a_end) && (b_end > a_begin);
        d_flags[result_idx + j] = overlaps;

        // Calcul de l'intersection si chevauchement
        if (overlaps) {
            d_r_begin[result_idx + j] = max(a_begin, b_begin);
            d_r_end[result_idx + j] = min(a_end, b_end);
        } else {
            d_r_begin[result_idx + j] = 0.0f; // Valeur par défaut si pas d'intersection
            d_r_end[result_idx + j] = 0.0f;
        }
    }
}

int main() {
    int n = 5;  // Taille des ensembles d'intervalles (réduite pour l'exemple)
    size_t size = n * sizeof(float);

    // Allocation des vecteurs sur l'hôte (CPU)
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end = (float*)malloc(size);
    float *h_r_begin = (float*)malloc(size * n); // Taille max = n * n intersections possibles
    float *h_r_end = (float*)malloc(size * n);
    int *h_flags = (int*)malloc(n * n * sizeof(int));

    // Initialisation des intervalles
    // a = {[0, 2[, [2, 4[, [4, 6[, [6, 8[, [8, 10[}
    // b = {[1, 3[, [3, 5[, [5, 7[, [7, 9[, [9, 11[}
    for (int i = 0; i < n; i++) {
        h_a_begin[i] = 4.0f * i;
        h_a_end[i] = 4.0f * i + 2.0f;
        h_b_begin[i] = 4.0f * i + 1.0f -4.0f;
        h_b_end[i] = 4.0f * i + 3.0f - 4.0f;
        for (int j = 0; j < n; j++) {
            h_r_begin[i * n + j] = 0.0f;
            h_r_end[i * n + j] = 0.0f;
            h_flags[i * n + j] = 0;
        }
    }

    // Allocation des vecteurs sur le device (GPU)
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end, *d_r_begin, *d_r_end;
    int *d_flags;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end, size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end, size);
    cudaMalloc(&d_r_begin, size * n); // Taille max pour toutes les intersections possibles
    cudaMalloc(&d_r_end, size * n);
    cudaMalloc(&d_flags, n * n * sizeof(int));

    // Copie des données de l'hôte vers le device
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_begin, h_r_begin, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_end, h_r_end, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Configuration et lancement du kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    intersection<<<blocksPerGrid, threadsPerBlock>>>(d_a_begin, d_a_end, n, d_b_begin, d_b_end, n, d_r_begin, d_r_end, d_flags);

    // Vérification des erreurs
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erreur CUDA: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copie des résultats du device vers l’hôte
    cudaMemcpy(h_r_begin, d_r_begin, size * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end, d_r_end, size * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flags, d_flags, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Affichage des résultats
    printf("Intervalles de a :\n");
    for (int i = 0; i < n; i++) {
        printf("[%.1f, %.1f[ ", h_a_begin[i], h_a_end[i]);
    }
    printf("\nIntervalles de b :\n");
    for (int i = 0; i < n; i++) {
        printf("[%.1f, %.1f[ ", h_b_begin[i], h_b_end[i]);
    }
    printf("\nIntersections :\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (h_flags[idx]) {
                printf("[%.1f, %.1f[ ", h_r_begin[idx], h_r_end[idx]);
            }
        }
    }
    printf("\n");

    // Libération de la mémoire
    free(h_a_begin);
    free(h_a_end);
    free(h_b_begin);
    free(h_b_end);
    free(h_r_begin);
    free(h_r_end);
    free(h_flags);
    cudaFree(d_a_begin);
    cudaFree(d_a_end);
    cudaFree(d_b_begin);
    cudaFree(d_b_end);
    cudaFree(d_r_begin);
    cudaFree(d_r_end);
    cudaFree(d_flags);

    return 0;
}
