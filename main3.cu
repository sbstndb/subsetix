#include <stdio.h>
#include <cuda_runtime.h>

__global__ void intersection_2d(
    float* d_a_begin, float* d_a_end, int a_size,
    float* d_b_begin, float* d_b_end, int b_size,
    float* d_r_begin, float* d_r_end, int* d_flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < a_size && j < b_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];
        float b_begin = d_b_begin[j];
        float b_end = d_b_end[j];

        int idx = i * b_size + j;
        int overlaps = (b_begin < a_end) && (b_end > a_begin);
        d_flags[idx] = overlaps;

        if (overlaps) {
            d_r_begin[idx] = max(a_begin, b_begin);
            d_r_end[idx] = min(a_end, b_end);
        } else {
            d_r_begin[idx] = 0.0f;
            d_r_end[idx] = 0.0f;
        }
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);

    // Allocation sur l'hôte
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end = (float*)malloc(size);
    float *h_r_begin = (float*)malloc(size * n);
    float *h_r_end = (float*)malloc(size * n);
    int *h_flags = (int*)malloc(n * n * sizeof(int));

    // Initialisation des intervalles
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

    // Allocation sur le device
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end, *d_r_begin, *d_r_end;
    int *d_flags;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end, size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end, size);
    cudaMalloc(&d_r_begin, size * n);
    cudaMalloc(&d_r_end, size * n);
    cudaMalloc(&d_flags, n * n * sizeof(int));

    // Copie des données vers le device
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_begin, h_r_begin, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_end, h_r_end, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Création des événements pour mesurer le temps
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configuration de la grille 2D
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Enregistrement du temps de début
    cudaEventRecord(start);

    // Lancement du noyau
    intersection_2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end, n,
        d_b_begin, d_b_end, n,
        d_r_begin, d_r_end, d_flags
    );

    // Enregistrement du temps de fin
    cudaEventRecord(stop);

    // Synchronisation pour attendre la fin du noyau
    cudaEventSynchronize(stop);

    // Calcul du temps écoulé
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Temps d'exécution du noyau : %.3f ms\n", milliseconds);

    // Vérification des erreurs
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erreur CUDA: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copie des résultats vers l'hôte
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

    // Libération des événements CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Libération de la mémoire
    free(h_a_begin); free(h_a_end); free(h_b_begin); free(h_b_end);
    free(h_r_begin); free(h_r_end); free(h_flags);
    cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
    cudaFree(d_r_begin); cudaFree(d_r_end); cudaFree(d_flags);

    return 0;
}
