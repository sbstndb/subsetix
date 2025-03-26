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



// Remplace les lambdas par des fonctions __device__
__device__ float max_device(float a, float b) {
    return (a > b) ? a : b;
}

__device__ float min_device(float a, float b) {
    return (a < b) ? a : b;
}

// Kernel utilisant un compteur atomique pour écrire les intersections
__global__ void intersection_atomic(
    float* d_a_begin, float* d_a_end, int a_size,
    float* d_b_begin, float* d_b_end, int b_size,
    float* d_r_begin, float* d_r_end,
    int* d_a_idx, int* d_b_idx, // Optionnel: pour stocker les indices des intervalles A et B
    int* d_counter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];

        // Utilisation des fonctions __device__
        int j_min = lower_bound_end(d_b_end, b_size, a_begin);
        int j_max = lower_bound_begin(d_b_begin, b_size, a_end);

        // Pour chaque intervalle B qui intersecte A[i]
        for (int j = j_min; j < j_max && j < b_size; j++) {
            float b_begin = d_b_begin[j];
            float b_end = d_b_end[j];

            // Calcul de l'intersection
            float inter_begin = max_device(a_begin, b_begin);
            float inter_end   = min_device(a_end, b_end);

            // Réserver une case dans le tableau résultat à l'aide d'une opération atomique
            int pos = atomicAdd(d_counter, 1);
            d_r_begin[pos] = inter_begin;
            d_r_end[pos] = inter_end;
            // (Optionnel) Sauvegarder les indices pour retrouver d'où vient l'intersection
            d_a_idx[pos] = i;
            d_b_idx[pos] = j;
        }
    }
}



int main() {
    int n = 10000;
    size_t size = n * sizeof(float);

    // Allocation mémoire pour les intervalles A et B sur l'hôte
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end   = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end   = (float*)malloc(size);

    // Initialisation des intervalles
    for (int i = 0; i < n; i++) {
        h_a_begin[i] = 4.0f * i;
        h_a_end[i]   = 4.0f * i + 2.0f;
        h_b_begin[i] = 4.0f * i + 1.0f - 4.0f;
        h_b_end[i]   = 4.0f * i + 3.0f - 4.0f;
    }

    // Estimation de la taille maximum du tableau résultat.
    // Dans le pire des cas, tous les A[i] intersectent plusieurs B[j].
    // On peut allouer un tableau de taille n*n pour être sûr.
    int max_intersections = n * n;
    float *h_r_begin = (float*)malloc(max_intersections * sizeof(float));
    float *h_r_end   = (float*)malloc(max_intersections * sizeof(float));
    int   *h_a_idx   = (int*)malloc(max_intersections * sizeof(int)); // Optionnel
    int   *h_b_idx   = (int*)malloc(max_intersections * sizeof(int)); // Optionnel

    // Allocation sur le device pour A et B
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end,   size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end,   size);

    // Allocation sur le device pour les résultats d'intersection
    float *d_r_begin, *d_r_end;
    int   *d_a_idx, *d_b_idx;  // Optionnel, pour sauvegarder les indices d'intersection
    cudaMalloc(&d_r_begin, max_intersections * sizeof(float));
    cudaMalloc(&d_r_end,   max_intersections * sizeof(float));
    cudaMalloc(&d_a_idx,   max_intersections * sizeof(int));
    cudaMalloc(&d_b_idx,   max_intersections * sizeof(int));

    // Allocation pour le compteur atomique
    int *d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Copier les intervalles de l'hôte vers le device
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end,   h_a_end,   size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end,   h_b_end,   size, cudaMemcpyHostToDevice);

    // Configuration du kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Enregistrement du temps de début
    cudaEventRecord(start, 0);
    // Exécution du kernel (une seule exécution ici, mais vous pouvez le lancer plusieurs fois)
    intersection_atomic<<<blocksPerGrid, threadsPerBlock>>>(
        d_a_begin, d_a_end, n,
        d_b_begin, d_b_end, n,
        d_r_begin, d_r_end,
        d_a_idx, d_b_idx,
        d_counter
    );
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Temps d'exécution du kernel: %.3f ms\n", elapsedTime);


    // Récupérer le nombre d'intersections calculées
    int h_counter;
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Nombre total d'intersections: %d\n", h_counter);

    // Copier les résultats du device vers l'hôte
    cudaMemcpy(h_r_begin, d_r_begin, h_counter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end,   d_r_end,   h_counter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_idx,   d_a_idx,   h_counter * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_idx,   d_b_idx,   h_counter * sizeof(int),   cudaMemcpyDeviceToHost);

    // (Optionnel) Affichage de quelques intersections
   
    for (int i = 0; i < h_counter; i++) {
        printf("Intersection %d: A[%d] et B[%d] -> [%.1f, %.1f[\n", 
               i, h_a_idx[i], h_b_idx[i], h_r_begin[i], h_r_end[i]);
    }

    // Libération des ressources
    free(h_a_begin); free(h_a_end); free(h_b_begin); free(h_b_end);
    free(h_r_begin); free(h_r_end); free(h_a_idx); free(h_b_idx);
    cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
    cudaFree(d_r_begin); cudaFree(d_r_end);
    cudaFree(d_a_idx); cudaFree(d_b_idx);
    cudaFree(d_counter);

    return 0;
}

