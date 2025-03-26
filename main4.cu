#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Structure pour représenter un point (début ou fin d'intervalle)
typedef struct {
    float value;
    int is_start; // 1 si début, 0 si fin
    int set;      // 0 pour a, 1 pour b
    int index;    // Indice dans l'ensemble original
} Event;

// Comparateur pour trier les événements
int compare_events(const void* a, const void* b) {
    Event* ea = (Event*)a;
    Event* eb = (Event*)b;
    if (ea->value != eb->value) return (ea->value > eb->value) ? 1 : -1;
    return (ea->is_start < eb->is_start) ? 1 : -1; // Débuts avant fins si égalité
}

// Noyau CUDA pour calculer les intersections dans un lot
__global__ void intersection_lot(
    float* d_a_begin, float* d_a_end, int* d_a_active, int a_count,
    float* d_b_begin, float* d_b_end, int* d_b_active, int b_count,
    float* d_r_begin, float* d_r_end, int* d_flags, int* d_result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a_count * b_count) return;

    int i = idx / b_count; // Indice dans a
    int j = idx % b_count; // Indice dans b

    if (d_a_active[i] && d_b_active[j]) {
        float a_begin = d_a_begin[i];
        float a_end = d_a_end[i];
        float b_begin = d_b_begin[j];
        float b_end = d_b_end[j];

        int overlaps = (b_begin < a_end) && (b_end > a_begin);
        if (overlaps) {
            int pos = atomicAdd(d_result_count, 1); // Incrément atomique pour position unique
            d_flags[pos] = 1;
            d_r_begin[pos] = max(a_begin, b_begin);
            d_r_end[pos] = min(a_end, b_end);
        }
    }
}

int main() {
    int n;
    printf("Entrez la taille des ensembles d'intervalles (n) : ");
    scanf("%d", &n);
    if (n <= 0) {
        printf("Erreur : n doit être positif.\n");
        return -1;
    }

    size_t size = n * sizeof(float);
    size_t result_size = n * n * sizeof(float);

    // Allocation sur l'hôte
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end = (float*)malloc(size);
    float *h_r_begin = (float*)malloc(result_size);
    float *h_r_end = (float*)malloc(result_size);
    int *h_flags = (int*)malloc(n * n * sizeof(int));
    Event *events = (Event*)malloc(4 * n * sizeof(Event)); // Correction : 4*n, pas 2*n

    if (!h_a_begin || !h_a_end || !h_b_begin || !h_b_end || !h_r_begin || !h_r_end || !h_flags || !events) {
        printf("Erreur d'allocation mémoire sur l'hôte.\n");
        free(h_a_begin); free(h_a_end); free(h_b_begin); free(h_b_end);
        free(h_r_begin); free(h_r_end); free(h_flags); free(events);
        return -1;
    }

    // Initialisation des intervalles avec chevauchements
    float step = 2.0f;
    for (int i = 0; i < n; i++) {
        h_a_begin[i] = i * step;
        h_a_end[i] = i * step + 2.5f;
        h_b_begin[i] = i * step + 0.7f;
        h_b_end[i] = i * step + 2.7f;
    }

    // Création des événements pour le tri
    for (int i = 0; i < n; i++) {
        events[2 * i] = {h_a_begin[i], 1, 0, i};         // Début de a
        events[2 * i + 1] = {h_a_end[i], 0, 0, i};      // Fin de a
        events[2 * n + 2 * i] = {h_b_begin[i], 1, 1, i}; // Début de b
        events[2 * n + 2 * i + 1] = {h_b_end[i], 0, 1, i}; // Fin de b
    }
    qsort(events, 4 * n, sizeof(Event), compare_events);

    // Allocation sur le device
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end, *d_r_begin, *d_r_end;
    int *d_a_active, *d_b_active, *d_flags, *d_result_count;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end, size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end, size);
    cudaMalloc(&d_a_active, n * sizeof(int));
    cudaMalloc(&d_b_active, n * sizeof(int));
    cudaMalloc(&d_r_begin, result_size);
    cudaMalloc(&d_r_end, result_size);
    cudaMalloc(&d_flags, n * n * sizeof(int));
    cudaMalloc(&d_result_count, sizeof(int));

    if (!d_a_begin || !d_a_end || !d_b_begin || !d_b_end || !d_a_active || !d_b_active ||
        !d_r_begin || !d_r_end || !d_flags || !d_result_count) {
        printf("Erreur d'allocation mémoire sur le GPU.\n");
        cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
        cudaFree(d_a_active); cudaFree(d_b_active); cudaFree(d_r_begin); cudaFree(d_r_end);
        cudaFree(d_flags); cudaFree(d_result_count);
        return -1;
    }

    // Copie initiale des données
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, size, cudaMemcpyHostToDevice);
    cudaMemset(d_result_count, 0, sizeof(int));

    // Traitement par balayage
    int* active_a = (int*)calloc(n, sizeof(int));
    int* active_b = (int*)calloc(n, sizeof(int));
    if (!active_a || !active_b) {
        printf("Erreur d'allocation mémoire pour active_a ou active_b.\n");
        free(active_a); free(active_b);
        return -1;
    }
    int a_count = 0, b_count = 0;
    int h_result_count = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;

    for (int i = 0; i < 4 * n; i++) {
        Event e = events[i];
        if (e.set == 0) { // Ensemble a
            if (e.is_start) {
                active_a[e.index] = 1;
                a_count++;
            } else {
                active_a[e.index] = 0;
                a_count--;
            }
        } else { // Ensemble b
            if (e.is_start) {
                active_b[e.index] = 1;
                b_count++;
            } else {
                active_b[e.index] = 0;
                b_count--;
            }
        }

        // Lancer un noyau à chaque changement significatif
        if (a_count > 0 && b_count > 0 && (i == 4 * n - 1 || events[i + 1].value != e.value)) {
            cudaMemcpy(d_a_active, active_a, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_active, active_b, n * sizeof(int), cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (a_count * b_count + threadsPerBlock - 1) / threadsPerBlock;

            cudaEventRecord(start);
            intersection_lot<<<blocksPerGrid, threadsPerBlock>>>(
                d_a_begin, d_a_end, d_a_active, a_count,
                d_b_begin, d_b_end, d_b_active, b_count,
                d_r_begin, d_r_end, d_flags, d_result_count
            );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;
        }
    }

    printf("Temps d'exécution total des noyaux pour n=%d : %.3f ms\n", n, total_time);

    // Récupération des résultats
    cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_begin, d_r_begin, h_result_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end, d_r_end, h_result_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flags, d_flags, h_result_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Affichage
    printf("Intervalles de a :\n");
    for (int i = 0; i < n; i++) printf("[%.1f, %.1f[ ", h_a_begin[i], h_a_end[i]);
    printf("\nIntervalles de b :\n");
    for (int i = 0; i < n; i++) printf("[%.1f, %.1f[ ", h_b_begin[i], h_b_end[i]);
    printf("\nIntersections (%d trouvées) :\n", h_result_count);
    for (int i = 0; i < h_result_count; i++) {
        if (h_flags[i]) printf("[%.1f, %.1f[ ", h_r_begin[i], h_r_end[i]);
    }
    printf("\n");

    // Libération mémoire
    free(h_a_begin); free(h_a_end); free(h_b_begin); free(h_b_end);
    free(h_r_begin); free(h_r_end); free(h_flags); free(events);
    free(active_a); free(active_b);
    cudaFree(d_a_begin); cudaFree(d_a_end); cudaFree(d_b_begin); cudaFree(d_b_end);
    cudaFree(d_a_active); cudaFree(d_b_active); cudaFree(d_r_begin); cudaFree(d_r_end);
    cudaFree(d_flags); cudaFree(d_result_count);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
