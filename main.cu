#include <stdio.h>
#include <cuda_runtime.h>

// Kernel CUDA pour additionner deux vecteurs
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


// Kernel cuda for algebra of sets
__global__ void intersection(int* d_a_begin, int* d_a_end, int* d_b_begin, int* d_b_end){
	int i = blockIdx.x * blockDim.x + threadIdx.x;	

}


int main() {
    int n = 1000;  // Taille des vecteurs
    size_t size = n * sizeof(float);

    // Allocation des vecteurs sur l'hôte (CPU)
    float *h_a_begin = (float*)malloc(size);
    float *h_a_end = (float*)malloc(size);
    float *h_b_begin = (float*)malloc(size);
    float *h_b_end = (float*)malloc(size);
    float *h_r_begin = (float*)malloc(size);
    float *h_r_end = (float*)malloc(size);

    // Initialisation set a
    for (int i = 0; i < n; i++) {
        h_a_begin[i] = 2*i;
        h_a_end[i] = 2*i+1;
	h_b_begin[i] = 2*i ; 
	h_b_end[i] = 2*i+1 ; 
	h_r_begin[i] = 0 ; 
	h_r_end[i] = 0 ; 
    }


    // Allocation des vecteurs sur le device (GPU)
    float *d_a_begin, *d_a_end, *d_b_begin, *d_b_end, *d_r_begin, *d_r_end;
    cudaMalloc(&d_a_begin, size);
    cudaMalloc(&d_a_end, size);
    cudaMalloc(&d_b_begin, size);
    cudaMalloc(&d_b_end, size);
    cudaMalloc(&d_r_begin, size);
    cudaMalloc(&d_r_end, size);



    // Copie des données de l'hôte vers le device
    cudaMemcpy(d_a_begin, h_a_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_end, h_a_end, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b_begin, h_b_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_end, h_b_end, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_r_begin, h_r_begin, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_end, h_r_end, size, cudaMemcpyHostToDevice);


    // Configuration et lancement du kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    //vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a_begin, d_a_end, d_b_begin, d_b_end, d_r_begin, d_r_end, n);

    // Vérification des erreurs
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erreur CUDA: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copie des résultats du device vers l'hôte
    cudaMemcpy(h_r_begin, d_r_begin, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_end, d_r_end, size, cudaMemcpyDeviceToHost);

    // Affichage de quelques résultats pour vérification
    printf("Quelques résultats :\n");
    for (int i = 0; i < 5; i++) {
        printf("%.1f  %.1f  %.1f  %.1f  %.1f  %.1f \n", h_a_begin[i], h_a_end[i], h_b_begin[i], h_b_end[i], h_r_begin[i], h_r_end[i]);
    }

    // Libération de la mémoire
    free(h_a_begin);
    free(h_a_end);
    free(h_b_begin);
    free(h_b_end);
    free(h_r_begin);
    free(h_r_end);    
    cudaFree(d_a_begin);
    cudaFree(d_a_end);
    cudaFree(d_b_begin);
    cudaFree(d_b_end);
    cudaFree(d_r_begin);
    cudaFree(d_r_end);


    return 0;
}
