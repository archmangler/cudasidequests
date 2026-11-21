#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#define SIZE (128 * 1024 * 1024)  /* 128M ints, 512 MB each, 1.5 GB on device */
#define THREADS 1024

#define CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

__global__ void vectorAdd(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    size_t size = (size_t)SIZE * sizeof(int);
    int *A, *B, *C, *d_A, *d_B, *d_C;

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    if (!A || !B || !C) { fprintf(stderr, "host malloc failed\n"); return 1; }

    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    for (int i = 0; i < SIZE; i++) { A[i] = i; B[i] = SIZE - i; }
    CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int blocks = (SIZE + THREADS - 1) / THREADS;
    vectorAdd<<<blocks, THREADS>>>(d_A, d_B, d_C, SIZE);
    CHECK(cudaGetLastError());
//    CHECK(cudaDeviceSynchronize());
   cudaDeviceSynchronize();
    CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < SIZE; i++) if (C[i] != SIZE) bad++;
    printf("checked %d elements, mismatches %d, C[0]=%d\n", SIZE, bad, C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return bad ? 1 : 0;
}
