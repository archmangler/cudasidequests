Vector add returns zeros

20 Nov 2026, RTX 3070 (8 GB), CUDA 13.0 driver, nvcc 12.8, Ubuntu 24.04 on RunPod pod 6sqf39rp5ettzq.

Result

The run is incorrect. Each printed sum should be 1073741824 (SIZE). The device result is 0.

4216805 + 1069525019 = 0
4216806 + 1069525018 = 0
4216807 + 1069525017 = 0
4216808 + 1069525016 = 0

4216805 + 1069525019 is 1073741824, so the host inputs are right and C was never written.

Cause

SIZE is 1024*1024*1024 elements. Each int array is 4 GB. Three device arrays are 12 GB. The RTX 3070 has 8 GB, so cudaMalloc fails. The program does not check the error. malloc pages stay zero, and the failed copy leaves C at 0.

The launch <<<1024*432, 1024>>> also starts only 452,984,832 threads, short of SIZE.

Code

#define SIZE 1024*1024*1024

__global__ void vectorAdd(int* A, int* B, int* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i] + B[i];
}

size_t size = (size_t)SIZE * sizeof(int);
A = (int*)malloc(size);
B = (int*)malloc(size);
C = (int*)malloc(size);
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_B, size);
cudaMalloc((void**)&d_C, size);
for (int i = 0; i < SIZE; i++) { A[i] = i; B[i] = SIZE - i; }
cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
vectorAdd<<<1024*432, 1024>>>(d_A, d_B, d_C, SIZE);
cudaDeviceSynchronize();
cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

Fix

Keep three arrays under 8 GB (about 2 GB each, SIZE near 512*1024*1024 or less). Launch one thread per element, and check cudaMalloc and cudaGetLastError.

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
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < SIZE; i++) if (C[i] != SIZE) bad++;
    printf("checked %d elements, mismatches %d, C[0]=%d\n", SIZE, bad, C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return bad ? 1 : 0;
}
