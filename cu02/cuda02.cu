#include <stdio.h>
#include "cuda_runtime.h"

#define SIZE 1024 //define the size of the vectors

//CUDA kernel for vector addition
__global__ void vectorAdd(int* A, int* B, int* C, int n) {
	int i = threadIndex.x;
	C[i] = A[i] + B[i];
}

int main() 
{
	//Step 1: allocate memory space
	int* A, * B , * C; // Host vectors
	int* d_A, * d_B , * d_C; // Host vectors
	int size = SIZE * sizeof(int);

	//Step 2: Allocate and initialise the hosts vectors.
	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);

	//Step 3 allocate the device vectors
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);


}
