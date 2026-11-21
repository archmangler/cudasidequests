#include <stdio.h>
#include "cuda_runtime.h"

//#define SIZE 2048 //define the size of the vectors
#define SIZE 1024*432*1024 //define the size of the vectors ---> more than 400 000 000 elements

//CUDA kernel for vector addition
__global__ void vectorAdd(int* A, int* B, int* C, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	C[i] = A[i] + B[i];
}

int main() 
{
	//Step 1: allocate memory space
	int* A, * B , * C; // Host vectors
	int* d_A, * d_B , * d_C; // Host vectors
	int size = SIZE * sizeof(int);

	//Step 2: Allocate and initialise the hosts vectors.
	A = (int*)malloco(size); random_ints(A, SIZE);
	B = (int*)malloc(size); random_ints(B, SIZE);
	C = (int*)malloc(size);

	//Step 3 allocate the device vectors
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	//Step 4: initialise the inputs
	for (int i = 0; i < SIZE; i++) {
		A[i] = i;
		B[i] = SIZE - i;
	}
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	 //Step 5: Launch the Vector Add CUDA kernel
	vectorAdd <<< 1024*432,1024 >>> (d_A, d_B, d_C, SIZE);

	cudaDeviceSynchronize();

	//Step 6. copy the result back to the host CPU+mem
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	printf("\nExecution finished\n");
	for (int i = 0; i < SIZE; i++) 
	{
		printf("%d + %d = %d",A[i],B[i], C[i]);
		printf("\n");
	}

	//Step 7. Clean Up
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);

}
