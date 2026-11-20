#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01() {
	//print the block and thread IDs
	//warp=32threads. (128 threads/block) --> (128/4 = 4 warps/block)
	int warp_ID_Value = 0;
	warp_ID_Value = threadIdx.x/32;
	printf("\nThe block ID is %d --- The thread ID is %d --- The warp ID %d\n",blockIdx.x, threadIdx.x, warp_ID_Value);
	//printf("\nThe block ID is %d --- The thread ID is %d",blockIdx.x, threadIdx.x);
}

int main() {
	//print the blocks and threads IDs
	//add two vectors has 48 elements
	//kernel_name<<<num_of_blocks, Num_of_threads_per_block>>>();
	test01 <<<2, 64>>> ();
	//The kernel launch is asynchronous, and main returns before it runs, so the process exits and the device printf is discarded.
	//Add a sync before return:
	//You should then see 128 lines (2 blocks × 64 threads). The nvcc architecture warning is not why this run is silent. 
	//nvidia-smi only shows memory in use while a process is running, and this one exits immediately.
	cudaDeviceSynchronize();

	return 0;
}

