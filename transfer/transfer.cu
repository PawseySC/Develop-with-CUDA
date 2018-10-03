#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

// prints error if detected and exits 
void inline check(cudaError_t err, const char* filename, int line)
{
	if (err != cudaSuccess) 
	{ 
		printf("%s-l%i: %s\n", filename, line, cudaGetErrorString(err)); 
		exit(EXIT_FAILURE);
	}
}

// prints start and end of integer array
void printArrayTerse(int* array, int length, int num)
{
	if (length<2*num) { num = length/2; }
	for (int i=0; i<num; i++)
	{
		printf("%i ",array[i]);
	}
	printf("... ");
    for (int i=length-num-1; i<length; i++)
    {
        printf("%i ",array[i]);
    }
    printf("\n");

}

// copies an array to the GPU and back
int main(int argc, char** argv)
{
	// variable declarations
	cudaError_t err;                 // variable for error codes
	int* hostArray;                  // pointer for array in host memory
	int* deviceArray;                // pointer for array in device memory
	int length = 262144;             // length of array
    int size = length*sizeof(int);   // size of array in bytes

	// allocate host memory
	err = cudaHostAlloc((void**)&hostArray,size,cudaHostAllocDefault);
	check(err, __FILE__, __LINE__);

	// allocate device memory
	err = cudaMalloc((void**)&deviceArray,size);
    check(err, __FILE__, __LINE__);

	// initialise host memory
	for(int i=0; i<length; i++)
	{
		hostArray[i] = i;
	}
	printArrayTerse(hostArray,length,8);

	// copy host to device
	// HINT: insert cudaMemcpy here <--
    check(err, __FILE__, __LINE__);
	
	// clear host memory
	memset(hostArray, 0, size);
	printArrayTerse(hostArray,length,8);

	// copy device to host
	// HINT: insert cudaMemcpy here <--
    check(err, __FILE__, __LINE__);
	printArrayTerse(hostArray,length,8);

	// free device memory
    err = cudaFree(deviceArray);
    check(err, __FILE__, __LINE__);

	// free host memory
	err = cudaFreeHost(hostArray);
    check(err, __FILE__, __LINE__);

	// exit
	return EXIT_SUCCESS;
}
