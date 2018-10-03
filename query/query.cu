#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

void inline check(cudaError_t err, const char* filename, int line)
{
	if (err != cudaSuccess) 
	{ 
		printf("%s-l%i: %s\n", filename, line, cudaGetErrorString(err)); 
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char** argv)
{
	cudaError_t err;	// variable for error codes
	int count;			// variable for number of devices
	int device;			// variable for active device id

	err = cudaGetDeviceCount(&count);
	check(err, __FILE__, __LINE__);

	printf("\nFound %i devices\n\n", count);

	for (device = 0; device < count; device++)
	{
		err = cudaSetDevice(device);
		check(err, __FILE__, __LINE__);

		struct cudaDeviceProp p;
		err = cudaGetDeviceProperties(&p, device);
		check(err, __FILE__, __LINE__);

		printf("Device %i : ", device);
		printf("%s ", p.name);
		printf("with %i SMs\n", p.multiProcessorCount);
	}
	
	printf("\n");

	return EXIT_SUCCESS;
}
