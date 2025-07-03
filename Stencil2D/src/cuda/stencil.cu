#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip> // For setprecision
#include <cuda.h>

#include "utils.hpp"

using namespace std;

void getDeviceProperties(int device, cudaDeviceProp* prop);
int getBlockSize(cudaDeviceProp* prop, int threads, int sharedPerThread, int regsPerThread);

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}


__global__
void stencil(float* a, float* a_aux, int rows, int cols, float diff)
{

    int i, j;
    for(i = 1; i < rows-1; i++){
        for(j = 1; j < cols-1; j++){
            a_aux[i * cols + j] = a[(i-1) * cols + j]
                                + a[(i+1) * cols + j]
                                + a[i * cols + (j+1)]
                                + a[(i+1) * cols + (j-1)]
                                -diff * a[i * cols + j];
        }   
    }
}


int main(int argc, char* argv[])
{
    string input = "./inputs/10x10.csv";
    int rows = 10;
    int cols = 10;
    int iterations = 1;
    float diff = 4.0;
    
    int size = (rows+2) * (cols+2);
    float *a = (float*) calloc(size, sizeof(float));
    readCSV(input, a, rows, cols+2);
    printMatrix(a, rows+2, cols+2);

    // Device Info
    CHECK_CUDA_CALL( cudaSetDevice(0) );
    CHECK_CUDA_CALL( cudaDeviceSynchronize() );
    cudaDeviceProp prop;
    getDeviceProperties(0, &prop);

    // Allocate GPU data structures
    float *d_a, *d_aux, *temp;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_aux, size * sizeof(float));

    // Send data to GPU
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    
    for(int i = 0; i < iterations; i++){

        stencil<<<1, 1, 1>>>(d_a, d_aux, rows+2, cols+2, diff);
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        temp = d_a;
        d_a = d_aux;
        d_aux = temp;
    }

    // Copy result to CPU
    cudaMemcpy(a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(a, rows+2, cols+2);


    //Free GPU memory
    cudaFree(d_a);
    cudaFree(d_aux);

    // Free CPU memory
    free(a);

    return 0;
}


/*
    Gets properties of cuda device
    DEBUG MODE: prints properties
*/
void getDeviceProperties(int device, cudaDeviceProp *prop)
{
    cudaGetDeviceProperties(prop, device);

    #ifdef DEBUG
    printf("\nDevice %d Properties\n", device);
    printf("  Memory Clock Rate (MHz): %d\n", prop->memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n", prop->memoryBusWidth);

    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
        2.0*prop->memoryClockRate*(prop->memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop->totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Bytes) %.1f\n",(float)(prop->sharedMemPerBlock));
    printf("  Shared memory per SM (Bytes) %.1f\n",(float)(prop->sharedMemPerMultiprocessor));
    
    printf("  SM count : %d\n", prop->multiProcessorCount);
    printf("  Warp-size: %d\n", prop->warpSize);
    printf("  max-threads-per-block: %d\n", prop->maxThreadsPerBlock);
    printf("  max-threads-per-multiprocessor: %d\n", prop->maxThreadsPerMultiProcessor);
    printf("  register-per-block: %d\n", prop->regsPerBlock);
    #endif
    
}

/*
    Compute ideal blockSize for a kernel
*/
int getBlockSize(cudaDeviceProp* prop, int threads, int sharedPerThread, int regsPerThread)
{

    int warpSize = prop->warpSize;
    int regsPerBlock = prop->regsPerBlock;
    int sharedMem = prop->sharedMemPerBlock;

    // For cc >= 3.0 we have at least 4 warpSchedulers per SM

    int blockSize = 4*warpSize;
    blockSize = min(blockSize, regsPerBlock/regsPerThread);
    blockSize = min(blockSize, sharedMem/sharedPerThread);
    blockSize = min(blockSize, prop->maxThreadsPerMultiProcessor);

    blockSize = warpSize * ceil(blockSize/warpSize);

    return blockSize;
}