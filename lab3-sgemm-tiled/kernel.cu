/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    // Calculate the X and Y of blocks and threads
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Compute the row and column of the targeted product element
    int threadRow = blockY * TILE_SIZE + threadY;
    int threadColumn = blockX * TILE_SIZE + threadX;

    // Store the value
    float calcVal = 0;

    // I think k is the correct value to use here, because it's the shared dimension
    for (int a = 0; a < k / TILE_SIZE; ++a) {
        
        // Load the tiles into shared memory
        Ashared[threadY][threadX] = A[threadRow * k + (a * TILE_SIZE+ threadX)];
        Bshared[threadY][threadX] = B[(a * TILE_SIZE + threadX) * k + threadColumn];

        __syncthreads();

        for (int b = 0; b < TILE_SIZE; ++b) {
            calcVal += Ashared[threadY][b] * Bshared[b][threadX];

            __syncthreads();
        }

        C[threadRow * k + threadColumn] = calcVal;

    }

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    // I think that m holds the width 

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE



}


