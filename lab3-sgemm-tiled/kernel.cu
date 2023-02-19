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

    // print the matrix
    // printf("%d ", A);

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *      
     *   A has m rows (height of m) and k columns (width of k)
     *   B has k rows (height of k) and n columns (width of n)
     *   C has m rows (height of m) and n columns (width of n)
     *   
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
    int threadRow = blockY * blockDim.y + threadY;
    int threadColumn = blockX * blockDim.x + threadX;

    // Store the value
    float calcVal = 0;

    // Loop through the tiles
    for (int a = 0; a < (TILE_SIZE + k - 1) / TILE_SIZE; ++a) {

        // Check if the thread is inside bounds of the A matrix
        if (a * TILE_SIZE + threadX < k && threadRow < m) {   
            // Load the tile into shared memory
            Ashared[threadY][threadX] = A[threadRow * k + (a * TILE_SIZE + threadX)];
        }
        else {
            // Set it to zero: it is outside the final matrix's range
            Ashared[threadY][threadX] = 0.0;
        }

        // Check if the thread is inside the bounds of the B matrix
        if (a * TILE_SIZE + threadY < k && threadColumn < n) {
            // Load the tile in shared memory
            Bshared[threadY][threadX] = B[(a * TILE_SIZE + threadY) * k + threadColumn];
        }
        else {
            // Set it to zero: it is outside the final matrix's range
            Bshared[threadY][threadX] = 0.0;
        }

        __syncthreads();

        // Work through the tile, adding up the running sum
        for (int b = 0; b < TILE_SIZE; ++b) {
            calcVal += Ashared[threadY][b] * Bshared[b][threadX];

            __syncthreads();
        }

        // Add all the in-bound elements to the C matrix
        if (threadRow < m && threadColumn < n) {
            C[threadRow * n + threadColumn] = calcVal;
        }
    }
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

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    dim3 gridDim((n - 1)/BLOCK_SIZE + 1, (m - 1)/BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    mysgemm<<< gridDim, blockDim >>> (m, n, k, A, B, C);

}
