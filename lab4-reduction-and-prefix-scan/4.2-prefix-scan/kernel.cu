/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/

__global__ void myScan(float *out, float *in, unsigned in_size) {

    // INSERT CODE HERE
    extern __shared__ float temp[];
    int thrId = threadIdx.x;
    int offset = 1;

    // Load input
    temp[thrId * 2] = in[thrId * 2];
	temp[thrId * 2 + 1] = in[thrId * 2 + 1];
    
    // The downsweep, which builds the sums in place.
    for (int d = in_size >> 1; d > 0; d >>= 1) {
        // d is the floored half of array size
        // d is divided by 2 with every iteration, ending when it rounds to 0
        __syncthreads();
        
        if (thrId < d) {
            int ai = offset * (thrId * 2 + 1) - 1;
            int bi = offset * (thrId * 2 + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Set the last element to 0
    if (thrId == 0) {
        temp[in_size - 1] = 0;
    }

    // Sweep down the tree
    for (int d = 1; d < in_size; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (thrId < d) {
            int ai = offset * (thrId * 2 + 1) - 1;
            int bi = offset * (thrId * 2 + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    out[thrId * 2] = temp[thrId * 2];
    out[thrId * 2 + 1] = temp[thrId * 2 + 1];
}

void preScan(float *out, float *in, unsigned in_size)
{
    // allocate the space
    // call preScan kernel
    // get n_blocks
    // call the kernal on each block

    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x = in_size; dim_grid.y = dim_grid.z = 1;
    myScan<<<dim_grid, dim_block>>>(out, in, in_size);
}