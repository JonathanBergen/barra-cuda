/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE

    // Shared memory
    __shared__ int sData[];

    // Store the thread id, in both local and global context
    unsigned int thrId = threadId.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load each thread's data
    sData[thrId] = in[i];
    __syncthreads();

    // Stride through the elements
    for (unsigned int stride = 2; stride < blockDim.x; stride *= 2) {
        if (thrId % stride == 0) {
            sData[thrId] += sData[thrId + stride / 2];
        }
        __syncthreads();
    }

    // Write the partial sum back to global memory
    if (thrId == 0) {
        out[blockIdx.x] = sData[0];
    }

}
