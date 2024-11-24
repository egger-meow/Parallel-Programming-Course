#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define THREADS_PER_BLOCK 256
#define GROUP_SIZE 2

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel (float lowerX, float lowerY, float stepX, float stepY, 
                             int maxIterations, int* img, size_t pitch, int resX, int resY)  {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int group = 0; group < GROUP_SIZE; ++group) {
        int idx = threadID * GROUP_SIZE + group;

        if (idx >= resX * resY)
            continue;

        int x = idx % resX;     
        int y = idx / resX;      

        float c_re = lowerX + x * stepX;
        float c_im = lowerY + y * stepY;

        int iter = mandel(c_re, c_im, maxIterations);

        int* row = (int*)((char*)img + y * pitch);
        row[x] = iter;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int threadsPerBlockLocal = THREADS_PER_BLOCK;
    int blocksPerGrid = (resX * resY + threadsPerBlockLocal - 1) / threadsPerBlockLocal;


    int* hostImg;
    cudaHostAlloc((void**)&hostImg, resX * resY * sizeof(int), cudaHostAllocDefault);

    int* devImg;
    size_t pitch;
    cudaMallocPitch((void**)&devImg, &pitch, resX * sizeof(int), resY);

    mandelKernel<<<blocksPerGrid, threadsPerBlockLocal>>>(
        lowerX, lowerY, stepX, stepY, maxIterations, devImg, pitch, resX, resY);

    cudaDeviceSynchronize();    
    cudaMemcpy2D(hostImg, resX * sizeof(int), devImg, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    for (int i = 0; i < resX * resY; ++i) 
        img[i] = hostImg[i];
    
    cudaFree(devImg);
    cudaFreeHost(hostImg);
}
