#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define THREADS_PER_BLOCK 256

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

// CUDA Kernel: Each thread computes one pixel
__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                             int maxIterations, int* img, size_t pitch, int resX, int resY) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate total number of pixels
    int totalPixels = resX * resY;

    // Ensure the thread ID is within the bounds of the image
    if (idx < totalPixels) {
        int x = idx % resX;      // Column index
        int y = idx / resX;      // Row index

        // Map pixel position to complex plane
        float c_re = lowerX + x * stepX;
        float c_im = lowerY + y * stepY;

        // Compute Mandelbrot iterations
        int iter = mandel(c_re, c_im, maxIterations);

        // Calculate the address considering the pitch
        int* row = (int*)((char*)img + y * pitch);
        row[x] = iter;
    }
}

// Host front-end function that allocates memory and launches the kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {
    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Calculate the total number of pixels
    size_t totalPixels = resX * resY;

    // Define block and grid sizes
    int threadsPerBlockLocal = THREADS_PER_BLOCK;
    int blocksPerGrid = (totalPixels + threadsPerBlockLocal - 1) / threadsPerBlockLocal;

    // Allocate pinned host memory using cudaHostAlloc
    int* host_img;
    cudaError_t err = cudaHostAlloc((void**)&host_img, resX * resY * sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory (error code %s)\n", 
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate pitched device memory using cudaMallocPitch
    int* dev_img;
    size_t pitch;
    err = cudaMallocPitch((void**)&dev_img, &pitch, resX * sizeof(int), resY);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pitched device memory (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFreeHost(host_img);
        exit(EXIT_FAILURE);
    }

    // Launch the kernel
    mandelKernel<<<blocksPerGrid, threadsPerBlockLocal>>>(lowerX, lowerY, stepX, stepY, 
                                                         maxIterations, dev_img, pitch, resX, resY);

    // Check for any errors during kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mandelKernel (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(dev_img);
        cudaFreeHost(host_img);
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching kernel!\n", 
                cudaGetErrorString(err));
        cudaFree(dev_img);
        cudaFreeHost(host_img);
        exit(EXIT_FAILURE);
    }

    // Copy the computed image data from device to host using cudaMemcpy2D
    err = cudaMemcpy2D(host_img, resX * sizeof(int), dev_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from device to host (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(dev_img);
        cudaFreeHost(host_img);
        exit(EXIT_FAILURE);
    }

    // Copy the image data from host_img to the provided img array
    // Assuming 'img' is pre-allocated and provided by the caller
    for (int i = 0; i < resX * resY; ++i) {
        img[i] = host_img[i];
    }

    // Free device and host memory
    cudaFree(dev_img);
    cudaFreeHost(host_img);
}