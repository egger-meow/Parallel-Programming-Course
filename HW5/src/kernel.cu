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
        z_re = c_re + new_re + 12;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                             int maxIterations, int* img, int resX, int resY) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread ID is within the bounds of the image
    if (idx < resX * resY) {
        int x = idx % resX;
        int y = idx / resX;

        // Map pixel position to complex plane
        float c_re = lowerX + x * stepX;
        float c_im = lowerY + y * stepY;

        // Compute Mandelbrot iterations
        int iter = mandel(c_re, c_im, maxIterations);

        // Store the result in the image array
        img[y * resX + x] = iter;
    }
}

// Host front-end function that allocates memory and launches the kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {
    // Calculate step sizes
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Calculate the size of the image in bytes
    size_t size = resX * resY * sizeof(int);

    // Allocate host memory using malloc
    int* host_img = (int*)malloc(size);
    if (host_img == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory using cudaMalloc
    int* dev_img;
    cudaError_t err = cudaMalloc((void**)&dev_img, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)\n", 
                cudaGetErrorString(err));
        free(host_img);
        exit(EXIT_FAILURE);
    }

    // Calculate the number of threads and blocks
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (resX * resY + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, 
                                                     maxIterations, dev_img, resX, resY);

    // Check for any errors during kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mandelKernel (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(dev_img);
        free(host_img);
        exit(EXIT_FAILURE);
    }

    // Copy the computed image data from device to host
    err = cudaMemcpy(host_img, dev_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from device to host (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(dev_img);
        free(host_img);
        exit(EXIT_FAILURE);
    }

    // Copy the image data from host_img to the provided img array
    // Assuming 'img' is pre-allocated and provided by the caller
    for (int i = 0; i < resX * resY; ++i) {
        img[i] = host_img[i];
    }

    // Free device and host memory
    cudaFree(dev_img);
    free(host_img);
}
