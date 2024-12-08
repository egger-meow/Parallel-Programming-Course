#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    
    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 0, &status);
    if (status != CL_SUCCESS) return;

    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * imageWidth * imageHeight,
                                      inputImage, &status);
    if (status != CL_SUCCESS) {
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * filterSize,
                                       filter, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                       sizeof(float) * imageWidth * imageHeight,
                                       NULL, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(filterBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(filterBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);

    size_t localWS[2] = {16, 16};  // Work-group size
    size_t globalWS[2] = {
        ((imageWidth + localWS[0] - 1) / localWS[0]) * localWS[0],
        ((imageHeight + localWS[1] - 1) / localWS[1]) * localWS[1]
    };

    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
                                   globalWS, localWS, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(filterBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    status = clEnqueueReadBuffer(cmdQueue, outputBuffer, CL_TRUE, 0,
                                sizeof(float) * imageWidth * imageHeight,
                                outputImage, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(cmdQueue);
}