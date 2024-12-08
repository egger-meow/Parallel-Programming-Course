#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    
    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 0, &status);
    if (status != CL_SUCCESS) return;

    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * imageWidth * imageHeight,
                                  inputImage, &status);
    if (status != CL_SUCCESS) {
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * filterWidth * filterWidth,
                                   filter, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * imageWidth * imageHeight,
                                   NULL, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(d_filter);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_filter);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    
    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_filter);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    size_t maxWorkGroupSize;
    clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                    sizeof(size_t), &maxWorkGroupSize, NULL);
    
    size_t localWS[2] = {36, 24};  
    if (864 > maxWorkGroupSize) {
        localWS[0] = 8;
        localWS[1] = 8;  
    }
    
    size_t globalWS[2] = {
        ((imageWidth + localWS[0] - 1) / localWS[0]) * localWS[0],
        ((imageHeight + localWS[1] - 1) / localWS[1]) * localWS[1]
    };

    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
                                   globalWS, localWS, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_filter);
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(cmdQueue);
        return;
    }

    status = clEnqueueReadBuffer(cmdQueue, d_output, CL_TRUE, 0,
                                sizeof(float) * imageWidth * imageHeight,
                                outputImage, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_input);
    clReleaseCommandQueue(cmdQueue);
}