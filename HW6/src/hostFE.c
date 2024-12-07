#include <stdio.h>
#include <stdlib.h>

#include "hostFE.h"
#include "helper.h"

#define MAX_SOURCE_SIZE (0x100000)

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    // Select platform
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    status = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (status != CL_SUCCESS) {
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Select device (GPU)
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (status != CL_SUCCESS) {
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Create context
    cl_context context_local = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(context_local, device_id, 0, &status);
    if (status != CL_SUCCESS) {
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Create buffers
    cl_mem input_mem = clCreateBuffer(context_local, CL_MEM_READ_ONLY,
                                      sizeof(float)*imageWidth*imageHeight, NULL, &status);
    if (status != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    cl_mem filter_mem = clCreateBuffer(context_local, CL_MEM_READ_ONLY,
                                       sizeof(float)*filterSize, NULL, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    cl_mem output_mem = clCreateBuffer(context_local, CL_MEM_WRITE_ONLY,
                                       sizeof(float)*imageWidth*imageHeight, NULL, &status);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Write data to buffers
    status = clEnqueueWriteBuffer(command_queue, input_mem, CL_TRUE, 0,
                                  sizeof(float)*imageWidth*imageHeight, inputImage, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    status = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0,
                                  sizeof(float)*filterSize, filter, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Read kernel source
    FILE *fp = fopen("kernel.cl", "r");
    if (!fp) {
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Create and build program
    cl_program program_local = clCreateProgramWithSource(context_local, 1,
                                                         (const char **)&source_str,
                                                         (const size_t *)&source_size,
                                                         &status);
    free(source_str);
    if (status != CL_SUCCESS) {
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    status = clBuildProgram(program_local, 1, &device_id, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program_local, "convolution", &status);
    if (status != CL_SUCCESS) {
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Set kernel args
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);

    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Run kernel
    size_t global_size[2] = {(size_t)imageWidth, (size_t)imageHeight};
    size_t local_size[2] = {16, 16}; // Adjust if necessary
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                    global_size, local_size, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Finish queue
    clFinish(command_queue);

    // Read result
    status = clEnqueueReadBuffer(command_queue, output_mem, CL_TRUE, 0,
                                 sizeof(float)*imageWidth*imageHeight, outputImage,
                                 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        
    fprintf(stderr, "OpenCL error %d: Failed to get platform IDs\n", status);
    return;
;
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program_local);
    clReleaseMemObject(output_mem);
    clReleaseMemObject(filter_mem);
    clReleaseMemObject(input_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context_local);
}
