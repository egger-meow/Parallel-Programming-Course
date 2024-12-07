
#include <stdio.h>
#include <stdlib.h>

#include "hostFE.h"
#include "helper.h"

// Define the maximum source size for the kernel
#define MAX_SOURCE_SIZE (0x100000)

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    // OpenCL variables
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    // Get Platform and Device Info
    status = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (status != CL_SUCCESS) {
        // Exit if failed to get platform IDs
        exit(1);
    }

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (status != CL_SUCCESS) {
        // Exit if failed to get device IDs
        exit(1);
    }

    // Create OpenCL context
    cl_context context_local = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        // Exit if failed to create context
        exit(1);
    }

    // Create Command Queue
    cl_command_queue command_queue = clCreateCommandQueue(context_local, device_id, 0, &status);
    if (status != CL_SUCCESS) {
        // Release context and exit if failed to create command queue
        clReleaseContext(context_local);
        exit(1);
    }

    // Create Memory Buffers
    cl_mem input_mem = clCreateBuffer(context_local, CL_MEM_READ_ONLY, 
                                      sizeof(float) * imageHeight * imageWidth, NULL, &status);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to create input buffer
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    cl_mem filter_mem = clCreateBuffer(context_local, CL_MEM_READ_ONLY, 
                                       sizeof(float) * filterSize, NULL, &status);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to create filter buffer
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    cl_mem output_mem = clCreateBuffer(context_local, CL_MEM_WRITE_ONLY, 
                                       sizeof(float) * imageHeight * imageWidth, NULL, &status);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to create output buffer
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Copy input data to memory buffers
    status = clEnqueueWriteBuffer(command_queue, input_mem, CL_TRUE, 0, 
                                  sizeof(float) * imageHeight * imageWidth, inputImage, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to write to input buffer
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    status = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0, 
                                  sizeof(float) * filterSize, filter, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to write to filter buffer
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char fileName[] = "kernel.cl";
    char *source_str;
    size_t source_size;

    fp = fopen(fileName, "r");
    if (!fp) {
        // Release resources and exit if failed to open kernel file
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Create a program from the kernel source
    cl_program program_local = clCreateProgramWithSource(context_local, 1, 
                                                         (const char **)&source_str, 
                                                         (const size_t *)&source_size, &status);
    free(source_str);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to create program
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Build the program
    status = clBuildProgram(program_local, 1, &device_id, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        // Get the build log to understand the error
        size_t log_size;
        clGetProgramBuildInfo(program_local, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program_local, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        // Since printing is not allowed, you might want to write the log to a file or handle it accordingly
        // For debugging purposes, you can temporarily enable printing here
        // printf("Build Log:\n%s\n", log);
        free(log);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program_local, "convolution", &status);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to create kernel
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Set the arguments of the kernel
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem);
    status |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageHeight);
    status |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filterWidth);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to set kernel arguments
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Execute the OpenCL kernel on the list
    size_t global_item_size[2] = { (size_t)imageWidth, (size_t)imageHeight };
    size_t local_item_size[2] = { 16, 16 }; // Adjust as needed based on the device

    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                    global_item_size, local_item_size, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to enqueue kernel
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Wait for the command queue to get serviced before reading back results
    clFinish(command_queue);

    // Read the memory buffer outputImage on the device to the local variable outputImage
    status = clEnqueueReadBuffer(command_queue, output_mem, CL_TRUE, 0, 
                                 sizeof(float) * imageHeight * imageWidth, outputImage, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        // Release resources and exit if failed to read output buffer
        clReleaseKernel(kernel);
        clReleaseProgram(program_local);
        clReleaseMemObject(output_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(input_mem);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context_local);
        exit(1);
    }

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program_local);
    clReleaseMemObject(output_mem);
    clReleaseMemObject(filter_mem);
    clReleaseMemObject(input_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context_local);
}
