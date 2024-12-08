void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    
    // Create command queue with profiling enabled
    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 
                                                   CL_QUEUE_PROFILING_ENABLE, &status);
    
    // Use pinned memory for better transfer speeds
    cl_mem h_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(float) * imageWidth * imageHeight, NULL, &status);
    cl_mem h_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(float) * imageWidth * imageHeight, NULL, &status);
    
    // Map buffers for efficient host access
    float* mapped_input = (float*)clEnqueueMapBuffer(cmdQueue, h_input, CL_TRUE, CL_MAP_WRITE,
                                                    0, sizeof(float) * imageWidth * imageHeight,
                                                    0, NULL, NULL, &status);
    memcpy(mapped_input, inputImage, sizeof(float) * imageWidth * imageHeight);
    
    // Create device buffers
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                  sizeof(float) * imageWidth * imageHeight,
                                  NULL, &status);
    
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * filterWidth * filterWidth,
                                   filter, &status);
    
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * imageWidth * imageHeight,
                                   NULL, &status);
    
    // Copy input data to device
    status = clEnqueueWriteBuffer(cmdQueue, d_input, CL_FALSE, 0,
                                sizeof(float) * imageWidth * imageHeight,
                                mapped_input, 0, NULL, NULL);
    
    // Unmap input buffer
    clEnqueueUnmapMemObject(cmdQueue, h_input, mapped_input, 0, NULL, NULL);
    
    // Create and setup kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    
    // Set work sizes optimized for the hardware
    size_t localWS[2] = {36, 24};  // Optimal work-group size for most GPUs
    size_t globalWS[2] = {
        ((imageWidth + localWS[0] - 1) / localWS[0]) * localWS[0],
        ((imageHeight + localWS[1] - 1) / localWS[1]) * localWS[1]
    };
    
    // Launch kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
                                   globalWS, localWS, 0, NULL, NULL);
    
    // Map output buffer for efficient host access
    float* mapped_output = (float*)clEnqueueMapBuffer(cmdQueue, h_output, CL_TRUE, CL_MAP_READ,
                                                     0, sizeof(float) * imageWidth * imageHeight,
                                                     0, NULL, NULL, &status);
    
    // Copy results back to host
    status = clEnqueueReadBuffer(cmdQueue, d_output, CL_TRUE, 0,
                                sizeof(float) * imageWidth * imageHeight,
                                mapped_output, 0, NULL, NULL);
    
    memcpy(outputImage, mapped_output, sizeof(float) * imageWidth * imageHeight);
    
    // Cleanup
    clEnqueueUnmapMemObject(cmdQueue, h_output, mapped_output, 0, NULL, NULL);
    clReleaseMemObject(h_input);
    clReleaseMemObject(h_output);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQueue);
}