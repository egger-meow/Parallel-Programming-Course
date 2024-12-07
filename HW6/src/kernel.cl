__kernel void convolution(
    __global const float* inputImage,
    __global const float* filter,
    __global float* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int filterWidth,
    __local float* localInput
) {
    // Get local and global coordinates
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);
    const int globalX = get_global_id(0);
    const int globalY = get_global_id(1);
    const int localWidth = get_local_size(0);
    const int localHeight = get_local_size(1);
    
    const int halfFilter = filterWidth / 2;
    // Calculate input tile size including padding for filter
    const int inputTileWidth = localWidth + 2 * halfFilter;
    const int inputTileHeight = localHeight + 2 * halfFilter;
    
    // Load input tile into local memory
    for (int y = localY; y < inputTileHeight; y += localHeight) {
        for (int x = localX; x < inputTileWidth; x += localWidth) {
            int globalInputX = globalX - halfFilter + (x - localX);
            int globalInputY = globalY - halfFilter + (y - localY);
            
            // Handle boundary conditions with zero padding
            float value = 0.0f;
            if (globalInputX >= 0 && globalInputX < imageWidth &&
                globalInputY >= 0 && globalInputY < imageHeight) {
                value = inputImage[globalInputY * imageWidth + globalInputX];
            }
            
            localInput[y * inputTileWidth + x] = value;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Ensure we're within image bounds
    if (globalX >= imageWidth || globalY >= imageHeight) {
        return;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < filterWidth; i++) {
        for (int j = 0; j < filterWidth; j++) {
            int localInputX = localX + j;
            int localInputY = localY + i;
            float filterVal = filter[i * filterWidth + j];
            float inputVal = localInput[localInputY * inputTileWidth + localInputX];
            sum += filterVal * inputVal;
        }
    }
    
    outputImage[globalY * imageWidth + globalX] = sum;
}