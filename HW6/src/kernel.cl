__kernel void convolution(
    __global const float* input,
    __global const float* filter,
    __global float* output,
    const int width,
    const int height,
    const int filterWidth)
{
    // Get global ID
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Return if out of bounds
    if (x >= width || y >= height) return;
    
    const int halfFilter = filterWidth / 2;
    float sum = 0.0f;

    // Loop unrolling for common filter sizes (7x7)
    #pragma unroll 4
    for (int i = -halfFilter; i <= halfFilter; i++) {
        #pragma unroll 4
        for (int j = -halfFilter; j <= halfFilter; j++) {
            const int currentY = y + i;
            const int currentX = x + j;
            
            // Check boundaries
            if (currentY >= 0 && currentY < height && 
                currentX >= 0 && currentX < width) {
                const int inputIdx = currentY * width + currentX;
                const int filterIdx = (i + halfFilter) * filterWidth + (j + halfFilter);
                sum += input[inputIdx] * filter[filterIdx];
            }
        }
    }
    
    // Write output
    output[y * width + x] = sum;
}