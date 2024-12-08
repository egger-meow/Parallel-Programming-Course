__kernel void convolution(
    __global const float* input,
    __global const float* filter,
    __global float* output,
    const int width,
    const int height,
    const int filterWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int halfFilter = filterWidth / 2;
    float sum = 0.0f;

    #pragma unroll 4
    for (int i = -halfFilter; i <= halfFilter; i++) {
        #pragma unroll 4
        for (int j = -halfFilter; j <= halfFilter; j++) {
            const int currentY = y + i;
            const int currentX = x + j;
            
            if (currentY >= 0 && currentY < height && 
                currentX >= 0 && currentX < width) {
                const int inputIdx = currentY * width + currentX;
                const int filterIdx = (i + halfFilter) * filterWidth + (j + halfFilter);
                sum += input[inputIdx] * filter[filterIdx];
            }
        }
    }
    

    
    output[y * width + x] = sum;
}