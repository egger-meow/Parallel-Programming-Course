__kernel void convolution(
    __global const float* restrict input,
    __global const float* restrict filter,
    __global float* restrict output,
    const int width,
    const int height,
    const int filterWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    const int halfFilter = filterWidth / 2;
    
    __local float localInput[44][56];  
    
    const int blockDimX = get_local_size(0);  // Will be 36
    const int blockDimY = get_local_size(1);  // Will be 24
    const int tile_start_x = get_group_id(0) * blockDimX - halfFilter;
    const int tile_start_y = get_group_id(1) * blockDimY - halfFilter;
    
    for (int dy = local_y; dy < blockDimY + 2 * halfFilter; dy += blockDimY) {
        int sourceY = tile_start_y + dy;
        for (int dx = local_x; dx < blockDimX + 2 * halfFilter; dx += blockDimX) {
            int sourceX = tile_start_x + dx;
            
            float value = 0.0f;
            if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
                value = input[sourceY * width + sourceX];
            }
            
            if (dy < 44 && dx < 56) {  // Updated bounds check
                localInput[dy][dx] = value;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float sum = 0.0f;
    
    const int local_mem_x = local_x + halfFilter;
    const int local_mem_y = local_y + halfFilter;
    
    if (local_mem_x < 56 - halfFilter && local_mem_y < 44 - halfFilter) {
        for (int i = -halfFilter; i <= halfFilter; i++) {
            for (int j = -halfFilter; j <= halfFilter; j++) {
                int filterIndex = (i + halfFilter) * filterWidth + (j + halfFilter);
                sum += localInput[local_mem_y + i][local_mem_x + j] * filter[filterIndex];
            }
        }
        
        output[y * width + x] = sum;
    }
}