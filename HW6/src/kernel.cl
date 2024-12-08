__kernel void convolution(
    __global const float* restrict input,
    __global const float* restrict filter,
    __global float* restrict output,
    const int width,
    const int height,
    const int filterWidth)
{
    // Get work-item coordinates
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    // Early exit for out-of-bounds threads
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate filter dimensions
    const int halfFilter = filterWidth / 2;
    
    // Calculate local memory size needed for the tile plus halo regions
    // Using dynamic local memory allocation instead of fixed size
    __local float localInput[32][32];  // Maximum size to accommodate work group and halo
    
    // Load tile and halo region into local memory
    const int blockDimX = get_local_size(0);
    const int blockDimY = get_local_size(1);
    const int tile_start_x = get_group_id(0) * blockDimX - halfFilter;
    const int tile_start_y = get_group_id(1) * blockDimY - halfFilter;
    
    // Each thread loads its own pixel and potentially helps with halo region
    for (int dy = local_y; dy < blockDimY + 2 * halfFilter; dy += blockDimY) {
        int sourceY = tile_start_y + dy;
        for (int dx = local_x; dx < blockDimX + 2 * halfFilter; dx += blockDimX) {
            int sourceX = tile_start_x + dx;
            
            // Safely handle boundary conditions
            float value = 0.0f;
            if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
                value = input[sourceY * width + sourceX];
            }
            
            // Ensure we don't write outside local memory bounds
            if (dy < 32 && dx < 32) {
                localInput[dy][dx] = value;
            }
        }
    }
    
    // Ensure all local memory writes are complete
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute convolution only for valid output pixels
    float sum = 0.0f;
    
    // The pixel we're computing in local memory coordinates
    const int local_mem_x = local_x + halfFilter;
    const int local_mem_y = local_y + halfFilter;
    
    // Only proceed if we're within valid local memory bounds
    if (local_mem_x < 32 - halfFilter && local_mem_y < 32 - halfFilter) {
        for (int i = -halfFilter; i <= halfFilter; i++) {
            for (int j = -halfFilter; j <= halfFilter; j++) {
                int filterIndex = (i + halfFilter) * filterWidth + (j + halfFilter);
                sum += localInput[local_mem_y + i][local_mem_x + j] * filter[filterIndex];
            }
        }
        
        // Write result to global memory
        output[y * width + x] = sum;
    }
}