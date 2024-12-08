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
    
    // Update local memory size to accommodate 36x24 work group plus halo regions
    // If halfFilter is 1, we need 38x26 size (36+2 x 24+2)
    // If halfFilter is 2, we need 40x28 size (36+4 x 24+4)
    // Let's make it slightly larger to be safe
    __local float localInput[44][56];  // Accommodates work group of 36x24 plus generous halo
    
    // Load tile and halo region into local memory
    const int blockDimX = get_local_size(0);  // Will be 36
    const int blockDimY = get_local_size(1);  // Will be 24
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
            if (dy < 44 && dx < 56) {  // Updated bounds check
                localInput[dy][dx] = value;
            }
        }
    }
    
    // Ensure all local memory writes are complete
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute convolution only for valid output pixels
    float sum = 0.0f;
    
    // The pixel we're computing in local memory coordinates
    const int local_mem_x = local_ax + halfFilter;
    const int local_mem_y = local_y + halfFilter;
    
    // Update bounds check for new local memory size
    if (local_mem_x < 56 - halfFilter && local_mem_y < 44 - halfFilter) {
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