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
    
    const int halfFilter = filterWidth / 2;
    
    // Local memory sized to fit 36x24 work group plus halo regions
    // Adding 2*halfFilter for each dimension to accommodate the filter radius
    __local float localInput[26][38];  // 24+2 x 36+2 for filter1's halo region
    
    // Calculate block dimensions and starting positions
    const int tile_start_x = get_group_id(0) * 36 - halfFilter;  // 36 is work group x dimension
    const int tile_start_y = get_group_id(1) * 24 - halfFilter;  // 24 is work group y dimension
    
    // Load data into local memory including halo region
    // Each thread might need to load multiple elements to cover the tile plus halo
    for (int dy = local_y; dy < 24 + 2*halfFilter; dy += 24) {
        for (int dx = local_x; dx < 36 + 2*halfFilter; dx += 36) {
            if (dy < 26 && dx < 38) {  // Check local memory bounds
                int sourceY = tile_start_y + dy;
                int sourceX = tile_start_x + dx;
                
                // Handle boundary conditions with zero padding
                float value = 0.0f;
                if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
                    value = input[sourceY * width + sourceX];
                }
                localInput[dy][dx] = value;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute convolution only if we're within valid bounds
    if (x < width && y < height) {
        float sum = 0.0f;
        
        // Calculate position in local memory
        const int local_mem_x = local_x + halfFilter;
        const int local_mem_y = local_y + halfFilter;
        
        // Ensure we're within valid local memory bounds
        if (local_mem_y < 26 - halfFilter && local_mem_x < 38 - halfFilter) {
            #pragma unroll
            for (int i = -halfFilter; i <= halfFilter; i++) {
                #pragma unroll
                for (int j = -halfFilter; j <= halfFilter; j++) {
                    const int filterIdx = (i + halfFilter) * filterWidth + (j + halfFilter);
                    sum += localInput[local_mem_y + i][local_mem_x + j] * filter[filterIdx];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}