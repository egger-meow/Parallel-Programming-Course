// Optimized kernel implementation
__kernel void convolution(
    __global const float* restrict input,
    __global const float* restrict filter,
    __global float* restrict output,
    const int width,
    const int height,
    const int filterWidth)
{
    // Get local work-item IDs
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    
    // Get global work-item IDs
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    // Early exit for out-of-bounds threads
    if (x >= width || y >= height) return;
    
    // Calculate filter dimensions
    const int halfFilter = filterWidth / 2;
    
    // Declare local memory for input tile
    __local float localInput[18][18];  // 16x16 work-group + 2-pixel halo for filter1
    
    // Load data into local memory including halo region
    const int tile_start_x = get_group_id(0) * get_local_size(0) - halfFilter;
    const int tile_start_y = get_group_id(1) * get_local_size(1) - halfFilter;
    
    // Load main block plus halo region
    for (int i = local_y; i < get_local_size(1) + 2*halfFilter; i += get_local_size(1)) {
        for (int j = local_x; j < get_local_size(0) + 2*halfFilter; j += get_local_size(0)) {
            int global_x = tile_start_x + j;
            int global_y = tile_start_y + i;
            
            // Handle boundary conditions with zero padding
            float value = 0.0f;
            if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                value = input[global_y * width + global_x];
            }
            localInput[i][j] = value;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate local memory position
    const int local_mem_x = local_x + halfFilter;
    const int local_mem_y = local_y + halfFilter;
    
    // Perform convolution using local memory
    float sum = 0.0f;
    
    #pragma unroll
    for (int i = -halfFilter; i <= halfFilter; i++) {
        #pragma unroll
        for (int j = -halfFilter; j <= halfFilter; j++) {
            const int filterIdx = (i + halfFilter) * filterWidth + (j + halfFilter);
            sum += localInput[local_mem_y + i][local_mem_x + j] * filter[filterIdx];
        }
    }
    
    // Write result
    output[y * width + x] = sum;
}