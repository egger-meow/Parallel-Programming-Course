// Example parameters - adjust as needed
#define BLOCK_SIZE 16

__kernel void convolution(
    __global const float* input,
    __global const float* filter,
    __global float* output,
    const int width,
    const int height,
    const int filterWidth
) {
    // Local and global coordinates
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int halfFilter = filterWidth / 2;

    // Extended block size to include filter halo
    const int extendedSize = BLOCK_SIZE + 2 * halfFilter;

    // Allocate local memory for the input tile
    __local float localTile[extendedSize][extendedSize];

    // Compute the global coordinates of the start of the block
    const int blockStartX = get_group_id(0)*BLOCK_SIZE - halfFilter;
    const int blockStartY = get_group_id(1)*BLOCK_SIZE - halfFilter;

    // Each thread loads one pixel (or more) into local memory
    // We'll load a tile of size extendedSize x extendedSize
    for (int tileY = ly; tileY < extendedSize; tileY += get_local_size(1)) {
        for (int tileX = lx; tileX < extendedSize; tileX += get_local_size(0)) {
            int globalX = blockStartX + tileX;
            int globalY = blockStartY + tileY;
            float val = 0.0f;

            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                val = input[globalY * width + globalX];
            }

            localTile[tileY][tileX] = val;
        }
    }

    // Ensure all threads have finished loading local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // If the current global thread is outside the image, just return early
    if (gx >= width || gy >= height) {
        return;
    }

    // Compute the local index of the pixel in the local tile
    int localX = lx + halfFilter;
    int localY = ly + halfFilter;

    float sum = 0.0f;

    // Unroll loops as much as is reasonable - exact factor depends on filterWidth
    // Here we show a manual unroll for up to a certain filter width.
    // If filterWidth is variable, you can still rely on #pragma unroll hints.

    // For general case:
    for (int i = -halfFilter; i <= halfFilter; i++) {
        #pragma unroll
        for (int j = -halfFilter; j <= halfFilter; j++) {
            float val = localTile[localY + i][localX + j];
            float fval = filter[(i + halfFilter)*filterWidth + (j + halfFilter)];
            sum += val * fval;
        }
    }

    // Write the result
    output[gy * width + gx] = sum;
}
