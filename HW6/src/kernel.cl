// kernel.cl

__kernel void convolution(
    __global const float* inputImage,
    __global const float* filter,
    __global float* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int filterWidth
) {
    // Get the x and y coordinates of the pixel
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Calculate the half size of the filter
    int halfFilter = filterWidth / 2;

    float sum = 0.0f;

    // Iterate over the filter
    for (int i = -halfFilter; i <= halfFilter; i++) {
        for (int j = -halfFilter; j <= halfFilter; j++) {
            int currentX = x + j;
            int currentY = y + i;

            // Check for boundary conditions (zero-padding)
            if (currentX >= 0 && currentX < imageWidth && 
                currentY >= 0 && currentY < imageHeight) {
                float imageVal = inputImage[currentY * imageWidth + currentX];
                float filterVal = filter[(i + halfFilter) * filterWidth + (j + halfFilter)];
                sum += imageVal * filterVal;
            }
            // Else, contribute 0 (implicitly by not adding anything)
        }
    }

    // Write the result to the output image
    outputImage[y * imageWidth + x] = sum;
}
