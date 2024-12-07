__kernel void convolution(
    __global const float* inputImage,
    __global const float* filter,
    __global float* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int filterWidth
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure valid pixel coordinates
    if (x >= imageWidth || y >= imageHeight) {
        return;
    }

    int halfFilter = filterWidth / 2;
    float sum = 0.0f;

    for (int i = -halfFilter; i <= halfFilter; i++) {
        for (int j = -halfFilter; j <= halfFilter; j++) {
            int currentX = x + j;
            int currentY = y + i;

            // Zero-padding condition
            float imageVal = 0.0f;
            if (currentX >= 0 && currentX < imageWidth && currentY >= 0 && currentY < imageHeight) {
                imageVal = inputImage[currentY * imageWidth + currentX];
            }

            float filterVal = filter[(i + halfFilter)*filterWidth + (j + halfFilter)];
            sum += imageVal * filterVal;
        }
    }

    outputImage[y * imageWidth + x] = sum;
}
