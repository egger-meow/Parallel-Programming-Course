// kernel.cl

__kernel void convolution(
    __global const float* inputImage,
    __global const float* filter,
    __global float* outputImage,
    const int imageWidth,
    const int imageHeight,
    const int filterWidth
) {
    // Get the coordinates of the pixel this work-item will compute
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure that we do not compute out-of-range pixels
    if (x >= imageWidth || y >= imageHeight) {
        return;
    }

    // Determine half the filter size (just like halffilterSize in serial version)
    int halfFilter = filterWidth / 2;
    float sum = 0.0f;

    // Iterate over the filter region
    // This matches:
    // for (k = -halffilterSize; k <= halffilterSize; k++)
    //   for (l = -halffilterSize; l <= halffilterSize; l++)
    for (int i = -halfFilter; i <= halfFilter; i++) {
        for (int j = -halfFilter; j <= halfFilter; j++) {
            int currentY = y + i;
            int currentX = x + j;

            // Boundary checks - zero-padding
            if (currentY >= 0 && currentY < imageHeight &&
                currentX >= 0 && currentX < imageWidth) {

                // Access input pixel
                float imageVal = inputImage[currentY * imageWidth + currentX];

                // Access filter coefficient
                // filter[(k + halffilterSize)*filterWidth + (l + halffilterSize)]
                float filterVal = filter[(i + halfFilter)*filterWidth + (j + halfFilter)];

                sum += imageVal * filterVal;
            }
        }
    }

    // Write the computed sum to output
    outputImage[y * imageWidth + x] = sum;
}
