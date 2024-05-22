#ifndef KERNEL_SIZE
#define KERNEL_SIZE 5
float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1,  4,  6,  4,  1},
        {4, 16, 24, 16,  4},
        {6, 24, 36, 24,  6},
        {4, 16, 24, 16,  4},
        {1,  4,  6,  4,  1}
    };
float normalization_factor = 256.0f;
#endif

__kernel void apply_gaussian_blur_kernel(__global const uchar* input, __global uchar* output, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int kernelRadius = KERNEL_SIZE / 2;
    
    float sum = 0.0f;
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            int imageX = x + kx - kernelRadius;
            int imageY = y + ky - kernelRadius;
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                sum += (input[imageY * width + imageX] * gaussian_kernel[ky][kx]);
            }
        }
    }
    
    // Normalize the sum to preserve brightness
    uchar clamped_value = (uchar)(sum / normalization_factor);
    
    // Write the clamped value to the output image
    output[y * width + x] = clamped_value;
}

