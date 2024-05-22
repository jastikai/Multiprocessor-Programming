__kernel void to_grayscale_kernel(__global const uchar4* input,
                                   __global uchar* output,
                                   const unsigned int width,
                                   const unsigned int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel = input[index];
        output[index] = 0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z;
    }
}
