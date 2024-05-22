__kernel void downsample_kernel(__global const uchar4* input, __global uchar4* output, unsigned width, unsigned height)
{
    unsigned new_width = width / 4;
    unsigned new_height = height / 4;

    // Calculate global indices
    unsigned global_id_x = get_global_id(0);
    unsigned global_id_y = get_global_id(1);

    // Calculate corresponding indices in the original image
    unsigned original_x = global_id_x * 4;
    unsigned original_y = global_id_y * 4;

    // Calculate the output index
    unsigned output_index = global_id_y * new_width + global_id_x;

    // Check if the original indices are within bounds
    if (original_x < width && original_y < height) {
        // Read the pixel from the original image and write to the output
        output[output_index] = input[original_y * width + original_x];
    } else {
        // If out of bounds, write black pixel
        output[output_index] = (uchar4)(0, 0, 0, 255);
    }
}
