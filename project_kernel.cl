// Jaakko Astikainen 2024

#include "project.h"

__kernel void downsample(__read_only image2d_t image,
                         unsigned width,
                         unsigned height,
                         unsigned new_width,
                         unsigned new_height,
                         __write_only image2d_t resized_image)
{
    // Get global ID in 2D
    unsigned int x = get_global_id(0); // Column index
    unsigned int y = get_global_id(1); // Row index

    if (x < new_width && y < new_height) {
        // Read pixel value from the original image
        uint4 pixel = read_imageui(image, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE, (int2)(x * DOWN_RATIO, y * DOWN_RATIO));

        // Write pixel value to the resized image
        write_imageui(resized_image, (int2)(x, y), pixel);
    }
}

__kernel void to_grayscale(__read_only image2d_t image,
                            unsigned width,
                            unsigned height,
                            __global uchar* gray_image)
{
    // Get global ID in 2D
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Read pixel value from the input image
        int2 coords = (int2)(x, y);
        uint4 pixel = read_imageui(image, coords);

        // Convert RGB to grayscale using luminance formula
        uchar gray = (uchar)(0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z);

        // Calculate index for accessing the pixel in the output image
        int index = y * width + x;

        // Store grayscale value in the output buffer
        gray_image[index] = gray;
    }
}


__kernel void compute_disparity_map(__global uchar* left_image,
                                    __global uchar* right_image,
                                    uint width,
                                    uint height,
                                    int direction,
                                    uint scaled_ndisp,
                                    __global uchar* disparity_map)
{
    // Get global indices
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    // Ensure the thread processes a valid pixel
    if (x < width && y < height) {
        int kernel_radius = KERNEL_SIZE / 2;
        int best_disparity = 0;
        float best_zncc = -1;
        
        // Adjust stereo matching direction based on input parameter
        int start_disparity = 0;//(direction == 0) ? 0 : scaled_ndisp - 1;
        int end_disparity = scaled_ndisp;//(direction == 0) ? scaled_ndisp : -1;
        int step = 1;//(direction == 0) ? 1 : -1;
        
        // Iterate over disparities
        for (int d = start_disparity; d != end_disparity; d += step) {
            // Check if the current disparity allows the kernel to fit within the image boundaries
            if (((!direction && x >= kernel_radius + d) || (direction && x >= kernel_radius)) && ((!direction && x < width - kernel_radius ) || (direction && x < width - kernel_radius - d)) && y >= kernel_radius && y < height - kernel_radius) {
                // Extract patches from left and right images
                uchar left_patch[KERNEL_SIZE * KERNEL_SIZE];
                uchar right_patch[KERNEL_SIZE * KERNEL_SIZE];
                for (int i = 0; i < KERNEL_SIZE; i++) {
                    for (int j = 0; j < KERNEL_SIZE; j++) {
                        int left_x = x - kernel_radius + j;
                        int left_y = y - kernel_radius + i;
                        int right_x = direction ? x - kernel_radius + j + d : x - kernel_radius + j - d;
                        int right_y = y - kernel_radius + i;
                        left_patch[i * KERNEL_SIZE + j] = left_image[left_y * width + left_x];
                        right_patch[i * KERNEL_SIZE + j] = right_image[right_y * width + right_x];
                    }
                }
                // Compute ZNCC score
                float mean_left = 0, mean_right = 0, std_left = 0, std_right = 0, cross_corr = 0;
                for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
                    mean_left += left_patch[i];
                    mean_right += right_patch[i];
                }
                mean_left /= (KERNEL_SIZE * KERNEL_SIZE);
                mean_right /= (KERNEL_SIZE * KERNEL_SIZE);
                for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
                    float diff_left = left_patch[i] - mean_left;
                    float diff_right = right_patch[i] - mean_right;
                    std_left += diff_left * diff_left;
                    std_right += diff_right * diff_right;
                    cross_corr += diff_left * diff_right;
                }
                std_left = sqrt(std_left / (KERNEL_SIZE * KERNEL_SIZE));
                std_right = sqrt(std_right / (KERNEL_SIZE * KERNEL_SIZE));
                float zncc = cross_corr / (std_left * std_right);
                // Update best disparity if needed
                if (zncc > best_zncc) {
                    best_zncc = zncc;
                    best_disparity = d;
                }
            }
        }
        // Normalize disparity value to the range of 0-255
        // disparity_map[y * width + x] = (uchar)((best_disparity * 255) / scaled_ndisp);
        disparity_map[y * width + x] = (uchar)(best_disparity);
    }
}

__kernel void cross_checking(__global uchar* left_disparity_map,
                              __global uchar* right_disparity_map,
                              __global uchar* consolidated_disparity_map,
                              const unsigned width,
                              const unsigned height,
                              const unsigned threshold,
                              uint scaled_ndisp) {
    // Get the global ID in 2D
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    // Calculate the index in the flattened array
    const unsigned int index = y * width + x;

    // Check if within bounds
    if (x < width && y < height) {
        // Perform cross-checking logic
        if ((int)(abs(left_disparity_map[index] - right_disparity_map[index - left_disparity_map[index]]))*255/scaled_ndisp > threshold) {
            // If the absolute difference is larger than the threshold, set the pixel value to zero
            consolidated_disparity_map[index] = 0;
        } else {
            // Otherwise, retain the pixel value from the left disparity map
            consolidated_disparity_map[index] = (uchar)(((int)left_disparity_map[index])*255/scaled_ndisp);
        }
    }
}

__kernel void occlusion_filling(__global uchar* disparity_map,
                                __global uchar* occlusion_map,
                                const unsigned int width,
                                const unsigned int height,
                                const int max_radius)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    
    if (x < width && y < height) {
        // Get the current pixel index
        const unsigned int index = y * width + x;
        occlusion_map[index] = disparity_map[index];
        
        // Check if the current pixel needs occlusion filling
        if (disparity_map[index] == 0) {
            // Variables to store the nearest non-zero neighbor
            int nearest_disparity = -1;
            float min_distance = FLT_MAX;
            
            // Iterate over the neighborhood
            for (int dy = -max_radius; dy <= max_radius; dy++) {
                for (int dx = -max_radius; dx <= max_radius; dx++) {
                    // Skip the current pixel
                    if (dx == 0 && dy == 0)
                        continue;
                    
                    // Calculate neighbor coordinates
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    // Ensure neighbor is within bounds
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        // Calculate Manhattan distance
                        float distance = abs(dx) + abs(dy);
                        
                        // Check if neighbor has a non-zero disparity
                        if (disparity_map[ny * width + nx] != 0) {
                            // Update nearest disparity if closer neighbor found
                            if (distance < min_distance) {
                                min_distance = distance;
                                nearest_disparity = disparity_map[ny * width + nx];
                            }
                        }
                    }
                }
            }
            
            // Update current pixel with nearest non-zero disparity
            occlusion_map[index] = (nearest_disparity != -1) ? nearest_disparity : 0;
        }
    }
}
