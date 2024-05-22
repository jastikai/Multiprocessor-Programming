// Jaakko Astikainen 2024

#include "lodepng/lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define KERNEL_SIZE 5

// Define a struct to hold the kernel and its dimensions
typedef struct {
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    int size;
} Kernel;

unsigned char* downsample(unsigned char* input, unsigned width, unsigned height);
unsigned char* to_grayscale(unsigned char* image, unsigned width, unsigned height);
unsigned char* apply_moving_filter(unsigned char *image, unsigned width, unsigned height, Kernel kernel);
void write_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
unsigned char* read_image(const char* filename, unsigned* width, unsigned* height);
double profiling_info(void (*func)(unsigned char*, unsigned, unsigned), unsigned char* image, unsigned width, unsigned height);
double profiling_info_filter(void (*func)(unsigned char*, unsigned, unsigned, Kernel), unsigned char* image, unsigned width, unsigned height, Kernel kernel);

int main(void)
{
    unsigned char *image, *downsampled_image, *gray_image, *filtered_image;
    unsigned width, height;
    double time_downsample, time_grayscale, time_filter;
    
    image = read_image("im0.png", &width, &height);
    
    // Sobel kernel
    int sobel_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {-1, -2, 0, 2, 1},
        {-2, -4, 0, 4, 2},
        {-3, -6, 0, 6, 3},
        {-2, -4, 0, 4, 2},
        {-1, -2, 0, 2, 1}
    };
    // Gaussian blur kernel
    float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1,  4,  6,  4,  1},
        {4, 16, 24, 16,  4},
        {6, 24, 36, 24,  6},
        {4, 16, 24, 16,  4},
        {1,  4,  6,  4,  1}
    };
    Kernel sobelKernel, gaussianKernel;

    sobelKernel.size = KERNEL_SIZE;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            sobelKernel.kernel[i][j] = sobel_kernel[i][j];
        }
    }

    gaussianKernel.size = KERNEL_SIZE;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            gaussianKernel.kernel[i][j] = gaussian_kernel[i][j];
        }
    }
    
    // Image processing
    // Downsample the image
    downsampled_image = downsample(image, width, height);
    // Convert the downsampled image to grayscale
    gray_image = to_grayscale(downsampled_image, width / 4, height / 4);
    // Filter the grayscale image
    filtered_image = apply_moving_filter(gray_image, width / 4, height / 4, gaussianKernel);
    
    
    // Profiling the function executions
    // Profile downsample function
    time_downsample = profiling_info(downsample, image, width, height);
    printf("Downsample function took %.10f seconds to execute.\n", time_downsample);

    // Profile to_grayscale function
    time_grayscale = profiling_info(to_grayscale, downsampled_image, width/4, height/4);
    printf("Grayscale function took %.10f seconds to execute.\n", time_grayscale);

    // Profile apply_moving_filter function
    time_filter = profiling_info_filter(apply_moving_filter, gray_image, width/4, height/4, sobelKernel); // or gaussianKernel
    printf("Filter function took %.10f seconds to execute.\n", time_filter);

    
    write_image("output_downsampled.png", downsampled_image, width/4, height/4);
    write_grayscale_image("output_grayscale.png", gray_image, width/4, height/4);
    write_grayscale_image("output_filtered.png", filtered_image, width/4, height/4);
    
    // Free the memory allocated by LodePNG for the images
    free(image);
    free(downsampled_image);
    free(gray_image);
    free(filtered_image);
    
    return 0;
}

unsigned char* downsample(unsigned char* image, unsigned width, unsigned height)
{
    unsigned new_width, new_height;
    unsigned char* resized_image;
    
    // Calculate new width and height for downscaled image (1/16 of original)
    new_width = width / 4; // Because we're scaling down by 4 in each dimension
    new_height = height / 4;
    
    // Allocate memory for resized image
    resized_image = (unsigned char*)malloc(new_width * new_height * 4);

    // Perform downsampling using nearest-neighbor interpolation
    for(unsigned y = 0; y < new_height; y++) {
        for(unsigned x = 0; x < new_width; x++) {
            unsigned idx_resized = (y * new_width + x) * 4;
            unsigned idx_original = ((y * 4) * width + (x * 4)) * 4;
            resized_image[idx_resized] = image[idx_original];
            resized_image[idx_resized + 1] = image[idx_original + 1];
            resized_image[idx_resized + 2] = image[idx_original + 2];
            resized_image[idx_resized + 3] = image[idx_original + 3];
        }
    }
    
    return resized_image;
}

unsigned char* to_grayscale(unsigned char* image, unsigned width, unsigned height)
{
    unsigned char* gray_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!gray_image) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for(unsigned y = 0; y < height; y++) {
        for(unsigned x = 0; x < width; x++) {
            unsigned idx = (y * width + x) * 4;
            unsigned char r = image[idx];
            unsigned char g = image[idx + 1];
            unsigned char b = image[idx + 2];
            unsigned char gray = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
            gray_image[y * width + x] = gray;
        }
    }

    return gray_image;
}


unsigned char* apply_moving_filter(unsigned char* image, unsigned width, unsigned height, Kernel kernel) {
    // Calculate the size of the padded image
    int paddedWidth = width + 2 * (kernel.size / 2);
    int paddedHeight = height + 2 * (kernel.size / 2);
    
    // Calculate kernel radius
    int kernelRadius = kernel.size / 2;

    // Allocate memory for the padded image
    unsigned char* paddedImage = (unsigned char*)malloc(paddedWidth * paddedHeight * sizeof(unsigned char));
    if (!paddedImage) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy the original image into the padded image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate the indices for the padded image
            int paddedX = x + kernelRadius;
            int paddedY = y + kernelRadius;
            // Copy the pixel value
            paddedImage[paddedY * paddedWidth + paddedX] = image[y * width + x];
        }
    }    



    // Apply the filter on the padded image
    unsigned char* filteredImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!filteredImage) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int y = kernelRadius; y < height + kernelRadius; y++) {
        for (int x = kernelRadius; x < width + kernelRadius; x++) {
            float sum = 0;
            for (int ky = 0; ky < kernel.size; ky++) {
                for (int kx = 0; kx < kernel.size; kx++) {
                    int imageX = x + kx - kernelRadius;
                    int imageY = y + ky - kernelRadius;
                    sum += paddedImage[imageY * paddedWidth + imageX] * kernel.kernel[ky][kx];
                }
            }
            // Normalize the sum to preserve brightness
            filteredImage[(y - kernelRadius) * width + (x - kernelRadius)] = (unsigned char)(sum / 256.0f); // Normalization by dividing by 256
        }
    }


    // Free the memory allocated for the padded image
    free(paddedImage);

    return filteredImage;
}


void write_image(const char* filename, const unsigned char* image, unsigned width, unsigned height)
{
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        printf("Failed to write image to file: %s\n", filename);
        exit(EXIT_FAILURE);
    } else {
        printf("Image written successfully: %s\n", filename);
    }
}

void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    // Allocate memory for the grayscale image
    unsigned char* grayscale_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!grayscale_image) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy the image data to the grayscale image
    memcpy(grayscale_image, image, width * height * sizeof(unsigned char));

    // Write the grayscale image to the file
    unsigned error = lodepng_encode_file(filename, grayscale_image, width, height, LCT_GREY, 8);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        printf("Failed to write image to file: %s\n", filename);
        exit(EXIT_FAILURE);
    } else {
        printf("Image written successfully: %s\n", filename);
    }

    // Free allocated memory
    free(grayscale_image);
}

unsigned char* read_image(const char* filename, unsigned* width, unsigned* height)
{
    unsigned error;
    unsigned char* image;

    error = lodepng_decode32_file(&image, width, height, filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(EXIT_FAILURE);
    }

    return image;
}

double profiling_info(void (*func)(unsigned char*, unsigned, unsigned), unsigned char* image, unsigned width, unsigned height)
{
    clock_t start, end;
    double cpu_time_used;

    // Start timing
    start = clock();

    // Execute the function
    func(image, width, height);

    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    return cpu_time_used;
}

double profiling_info_filter(void (*func)(unsigned char*, unsigned, unsigned, Kernel), unsigned char* image, unsigned width, unsigned height, Kernel kernel)
{
    clock_t start, end;
    double cpu_time_used;

    // Start timing
    start = clock();

    // Execute the function
    func(image, width, height, kernel);

    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    return cpu_time_used;
}