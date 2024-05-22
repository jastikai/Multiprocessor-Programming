// Jaakko Astikainen 2024

#include "lodepng/lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "project.h"

// Define a struct to hold the kernel and its dimensions
typedef struct {
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    int size;
} Kernel;
// Define struct to pass data to threads if needed
typedef struct {
    unsigned char* left_image;
    unsigned char* right_image;
    unsigned char* disparity_map;
    int width;
    int height;
    int direction;
    int start_row;
    int end_row;
} ThreadData;

typedef struct {
    unsigned char* input_image;
    unsigned char* output_image;
    int width;
    int height;
    int direction;
    int start_row;
    int end_row;
} ThreadDataSimple;

unsigned char* downsample(unsigned char* input, unsigned width, unsigned height);
unsigned char* to_grayscale(unsigned char* image, unsigned width, unsigned height);
unsigned char* apply_moving_filter(unsigned char *image, unsigned width, unsigned height, Kernel kernel);
void write_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
unsigned char* read_image(const char* filename, unsigned* width, unsigned* height);
float compute_zncc(const unsigned char* left_patch, const unsigned char* right_patch, int size);
void* create_disparity_map_thread(void* arg);
unsigned char* create_disparity_map(unsigned char* left_image, unsigned char* right_image, unsigned width, unsigned height, int direction);
unsigned char* cross_checking(unsigned char* left_disparity_map, unsigned char* right_disparity_map, unsigned width, unsigned height, unsigned threshold);
void occlusion_filling(unsigned char* disparity_map, unsigned width, unsigned height);
void* occlusion_filling_thread(void* arg);
int find_nearest_non_zero(unsigned char* disparity_map, unsigned width, unsigned height, int x, int y, int radius);
int max(int a, int b);
int min(int a, int b);
int is_queue_empty(int front, int rear);
void enqueue(int* queue_x, int* queue_y, int* rear, int x, int y);
void dequeue(int* queue_x, int* queue_y, int* front, int* x, int* y);
void discardBorderPixels(unsigned char* image, int width, int height, int border);

int main(void)
{
    unsigned char *left_image, *right_image;
    unsigned char *resized_left, *resized_right;
    unsigned char *gray_left, *gray_right;
    unsigned width1, height1;
    unsigned width2, height2;
    
    
    left_image = read_image("im0.png", &width1, &height1);
    right_image = read_image("im1.png", &width2, &height2);
    
    unsigned new_width = width1/DOWN_RATIO;
    unsigned new_height = height1/DOWN_RATIO;
    
    clock_t start, end;
    double cpu_time_used;

    // Start timing
    start = clock();

    
    // downsample
    resized_left = downsample(left_image, width1, height1);
    resized_right = downsample(right_image, width2, height2);
    
    // to grayscale
    gray_left = to_grayscale(resized_left, new_width, new_height);
    gray_right = to_grayscale(resized_right, new_width, new_height);
    
    // disparity maps
    unsigned char *left_disparity_map = (unsigned char*)(malloc(sizeof(unsigned char)*new_height*new_width));
    unsigned char *right_disparity_map = (unsigned char*)(malloc(sizeof(unsigned char)*new_height*new_width));
    
    
    
    
    //    MULTI-THREADING DISPARITY -------------------------------------------------------------------------

    // Create an array of pthreads and ThreadData structs
    pthread_t threads[THREADS];
    ThreadData thread_data[THREADS];

    // Divide the workload for left and right disparity maps among separate sets of threads
    int rows_per_thread = new_height / THREADS_PER_IM;
    for (int i = 0; i < THREADS_PER_IM; ++i) {
    // Thread data for left disparity map
    thread_data[i].left_image = gray_left;
    thread_data[i].right_image = gray_right;
    thread_data[i].disparity_map = left_disparity_map;
    thread_data[i].width = new_width;
    thread_data[i].height = new_height;
    thread_data[i].direction = 0; // Left disparity map
    thread_data[i].start_row = i * rows_per_thread;
    thread_data[i].end_row = (i == THREADS_PER_IM - 1) ? new_height : (i + 1) * rows_per_thread;
    
    // Thread data for right disparity map
    thread_data[THREADS_PER_IM + i].left_image = gray_right;
    thread_data[THREADS_PER_IM + i].right_image = gray_left;
    thread_data[THREADS_PER_IM + i].disparity_map = right_disparity_map;
    thread_data[THREADS_PER_IM + i].width = new_width;
    thread_data[THREADS_PER_IM + i].height = new_height;
    thread_data[THREADS_PER_IM + i].direction = 1; // Right disparity map
    thread_data[THREADS_PER_IM + i].start_row = i * rows_per_thread;
    thread_data[THREADS_PER_IM + i].end_row = (i == THREADS_PER_IM - 1) ? new_height : (i + 1) * rows_per_thread;
    }
    
    // Create threads for left disparity map
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        pthread_create(&threads[i], NULL, create_disparity_map_thread, &thread_data[i]);
    }

    // Create threads for right disparity map
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        pthread_create(&threads[THREADS_PER_IM + i], NULL, create_disparity_map_thread, &thread_data[THREADS_PER_IM + i]);
    }

    // Wait for all threads to finish processing left disparity map
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Wait for all threads to finish processing right disparity map
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        pthread_join(threads[THREADS_PER_IM + i], NULL);
    }

    discardBorderPixels(left_disparity_map, new_width, new_height, KERNEL_RADIUS);
    discardBorderPixels(right_disparity_map, new_width, new_height, KERNEL_RADIUS);
    
    write_grayscale_image("left_disp.png", left_disparity_map, new_width, new_height);
    write_grayscale_image("right_disp.png", right_disparity_map, new_width, new_height);
    
    // ------------------------------------------------------------------------------------------------------
    // cross-checking
    unsigned char *consolidated_map = cross_checking(left_disparity_map, right_disparity_map, new_width, new_height, THRESHOLD);
    write_grayscale_image("consolidated.png", consolidated_map, new_width, new_height);
    
    // occlusion filling
    // Multi-threading for occlusion filling
    pthread_t occlusion_threads[THREADS_PER_IM];
    ThreadDataSimple occlusion_data[THREADS_PER_IM];
    int rows_per_thread_occlusion = new_height / THREADS_PER_IM;

    // Create and join threads for occlusion filling
    unsigned char *filled_map = (unsigned char*)(malloc(sizeof(unsigned char)*new_height*new_width));
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        occlusion_data[i].input_image = consolidated_map;
        occlusion_data[i].output_image = filled_map;
        occlusion_data[i].width = new_width;
        occlusion_data[i].height = new_height;
        occlusion_data[i].direction = 0;
        occlusion_data[i].start_row = i * rows_per_thread_occlusion;
        occlusion_data[i].end_row = (i == THREADS_PER_IM - 1) ? new_height : (i + 1) * rows_per_thread_occlusion;
        pthread_create(&occlusion_threads[i], NULL, occlusion_filling_thread, &occlusion_data[i]);
    }
    for (int i = 0; i < THREADS_PER_IM; ++i) {
        pthread_join(occlusion_threads[i], NULL);
    }

    // occlusion_filling(consolidated_map, new_width, new_height);
    // discardBorderPixels(filled_map, new_width, new_height, KERNEL_RADIUS);
    write_grayscale_image("depthmap.png", filled_map, new_width, new_height);
    
    
    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Processing time: %.10f seconds\n", cpu_time_used);
    
    
    
    
    // Free allocated memory
    free(left_disparity_map);
    free(right_disparity_map);
    free(consolidated_map);
    free(left_image);
    free(right_image);
    free(resized_left);
    free(resized_right);
    free(gray_left);
    free(gray_right);
    free(filled_map);
    
    return 0;
}

unsigned char* downsample(unsigned char* image, unsigned width, unsigned height)
{
    unsigned new_width, new_height;
    unsigned char* resized_image;
    
    // Calculate new width and height for downscaled image
    new_width = width / DOWN_RATIO;
    new_height = height / DOWN_RATIO;
    
    // Allocate memory for resized image
    resized_image = (unsigned char*)malloc(new_width * new_height * 4); // Each pixel has 4 channels

    // Perform downsampling using nearest-neighbor interpolation
    for(unsigned y = 0; y < new_height; y++) {
        for(unsigned x = 0; x < new_width; x++) {
            unsigned idx_resized = (y * new_width + x) * 4; // Each pixel has 4 channels
            unsigned idx_original = ((y * DOWN_RATIO) * width + (x * DOWN_RATIO)) * 4; // Each pixel has 4 channels
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

float compute_zncc(const unsigned char* left_patch, const unsigned char* right_patch, int size) {
    float mean_left = 0, mean_right = 0, std_left = 0, std_right = 0, cross_corr = 0;
    
    // Compute means
    for (int i = 0; i < size; i++) {
        mean_left += left_patch[i];
        mean_right += right_patch[i];
    }
    mean_left /= size;
    mean_right /= size;

    // Compute standard deviations and cross-correlation
    for (int i = 0; i < size; i++) {
        float diff_left = left_patch[i] - mean_left;
        float diff_right = right_patch[i] - mean_right;
        std_left += diff_left * diff_left;
        std_right += diff_right * diff_right;
        cross_corr += diff_left * diff_right;
    }
    std_left = sqrt(std_left / size);
    std_right = sqrt(std_right / size);
    // cross_corr /= size;
    
    // Compute ZNCC score
    float zncc_score = cross_corr / (std_left * std_right);
    return zncc_score;
}

unsigned char* create_disparity_map(unsigned char* left_image, unsigned char* right_image, unsigned width, unsigned height, int direction) {
    int kernel_radius = KERNEL_SIZE / 2;
    int scaled_ndisp = 260 / DOWN_RATIO;
    unsigned char *disparity_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *left_patch = (unsigned char*)(malloc(sizeof(unsigned char)*KERNEL_SIZE*KERNEL_SIZE));
    unsigned char *right_patch = (unsigned char*)(malloc(sizeof(unsigned char)*KERNEL_SIZE*KERNEL_SIZE));
    // unsigned char* normalized_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    
    // Iterate over all pixels, excluding border pixels
    for (int y = kernel_radius; y < height - kernel_radius; y++) {
        for (int x = kernel_radius; x < width - kernel_radius; x++) {
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
                    for (int i = 0; i < KERNEL_SIZE; i++) {
                        for (int j = 0; j < KERNEL_SIZE; j++) {
                            // Check if the current patch indices are within the image boundaries
                            // if (y - kernel_radius + i >= 0 && y - kernel_radius + i < height &&
                            //     x - kernel_radius + j >= 0 && x - kernel_radius + j < width &&
                            //     x - kernel_radius + j - d >= 0 && x - kernel_radius + j - d < width) {
                                left_patch[i * KERNEL_SIZE + j] = left_image[(y - kernel_radius + i) * width + (x - kernel_radius + j)];
                                if (direction) {
                                    right_patch[i * KERNEL_SIZE + j] = right_image[(y - kernel_radius + i) * width + (x - kernel_radius + j + d)];
                                }
                                else {
                                    right_patch[i * KERNEL_SIZE + j] = right_image[(y - kernel_radius + i) * width + (x - kernel_radius + j - d)];
                                }
                            // }
                        }
                    }
                    // Compute ZNCC score
                    float zncc = compute_zncc(left_patch, right_patch, KERNEL_SIZE * KERNEL_SIZE);
                    // Update best disparity if needed
                    if (zncc > best_zncc) {
                        best_zncc = zncc;
                        best_disparity = d;
                    }
                }
            }
            // Assign disparity to pixel
            disparity_map[y * width + x] = best_disparity;
        }
    }
    free(left_patch);
    free(right_patch);
    
    // for (int i = 0; i < width * height; i++) {
    //     normalized_map[i] = (unsigned char)((disparity_map[i] / (float)scaled_ndisp) * 255);
    // }
    
    // return normalized_map;
    return disparity_map;
}

void* create_disparity_map_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // Extract relevant data from ThreadData struct
    unsigned char* left_image = data->left_image;
    unsigned char* right_image = data->right_image;
    unsigned char* disparity_map = data->disparity_map;
    int width = data->width;
    int height = data->height;
    int direction = data->direction;
    int start_row = data->start_row;
    int end_row = data->end_row;
    
    
    int kernel_radius = KERNEL_SIZE / 2;
    int scaled_ndisp = 260 / DOWN_RATIO;
    unsigned char *left_patch = (unsigned char*)(malloc(sizeof(unsigned char)*KERNEL_SIZE*KERNEL_SIZE));
    unsigned char *right_patch = (unsigned char*)(malloc(sizeof(unsigned char)*KERNEL_SIZE*KERNEL_SIZE));
    // unsigned char* normalized_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    // Adjust stereo matching direction based on input parameter
    int start_disparity = 0;//(direction == 0) ? 0 : scaled_ndisp - 1;
    int end_disparity = scaled_ndisp;//(direction == 0) ? scaled_ndisp : -1;
    int step = 1;//(direction == 0) ? 1 : -1;
    
    if (start_row < kernel_radius)
        start_row = kernel_radius;
    if (end_row > (height-kernel_radius))
        end_row = height-kernel_radius;
    
    // Iterate over all pixels, excluding border pixels
    for (int y = start_row; y < end_row; y++) {
        for (int x = kernel_radius; x < width - kernel_radius; x++) {
            int best_disparity = 0;
            float best_zncc = -1;
            
            // Iterate over disparities
            for (int d = start_disparity; d != end_disparity; d += step) {
                // Check if the current disparity allows the kernel to fit within the image boundaries
                if (((!direction && x >= kernel_radius + d) || (direction && x >= kernel_radius)) && ((!direction && x < width - kernel_radius ) || (direction && x < width - kernel_radius - d)) && y >= kernel_radius && y < height - kernel_radius) {
                    // Extract patches from left and right images
                    for (int i = 0; i < KERNEL_SIZE; i++) {
                        for (int j = 0; j < KERNEL_SIZE; j++) {
                            // Check if the current patch indices are within the image boundaries
                            // if (y - kernel_radius + i >= 0 && y - kernel_radius + i < height &&
                            //     x - kernel_radius + j >= 0 && x - kernel_radius + j < width &&
                            //     x - kernel_radius + j - d >= 0 && x - kernel_radius + j - d < width) {
                                left_patch[i * KERNEL_SIZE + j] = left_image[(y - kernel_radius + i) * width + (x - kernel_radius + j)];
                                if (direction) {
                                    right_patch[i * KERNEL_SIZE + j] = right_image[(y - kernel_radius + i) * width + (x - kernel_radius + j + d)];
                                }
                                else {
                                    right_patch[i * KERNEL_SIZE + j] = right_image[(y - kernel_radius + i) * width + (x - kernel_radius + j - d)];
                                }
                            // }
                        }
                    }
                    // Compute ZNCC score
                    float zncc = compute_zncc(left_patch, right_patch, KERNEL_SIZE * KERNEL_SIZE);
                    // Update best disparity if needed
                    if (zncc > best_zncc) {
                        best_zncc = zncc;
                        best_disparity = d;
                    }
                }
            }
            // Assign disparity to pixel
            disparity_map[y * width + x] = best_disparity;
        }
    }
    free(left_patch);
    free(right_patch);
    // free(normalized_map);
    
    // normalize result
    // for (int y = start_row; y < end_row; y++) {
    //     for (int x = 0; x < width; x++) {
    //         int index = y * width + x;
    //         disparity_map[index] = (unsigned char)((disparity_map[index] / (float)scaled_ndisp) * 255);
    //     }
    // }

}


unsigned char* cross_checking(unsigned char* left_disparity_map, unsigned char* right_disparity_map, unsigned width, unsigned height, unsigned threshold) {
    unsigned char* consolidated_disparity_map = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    int scaled_ndisp = 260 / DOWN_RATIO;
    if (!consolidated_disparity_map) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }


    for (int i = 0; i < width * height; i++) {
        if (abs((float)(left_disparity_map[i] - right_disparity_map[i - left_disparity_map[i]])) * 255 / scaled_ndisp > threshold) {
            // If the absolute difference is larger than the threshold, set the pixel value to zero
            consolidated_disparity_map[i] = 0;
        } else {
            // Otherwise, retain the pixel value from the left disparity map
            consolidated_disparity_map[i] = (unsigned char)((int)left_disparity_map[i] * 255 / scaled_ndisp);
        }
    }

    return consolidated_disparity_map;
}


void occlusion_filling(unsigned char* disparity_map, unsigned width, unsigned height) {
    // Create a queue to store pixel coordinates
    int queue_capacity = width * height;
    int queue_front = 0;
    int queue_rear = -1;
    int* queue_x = (int*)malloc(queue_capacity * sizeof(int));
    int* queue_y = (int*)malloc(queue_capacity * sizeof(int));
    
    // Initialize the queue with known disparity pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (disparity_map[y * width + x] != 0) {
                enqueue(queue_x, queue_y, &queue_rear, x, y);
            }
        }
    }
    
    // Process pixels in the queue
    while (!is_queue_empty(queue_front, queue_rear)) {
        int current_x, current_y;
        dequeue(queue_x, queue_y, &queue_front, &current_x, &current_y);
        
        // Process neighbors of the current pixel
        for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; dy++) {
            for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; dx++) {
                int nx = current_x + dx;
                int ny = current_y + dy;
                // Check if the neighbor is within the image bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Check if the neighbor has an occluded disparity
                    if (disparity_map[ny * width + nx] == 0) {
                        // Fill occlusion using nearest non-zero neighbor
                        int nearest_non_zero = find_nearest_non_zero(disparity_map, width, height, nx, ny, MAX_RADIUS);
                        if (nearest_non_zero != -1) {
                            disparity_map[ny * width + nx] = nearest_non_zero;
                            // Add the filled pixel to the queue for further processing
                            enqueue(queue_x, queue_y, &queue_rear, nx, ny);
                        }
                    }
                }
            }
        }
    }
    
    // Free the memory allocated for the queue
    free(queue_x);
    free(queue_y);
}

void* occlusion_filling_thread(void* arg) {
    ThreadDataSimple* data = (ThreadDataSimple*)arg;

    // Extract relevant data from ThreadDataSimple struct
    unsigned char* disparity_map = data->input_image;
    unsigned char* filled_map = data->output_image;
    unsigned int width = data->width;
    unsigned int height = data->height;
    unsigned int start_row = data->start_row;
    unsigned int end_row = data->end_row;

    // Perform occlusion filling for the specified rows
    for (unsigned int y = start_row; y < end_row; y++) {
        for (unsigned int x = 0; x < width; x++) {
            // Perform occlusion filling for the current pixel
            if (disparity_map[y * width + x] == 0) {
                int nearest_non_zero = find_nearest_non_zero(disparity_map, width, height, x, y, MAX_RADIUS);
                if (nearest_non_zero != -1) {
                    filled_map[y * width + x] = nearest_non_zero;
                }
                else {
                    filled_map[y * width + x] = 0;
                }
            }
            else {
                filled_map[y * width + x] = disparity_map[y * width + x];
            }
        }
    }
}

// Function to check if the queue is empty
int is_queue_empty(int front, int rear) {
    return (rear < front);
}

// Function to add an element to the queue
void enqueue(int* queue_x, int* queue_y, int* rear, int x, int y) {
    (*rear)++;
    queue_x[*rear] = x;
    queue_y[*rear] = y;
}

// Function to remove an element from the queue
void dequeue(int* queue_x, int* queue_y, int* front, int* x, int* y) {
    *x = queue_x[*front];
    *y = queue_y[*front];
    (*front)++;
}



int find_nearest_non_zero(unsigned char* disparity_map, unsigned width, unsigned height, int x, int y, int radius) {
    // Calculate the boundaries of the search area
    int x_min = max(0, x - radius);
    int x_max = min(width - 1, x + radius);
    int y_min = max(0, y - radius);
    int y_max = min(height - 1, y + radius);
    int nearest = -1;
    unsigned int distance = UINT_MAX;

    // Search for the nearest non-zero disparity value
    for (int j = y_min; j <= y_max; j++) {
        for (int i = x_min; i <= x_max; i++) {
            if (disparity_map[j * width + i] != 0 && abs(x - i) + abs(y - j) < distance) {
                nearest = disparity_map[j * width + i];
                distance = abs(x - i) + abs(y - j);
            }
        }
    }

    return nearest; // Return -1 if no non-zero disparity value is found in the search area
}

// Function to find the maximum of two integers
int max(int a, int b) {
    return (a > b) ? a : b;
}

// Function to find the minimum of two integers
int min(int a, int b) {
    return (a < b) ? a : b;
}

void discardBorderPixels(unsigned char* image, int width, int height, int border)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (x < border || y < border || x >= (width - border) || y >= (height-border))
                image[y*width + x] = 0;
        }
    }
}