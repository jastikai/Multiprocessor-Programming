// Jaakko Astikainen 2024

#include "lodepng/lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <string.h>

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 5
float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1,  4,  6,  4,  1},
        {4, 16, 24, 16,  4},
        {6, 24, 36, 24,  6},
        {4, 16, 24, 16,  4},
        {1,  4,  6,  4,  1}
    };
float normalization_factor = 256.0;
#endif

// Define a struct to hold the kernel and its dimensions
typedef struct {
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    int size;
} Kernel;

typedef unsigned char uchar;

void write_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
unsigned char* read_image(const char* filename, unsigned* width, unsigned* height);
double profiling_info(cl_event event);

int main(void)
{
    unsigned char *image, *downsampled_image, *gray_image, *filtered_image;
    unsigned width, height;
    double time_downsample, time_grayscale, time_filter;
    
    // OpenCL variables
    cl_int           err;
    cl_uint          num_platforms;
    cl_platform_id  *platforms;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program_downsample, program_grayscale, program_filter;
    cl_kernel        kernel_downsample, kernel_grayscale, kernel_filter;
    cl_event         event_downsample, event_grayscale, event_filter;
    
    // setup kernels from source files
    char *kernel_source_downsample, *kernel_source_grayscale, *kernel_source_filter;
    size_t source_size, program_size;
    // *** DOWNSAMPLING SOURCE ***
    FILE *fp1 = fopen("downsample.cl", "rb");
    if (!fp1) {
        printf("Failed to load downsample kernel\n");
        return 1;
    }
    fseek(fp1, 0, SEEK_END);
    program_size = ftell(fp1);
    rewind(fp1);
    kernel_source_downsample = (char*)malloc(program_size + 1);
    kernel_source_downsample[program_size] = '\0';
    fread(kernel_source_downsample, sizeof(char), program_size, fp1);
    fclose(fp1);
    // *** GRAYSCALE SOURCE ***
    FILE *fp2 = fopen("grayscale.cl", "rb");
    if (!fp2) {
        printf("Failed to load grayscale kernel\n");
        return 1;
    }
    fseek(fp2, 0, SEEK_END);
    program_size = ftell(fp2);
    rewind(fp2);
    kernel_source_grayscale = (char*)malloc(program_size + 1);
    kernel_source_grayscale[program_size] = '\0';
    fread(kernel_source_grayscale, sizeof(char), program_size, fp2);
    fclose(fp2);
    // *** FILTER SOURCE ***
    FILE *fp3 = fopen("moving_filter.cl", "rb");
    if (!fp3) {
        printf("Failed to load filter kernel\n");
        return 1;
    }
    fseek(fp3, 0, SEEK_END);
    program_size = ftell(fp3);
    rewind(fp3);
    kernel_source_filter = (char*)malloc(program_size + 1);
    kernel_source_filter[program_size] = '\0';
    fread(kernel_source_filter, sizeof(char), program_size, fp3);
    fclose(fp3);
    // KERNEL SOURCE READING DONE -----------------------------------------------------------------------------
    
    // read input image
    image = read_image("im0.png", &width, &height);
    // for downsampled images
    unsigned new_width = width/4;
    unsigned new_height = height/4;
    
    // setup platform
    // PLATFORM
    int num_max_platforms = 1;
    err = clGetPlatformIDs(num_max_platforms, NULL, &num_platforms);
    printf("Num platforms detected: %d\n", num_platforms);

    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_max_platforms, platforms, &num_platforms);

    if(num_platforms < 1)
    {
        printf("No platform detected, exit\n");
        exit(1);
    }
    // platform set up -----------------------------------------------------------------------------------------
    
    // setup device, context and queue
    //DEVICE (could be CL_DEVICE_TYPE_GPU)
    //
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    //CONTEXT
    //
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    //QUEUE
    //
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    // device, context and queue set up ------------------------------------------------------------------------
    
    // read kernels and compile
    // downsampling program
    program_downsample = clCreateProgramWithSource(context, 1, &kernel_source_downsample, NULL, &err);
    err = clBuildProgram(program_downsample, 0, NULL, NULL, NULL, NULL);
    // grayscale program
    program_grayscale = clCreateProgramWithSource(context, 1, &kernel_source_grayscale, NULL, &err);
    err = clBuildProgram(program_grayscale, 0, NULL, NULL, NULL, NULL);
    // filtering program
    program_filter = clCreateProgramWithSource(context, 1, &kernel_source_filter, NULL, &err);
    err = clBuildProgram(program_filter, 0, NULL, NULL, NULL, NULL);
    // kernel compilation done --------------------------------------------------------------------------------
    
    // Create and set kernel arguments
    // downsampling kernel
    kernel_downsample = clCreateKernel(program_downsample, "downsample_kernel", &err);
    cl_mem buffer_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar)*width*height*4, NULL, NULL);
    cl_mem buffer_output1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar)*new_width*new_height*4, NULL, &err);
    clSetKernelArg(kernel_downsample, 0, sizeof(cl_mem), &buffer_input1);
    clSetKernelArg(kernel_downsample, 1, sizeof(cl_mem), &buffer_output1);
    clSetKernelArg(kernel_downsample, 2, sizeof(unsigned), &width);
    clSetKernelArg(kernel_downsample, 3, sizeof(unsigned), &height);

    // grayscale kernel
    kernel_grayscale = clCreateKernel(program_grayscale, "to_grayscale_kernel", &err);
    cl_mem buffer_input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar)*new_width*new_height*4, NULL, NULL);
    cl_mem buffer_output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar)*new_width*new_height, NULL, &err);
    clSetKernelArg(kernel_grayscale, 0, sizeof(cl_mem), &buffer_input2);
    clSetKernelArg(kernel_grayscale, 1, sizeof(cl_mem), &buffer_output2);
    clSetKernelArg(kernel_grayscale, 2, sizeof(unsigned), &new_width);
    clSetKernelArg(kernel_grayscale, 3, sizeof(unsigned), &new_height);

    // filter kernel
    kernel_filter = clCreateKernel(program_filter, "apply_gaussian_blur_kernel", &err);
    cl_mem buffer_input3= clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar)*new_width*new_height, NULL, NULL);
    cl_mem buffer_output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar)*new_width*new_height, NULL, &err);
    clSetKernelArg(kernel_filter, 0, sizeof(cl_mem), &buffer_input3);
    clSetKernelArg(kernel_filter, 1, sizeof(cl_mem), &buffer_output3);
    clSetKernelArg(kernel_filter, 2, sizeof(unsigned), &new_width);
    clSetKernelArg(kernel_filter, 3, sizeof(unsigned), &new_height);

    // kernel arguments set ---------------------------------------------------------------------------------------
    
    // run downsampling kernel
   
    downsampled_image = (unsigned char*)malloc(sizeof(uchar) * new_width * new_height * 4);
    // enqueue write buffer for downsampling
    clEnqueueWriteBuffer(queue, buffer_input1, CL_TRUE, 0, sizeof(uchar)*width*height*4, image, 0, NULL, &event_downsample);
    
    // run kernel and fill read buffer
    size_t global_work_size[2] = {new_width, new_height};  // Global work size covering entire output image
    size_t local_work_size[2] = {1, 1};  // Local work size, one work item per work group
    err = clEnqueueNDRangeKernel(queue, kernel_downsample, 2, NULL, global_work_size, local_work_size, 0, NULL, &event_downsample);
    err = clEnqueueReadBuffer(queue, buffer_output1, CL_TRUE, 0, sizeof(uchar)*new_width*new_height*4, downsampled_image, 0, NULL, &event_downsample);
    clFinish(queue);
    
    // get profiling info from downsampling
    double kernel_exec_time1 = profiling_info(event_downsample);
    
    printf("Downsampling kernel execution time: %.10f seconds\n", kernel_exec_time1);
    write_image("output_downsampled.png", downsampled_image, new_width, new_height);
    // downsampling done -------------------------------------------------------------------------------------------
    
    // run grayscale kernel
    gray_image = (unsigned char*)malloc(sizeof(uchar) * new_width * new_height);
    clEnqueueWriteBuffer(queue, buffer_input2, CL_TRUE, 0, sizeof(uchar)*new_width*new_height*4, downsampled_image, 0, NULL, &event_grayscale);
    err = clEnqueueNDRangeKernel(queue, kernel_grayscale, 2, NULL, global_work_size, local_work_size, 0, NULL, &event_grayscale);
    err = clEnqueueReadBuffer(queue, buffer_output2, CL_TRUE, 0, sizeof(uchar)*new_width*new_height, gray_image, 0, NULL, &event_grayscale);
    clFinish(queue);

    // get profiling info from grayscale
    double kernel_exec_time2 = profiling_info(event_grayscale);
    printf("Grayscale kernel execution time: %.10f seconds\n", kernel_exec_time2);

    // Write the grayscale image to file
    write_grayscale_image("output_grayscale.png", gray_image, new_width, new_height);
    // grayscaling done --------------------------------------------------------------------------------------------
    
    // run filtering kernel
    filtered_image = (unsigned char*)malloc(sizeof(uchar) * new_width * new_height);
    clEnqueueWriteBuffer(queue, buffer_input3, CL_TRUE, 0, sizeof(uchar)*new_width*new_height, gray_image, 0, NULL, &event_filter);
    err = clEnqueueNDRangeKernel(queue, kernel_filter, 2, NULL, global_work_size, local_work_size, 0, NULL, &event_filter);
    err = clEnqueueReadBuffer(queue, buffer_output3, CL_TRUE, 0, sizeof(uchar)*new_width*new_height, filtered_image, 0, NULL, &event_filter);
    clFinish(queue);

    // get profiling info from grayscale
    double kernel_exec_time3 = profiling_info(event_filter);
    printf("Filtering kernel execution time: %.10f seconds\n", kernel_exec_time3);

    // Write the filtered image to file
    write_grayscale_image("output_filtered.png", filtered_image, new_width, new_height);
    // filtering done ----------------------------------------------------------------------------------------------
    
    // free memory
    free(image);
    free(downsampled_image);
    free(gray_image);
    free(filtered_image);
    clReleaseMemObject(buffer_input1);
    clReleaseMemObject(buffer_output1);
    clReleaseMemObject(buffer_input2);
    clReleaseMemObject(buffer_output2);
    clReleaseMemObject(buffer_input3);
    clReleaseMemObject(buffer_output3);
    clReleaseKernel(kernel_downsample);
    clReleaseKernel(kernel_grayscale);
    clReleaseKernel(kernel_filter);
    clReleaseProgram(program_downsample);
    clReleaseProgram(program_grayscale);
    clReleaseProgram(program_filter);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseEvent(event_downsample);
    clReleaseEvent(event_grayscale);
    clReleaseEvent(event_filter);
    
    return 0;
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

double profiling_info(cl_event event)
{
    cl_ulong start, end;
    cl_int err;
    
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get profiling information\n");
    }
    // Calculate kernel execution time
    double kernel_exec_time = (end - start) * 1.0e-9; // Convert nanoseconds to seconds
    
    return kernel_exec_time;
}