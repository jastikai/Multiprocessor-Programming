// Jaakko Astikainen 2024
// Include necessary headers
#include "lodepng/lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <string.h>
#include <stdbool.h>
#include "project.h"

cl_program load_kernel(const char* kernel_file, cl_context context, cl_device_id device);
cl_kernel create_kernel(cl_program program, const char* kernel_name);
void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height);
unsigned char* read_image(const char* filename, unsigned* width, unsigned* height);
double profiling_info(cl_event event);
cl_mem createImageObjectFromData(cl_context context, unsigned char* imageData, unsigned int width, unsigned int height);
void debug_write_image(const char* filename, cl_context context, cl_command_queue queue, cl_mem buffer, unsigned int width, unsigned int height);

int main(void)
{
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id devices[2];
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel downsample_kernel_left, downsample_kernel_right, grayscale_kernel_left, grayscale_kernel_right, disparity_kernel_left, disparity_kernel_right, cross_check_kernel, occlusion_fill_kernel;
    int num_platforms;
    cl_platform_id* platforms;
    cl_int err = CL_SUCCESS;
    cl_int ret;

    // Image variables
    unsigned char *left_image, *right_image;
    unsigned width, height;
    

    // Read images
    left_image = read_image("im0.png", &width, &height);
    right_image = read_image("im1.png", &width, &height);
    
    unsigned new_width = width/DOWN_RATIO;
    unsigned new_height= height/DOWN_RATIO;
    unsigned char *depth_map = (unsigned char*)(malloc(sizeof(unsigned char) * new_height * new_width));
    int scaled_ndisp = 260 / DOWN_RATIO;

        
    //---------------------------------------------------------------------------------------------------------
    // Set up OpenCL environment
    // Initialize platform, device, context, and command queue
    int num_max_platforms = 2;
    err = clGetPlatformIDs(num_max_platforms, NULL, &num_platforms);
    printf("Num platforms detected: %d\n", num_platforms);

    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_max_platforms, platforms, &num_platforms);

    if(num_platforms < 1)
    {
        printf("No platform detected, exit\n");
        exit(1);
    }
    cl_uint num_devices = 2;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 2, devices, &num_devices);
    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    device = devices[0];
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    
    // Get device info
    cl_device_local_mem_type local_mem_type;
    cl_ulong local_mem_size;
    cl_uint max_compute_units;
    cl_uint max_clock_frequency;
    cl_ulong max_constant_buffer_size;
    size_t max_work_group_size;
    size_t max_work_item_sizes[3];
    cl_uint im_support;
    char platform_name[1024];
    char device_name[1024];
    
    bool nvidia_flag = false;
    
    // Check device image support
    ret = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_uint), NULL, NULL);
    if (ret != CL_SUCCESS )
    {
        printf("Device does not support image format\n");
        nvidia_flag = true;
    }
    
    // Get platform name
    ret = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting platform name\n");
        return 1;
    }
    
    // Get device name
    ret = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device name\n");
        return 1;
    }

    // Get device local memory type
    ret = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &local_mem_type, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device local memory type\n");
        return 1;
    }

    // Get device local memory size
    ret = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device local memory size\n");
        return 1;
    }

    // Get device max compute units
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device max compute units\n");
        return 1;
    }

    // Get device max clock frequency
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_frequency, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device max clock frequency\n");
        return 1;
    }

    // Get device max constant buffer size
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &max_constant_buffer_size, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device max constant buffer size\n");
        return 1;
    }

    // Get device max work group size
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device max work group size\n");
        return 1;
    }

    // Get device max work item sizes
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, max_work_item_sizes, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting device max work item sizes\n");
        return 1;
    }

    // Print out the device info
    printf("Using OpenCL Platform : %s\n", platform_name);
    printf("Using GPU Device : %s\n", device_name);
    printf("Device Local Memory Type: %d\n", local_mem_type);
    printf("Device Local Memory Size: %llu bytes\n", local_mem_size);
    printf("Device Max Compute Units: %u\n", max_compute_units);
    printf("Device Max Clock Frequency: %u MHz\n", max_clock_frequency);
    printf("Device Max Constant Buffer Size: %llu bytes\n", max_constant_buffer_size);
    printf("Device Max Work Group Size: %zu\n", max_work_group_size);
    printf("Device Max Work Item Sizes: %zu, %zu, %zu\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);
    
    
    size_t global_work_size[2];
    size_t local_work_size[2];

    // Set local work size based on the device's max work group size
    if (nvidia_flag)
    {
        local_work_size[0] = LOCAL_WORK_SIZE_NVIDIA; 
        local_work_size[1] = LOCAL_WORK_SIZE_NVIDIA;
    }
    else
    {
        local_work_size[0] = LOCAL_WORK_SIZE_INTEL; 
        local_work_size[1] = LOCAL_WORK_SIZE_INTEL;
    }

    // Ensure local work size does not exceed device's max work group size
    if (local_work_size[0] > max_work_group_size) {
        local_work_size[0] = max_work_group_size;
    }
    if (local_work_size[1] > max_work_group_size) {
        local_work_size[1] = max_work_group_size;
    }

    // Ensure global work size is a multiple of local work size
    global_work_size[0] = ((width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = ((height + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];
    printf("Local work sizes: %d %d\n", local_work_size[0], local_work_size[1]);
    printf("Global work sizes: %d %d\n", global_work_size[0], global_work_size[1]);
    //---------------------------------------------------------------------------------------------------------
    // Load kernels
    if (nvidia_flag)
        program = load_kernel("project_kernel_nvidia.cl", context, device);
    else
        program = load_kernel("project_kernel.cl", context, device);

    // Create kernels
    downsample_kernel_left = create_kernel(program, "downsample");
    downsample_kernel_right = create_kernel(program, "downsample");
    grayscale_kernel_left = create_kernel(program, "to_grayscale");
    grayscale_kernel_right = create_kernel(program, "to_grayscale");
    disparity_kernel_left = create_kernel(program, "compute_disparity_map");
    disparity_kernel_right = create_kernel(program, "compute_disparity_map");
    cross_check_kernel = create_kernel(program, "cross_checking");
    occlusion_fill_kernel = create_kernel(program, "occlusion_filling");

    // Allocate memory buffers on the GPU
    cl_mem left_image_buffer, right_image_buffer;
    cl_mem left_downsample_buffer, right_downsample_buffer;
    cl_mem left_gray_buffer, right_gray_buffer;
    cl_mem left_disparity_map_buffer, right_disparity_map_buffer;
    cl_mem cross_check_buffer;
    cl_mem occlusion_map_buffer;

    // Create OpenCL buffers for processing
    // left_downsample_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                                    sizeof(unsigned char) * new_width * new_height * 4, NULL, &err);
    // right_downsample_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                                     sizeof(unsigned char) * new_width * new_height * 4, NULL, &err);
    left_gray_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(unsigned char) * new_width * new_height, NULL, &err);
    right_gray_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(unsigned char) * new_width * new_height, NULL, &err);
    left_disparity_map_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(unsigned char) * new_width * new_height, NULL, &err);
    right_disparity_map_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(unsigned char) * new_width * new_height, NULL, &err);
    cross_check_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(unsigned char) * new_width * new_height, NULL, &err);
    occlusion_map_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      sizeof(unsigned char) * new_width * new_height, NULL, &err);
                                                               
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to create OpenCL buffers: %d\n", err);
        exit(EXIT_FAILURE);
    }
    //---------------------------------------------------------------------------------------------------------
    // Transfer images from host to GPU memory
    // Enqueue commands to transfer left_image and right_image to GPU buffers
    
    unsigned char* downsampled_left = (unsigned char*)(malloc(sizeof(unsigned char)*new_height*new_width*4));
    unsigned char* downsampled_right = (unsigned char*)(malloc(sizeof(unsigned char)*new_height*new_width*4));

    // Create image objects for input and downsampled images
    if (!nvidia_flag)
    {
        left_image_buffer = createImageObjectFromData(context, left_image, width, height);
	    right_image_buffer = createImageObjectFromData(context, right_image, width, height);
	    left_downsample_buffer = createImageObjectFromData(context, downsampled_left, new_width, new_height);
	    right_downsample_buffer = createImageObjectFromData(context, downsampled_right, new_width, new_height);
    }
    else
    {
        left_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(unsigned char) * width * height * 4, NULL, &err);
        right_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(unsigned char) * width * height * 4, NULL, &err);
        left_downsample_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(unsigned char) * new_width * new_height * 4, NULL, &err);
        right_downsample_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(unsigned char) * new_width * new_height * 4, NULL, &err);
        clEnqueueWriteBuffer(queue, left_image_buffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * 4, left_image, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, right_image_buffer, CL_TRUE, 0, sizeof(unsigned char) * width * height * 4, right_image, 0, NULL, NULL);
    }
    
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to create image buffers: %d\n", err);
        exit(EXIT_FAILURE);
    }

    //---------------------------------------------------------------------------------------------------------
    // Execute downsampling and grayscale kernels
    // Enqueue commands to execute the downsampling and grayscale kernels
    cl_event downsample_event_left, downsample_event_right, grayscale_event_left, grayscale_event_right;
    
    // Set kernel arguments
    clSetKernelArg(downsample_kernel_left, 0, sizeof(cl_mem), &left_image_buffer);
    clSetKernelArg(downsample_kernel_left, 1, sizeof(unsigned), &width);
    clSetKernelArg(downsample_kernel_left, 2, sizeof(unsigned), &height);
    clSetKernelArg(downsample_kernel_left, 3, sizeof(unsigned), &new_width);
    clSetKernelArg(downsample_kernel_left, 4, sizeof(unsigned), &new_height);
    clSetKernelArg(downsample_kernel_left, 5, sizeof(cl_mem), &left_downsample_buffer);
    
    clSetKernelArg(downsample_kernel_right, 0, sizeof(cl_mem), &right_image_buffer);
    clSetKernelArg(downsample_kernel_right, 1, sizeof(unsigned), &width);
    clSetKernelArg(downsample_kernel_right, 2, sizeof(unsigned), &height);
    clSetKernelArg(downsample_kernel_right, 3, sizeof(unsigned), &new_width);
    clSetKernelArg(downsample_kernel_right, 4, sizeof(unsigned), &new_height);
    clSetKernelArg(downsample_kernel_right, 5, sizeof(cl_mem), &right_downsample_buffer);
    
    clSetKernelArg(grayscale_kernel_left, 0, sizeof(cl_mem), &left_downsample_buffer);
    clSetKernelArg(grayscale_kernel_left, 1, sizeof(unsigned), &new_width);
    clSetKernelArg(grayscale_kernel_left, 2, sizeof(unsigned), &new_height);
    clSetKernelArg(grayscale_kernel_left, 3, sizeof(cl_mem), &left_gray_buffer);
    
    clSetKernelArg(grayscale_kernel_right, 0, sizeof(cl_mem), &right_downsample_buffer);
    clSetKernelArg(grayscale_kernel_right, 1, sizeof(unsigned), &new_width);
    clSetKernelArg(grayscale_kernel_right, 2, sizeof(unsigned), &new_height);
    clSetKernelArg(grayscale_kernel_right, 3, sizeof(cl_mem), &right_gray_buffer);

    // Enqueue commands to execute downsampling kernel
    err = clEnqueueNDRangeKernel(queue, downsample_kernel_left, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &downsample_event_left);
    err = clEnqueueNDRangeKernel(queue, downsample_kernel_right, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &downsample_event_right);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute downsampling kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
    
    // Wait for the downsampling kernel to finish before grayscale
    clWaitForEvents(1, &downsample_event_left);
    clWaitForEvents(1, &downsample_event_right);
        
    
    double downsample_exec_left = profiling_info(downsample_event_left);
    double downsample_exec_right = profiling_info(downsample_event_left);
    printf("Images downsampled in: %.10f seconds\n", downsample_exec_left+downsample_exec_right);
    

    // Enqueue commands to execute grayscale kernel
    err = clEnqueueNDRangeKernel(queue, grayscale_kernel_left, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &grayscale_event_left);
    err = clEnqueueNDRangeKernel(queue, grayscale_kernel_right, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &grayscale_event_right);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute grayscale kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Wait for the grayscale kernel to finish before stereo disparity

    clWaitForEvents(1, &grayscale_event_left);
    clWaitForEvents(1, &grayscale_event_right);
    
    double grayscale_exec_left = profiling_info(grayscale_event_left);
    double grayscale_exec_right = profiling_info(grayscale_event_right);
    printf("Images grayscaled in: %.10f seconds\n", grayscale_exec_left+grayscale_exec_right);
    
    // debug_write_image("debug_greyscale_right.png", context, queue, right_gray_buffer, new_width, new_height);
    // debug_write_image("debug_greyscale_left.png", context, queue, left_gray_buffer, new_width, new_height);

    //---------------------------------------------------------------------------------------------------------
    // Execute disparity kernel

    int direction_left = 0;
    int direction_right = 1;
    
    
    // Set kernel arguments for disparity calculation
    clSetKernelArg(disparity_kernel_left, 0, sizeof(cl_mem), &left_gray_buffer);
    clSetKernelArg(disparity_kernel_left, 1, sizeof(cl_mem), &right_gray_buffer);
    clSetKernelArg(disparity_kernel_left, 2, sizeof(unsigned), &new_width);
    clSetKernelArg(disparity_kernel_left, 3, sizeof(unsigned), &new_height);
    clSetKernelArg(disparity_kernel_left, 4, sizeof(int), &direction_left);
    clSetKernelArg(disparity_kernel_left, 5, sizeof(int), &scaled_ndisp);
    clSetKernelArg(disparity_kernel_left, 6, sizeof(cl_mem), &left_disparity_map_buffer);
    
    clSetKernelArg(disparity_kernel_right, 0, sizeof(cl_mem), &right_gray_buffer);
    clSetKernelArg(disparity_kernel_right, 1, sizeof(cl_mem), &left_gray_buffer);
    clSetKernelArg(disparity_kernel_right, 2, sizeof(unsigned), &new_width);
    clSetKernelArg(disparity_kernel_right, 3, sizeof(unsigned), &new_height);
    clSetKernelArg(disparity_kernel_right, 4, sizeof(int), &direction_right);
    clSetKernelArg(disparity_kernel_right, 5, sizeof(int), &scaled_ndisp);
    clSetKernelArg(disparity_kernel_right, 6, sizeof(cl_mem), &right_disparity_map_buffer);

    // Enqueue commands to execute the disparity kernels
    cl_event disparity_event_left, disparity_event_right;
    
    // Enqueue command to execute the left disparity kernel
    err = clEnqueueNDRangeKernel(queue, disparity_kernel_left, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &disparity_event_left);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute left disparity kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Enqueue command to execute the right disparity kernel
    err = clEnqueueNDRangeKernel(queue, disparity_kernel_right, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &disparity_event_right);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute right disparity kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Wait for both disparity kernels to finish
    clWaitForEvents(1, &disparity_event_left);
    clWaitForEvents(1, &disparity_event_right);

    // Calculate execution times for each kernel
    double disparity_exec_left = profiling_info(disparity_event_left);
    double disparity_exec_right = profiling_info(disparity_event_right);

    printf("Disparity maps calculated in: %.10f seconds\n", disparity_exec_right+disparity_exec_left);

    // debug_write_image("debug_disparity_right.png", context, queue, right_disparity_map_buffer, new_width, new_height);
    // debug_write_image("debug_disparity_left.png", context, queue, left_disparity_map_buffer, new_width, new_height);
    
    //---------------------------------------------------------------------------------------------------------
    // Execute cross-checking and occlusion filling kernels
    // Enqueue commands to execute the cross-checking and occlusion filling kernels
    cl_event cross_checking_event, occlusion_filling_event;
    
    int threshold = THRESHOLD;
    
    // Set kernel arguments
    clSetKernelArg(cross_check_kernel, 0, sizeof(cl_mem), &left_disparity_map_buffer);
    clSetKernelArg(cross_check_kernel, 1, sizeof(cl_mem), &right_disparity_map_buffer);
    clSetKernelArg(cross_check_kernel, 2, sizeof(cl_mem), &cross_check_buffer);
    clSetKernelArg(cross_check_kernel, 3, sizeof(unsigned), &new_width);
    clSetKernelArg(cross_check_kernel, 4, sizeof(unsigned), &new_height);
    clSetKernelArg(cross_check_kernel, 5, sizeof(int), &threshold);
    clSetKernelArg(cross_check_kernel, 6, sizeof(int), &scaled_ndisp);
    
    // Enqueue commands to execute the cross-checking kernel
    err = clEnqueueNDRangeKernel(queue, cross_check_kernel, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &cross_checking_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute cross-checking kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }
    
    // Wait for the cross-checking kernel to finish
    clWaitForEvents(1, &cross_checking_event);
    
    double crosscheck_exec = profiling_info(cross_checking_event);
    
    // debug_write_image("debug_cross_check.png", context, queue, cross_check_buffer, new_width, new_height);

    int max_radius = MAX_RADIUS;
    
    // Set kernel arguments for occlusion filling
    clSetKernelArg(occlusion_fill_kernel, 0, sizeof(cl_mem), &cross_check_buffer);
    clSetKernelArg(occlusion_fill_kernel, 1, sizeof(cl_mem), &occlusion_map_buffer);
    clSetKernelArg(occlusion_fill_kernel, 2, sizeof(unsigned), &new_width);
    clSetKernelArg(occlusion_fill_kernel, 3, sizeof(unsigned), &new_height);
    clSetKernelArg(occlusion_fill_kernel, 4, sizeof(unsigned), &max_radius);

    // Enqueue commands to execute the occlusion filling kernel
    err = clEnqueueNDRangeKernel(queue, occlusion_fill_kernel, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, &occlusion_filling_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute occlusion filling kernel: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Wait for the occlusion filling kernel to finish
    
    clWaitForEvents(1, &occlusion_filling_event);
    
    double occlusion_exec = profiling_info(occlusion_filling_event);
    printf("Post-processing done in: %.10f seconds\n", occlusion_exec+crosscheck_exec);
    
    // debug_write_image("debug_occlusion.png", context, queue, occlusion_map_buffer, new_width, new_height);
    //---------------------------------------------------------------------------------------------------------
    // Transfer processed disparity map from GPU to host memory
    // Enqueue commands to transfer the processed disparity map from GPU to host
    
    
    // Define the size of the disparity map buffer in bytes
    size_t disparity_map_size = new_width * new_height * sizeof(unsigned char);

    // Create a buffer on the host to store the processed disparity map
    unsigned char* host_disparity_map = (unsigned char*)malloc(disparity_map_size);
    if (!host_disparity_map) {
        fprintf(stderr, "Failed to allocate memory for host disparity map\n");
        exit(EXIT_FAILURE);
    }
    
    cl_event reading_event;
    
    // Enqueue reading event
    err = clEnqueueReadBuffer(queue, occlusion_map_buffer, CL_TRUE, 0, new_width * new_height * sizeof(unsigned char), host_disparity_map, 0, NULL, &reading_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading buffer: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Wait for the read event to complete
    clWaitForEvents(1, &reading_event);
    
    //---------------------------------------------------------------------------------------------------------
    // At this point, the host_disparity_map buffer contains the processed disparity map data

    // Write processed disparity map to disk
    write_grayscale_image("disparity_map.png", host_disparity_map, new_width, new_height);
        
    // Release OpenCL resources
    clReleaseKernel(downsample_kernel_left);
    clReleaseKernel(downsample_kernel_right);
    clReleaseKernel(grayscale_kernel_left);
    clReleaseKernel(grayscale_kernel_right);
    clReleaseKernel(disparity_kernel_left);
    clReleaseKernel(disparity_kernel_right);
    clReleaseKernel(cross_check_kernel);
    clReleaseKernel(occlusion_fill_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Free allocated memory
    free(left_image);
    free(right_image);
    free(host_disparity_map);
    clReleaseMemObject(left_image_buffer);
    clReleaseMemObject(right_image_buffer);
    clReleaseMemObject(left_disparity_map_buffer);
    clReleaseMemObject(right_disparity_map_buffer);
    clReleaseMemObject(right_downsample_buffer);
    clReleaseMemObject(left_downsample_buffer);
    clReleaseMemObject(left_gray_buffer);
    clReleaseMemObject(right_gray_buffer);
    clReleaseMemObject(cross_check_buffer);
    clReleaseMemObject(occlusion_map_buffer);
    

    // Check for errors and handle exceptions
    if (err != CL_SUCCESS) {
        printf("Error %d occurred during cleanup\n", err);
        exit(EXIT_FAILURE);
    }

    return 0;
}

// Function to load OpenCL kernel from file
cl_program load_kernel(const char* kernel_file, cl_context context, cl_device_id device) {
    // Read kernel source code from file
    FILE* file = fopen(kernel_file, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    size_t source_size = ftell(file);
    rewind(file);
    char* source_code = (char*)malloc(source_size + 1);
    if (!source_code) {
        fclose(file);
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    fread(source_code, sizeof(char), source_size, file);
    source_code[source_size] = '\0';
    fclose(file);

    // Create OpenCL program from source code
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL program\n");
        exit(EXIT_FAILURE);
    }

    // Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build OpenCL program\n");
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    free(source_code);
    return program;
}

// Function to create an OpenCL kernel
cl_kernel create_kernel(cl_program program, const char* kernel_name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel: %s\n", kernel_name);
        exit(EXIT_FAILURE);
    }
    return kernel;
}

// Function to write a grayscale image to disk
void write_grayscale_image(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    // Write the grayscale image to the file
    unsigned error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        printf("Failed to write image to file: %s\n", filename);
        exit(EXIT_FAILURE);
    } else {
        printf("Image written successfully: %s\n", filename);
    }
}

// Function to read an image from disk
unsigned char* read_image(const char* filename, unsigned* width, unsigned* height) {
    unsigned error;
    unsigned char* image;

    error = lodepng_decode32_file(&image, width, height, filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        printf("Failed to read image from file: %s\n", filename);
        exit(EXIT_FAILURE);
    } else {
        printf("Image read successfully: %s\n", filename);
    }

    return image;
}

// Function for even profiling
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

// Create OpenCL image object from image data
cl_mem createImageObjectFromData(cl_context context, unsigned char* imageData, unsigned int width, unsigned int height) {
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    cl_mem image = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &format, &desc, imageData, NULL);
    if (!image) {
        printf("Error creating RGBA OpenCL image object\n");
        return NULL;
    }
    return image;
}

// Read images from buffers and write to disk
void debug_write_image(const char* filename, cl_context context, cl_command_queue queue, cl_mem buffer, unsigned int width, unsigned int height) {
    // Define variables for reading buffer
    unsigned char* host_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!host_image) {
        fprintf(stderr, "Failed to allocate memory for host image buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Enqueue command to read buffer from GPU to host
    cl_event read_event;
    cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, width * height * sizeof(unsigned char), host_image, 0, NULL, &read_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading buffer: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Wait for the read event to complete
    clWaitForEvents(1, &read_event);

    // Write the image to disk
    unsigned error = lodepng_encode_file(filename, host_image, width, height, LCT_GREY, 8);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        printf("Failed to write image to file: %s\n", filename);
        exit(EXIT_FAILURE);
    } else {
        printf("Image written successfully: %s\n", filename);
    }

    // Free allocated memory
    free(host_image);
}