// Jaakko Astikainen 2024

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <string.h>

#define WIDTH 100
#define HEIGHT 100
#define READ_BUFF 5

void add_matrix(float* A, float* B, float* result);

int main()
{
    cl_int           err;
    cl_uint          num_platforms;
    cl_platform_id  *platforms;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    char             result[READ_BUFF];

    // load kernel from file
    char *kernel_source;
    size_t source_size, program_size;
    FILE *fp = fopen("matrix_add.cl", "rb");
    if (!fp) {
        printf("Failed to load kernel\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    kernel_source = (char*)malloc(program_size + 1);
    kernel_source[program_size] = '\0';
    fread(kernel_source, sizeof(char), program_size, fp);
    fclose(fp);
    // printf("%s\n", kernel_source);
    
    
    
    // run matrix addition on host
    float* matrix_1 = (float*) malloc(sizeof(float) * WIDTH * HEIGHT);
    float* matrix_2 = (float*) malloc(sizeof(float) * WIDTH * HEIGHT);
    float* result_mat = (float*) malloc(sizeof(float) * WIDTH * HEIGHT);
    float* result_mat_cl = (float*) malloc(sizeof(float) * WIDTH * HEIGHT);
    
    // fill matrices
    srand(time(NULL));
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            int o1 = rand();
            int n1 = rand() + 1;
            int o2 = rand();
            int n2 = rand() + 1;
            matrix_1[i*HEIGHT + j] = (float)o1/n1;
            matrix_2[i*HEIGHT + j] = (float)o2/n2;
        }
    }
    
    printf("Running matrix addition on host:\n");
    add_matrix(matrix_1, matrix_2, result_mat);
    
    
    printf("Running matrix addition on OpenCL:\n");
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

    //DEVICE (could be CL_DEVICE_TYPE_GPU)
    //
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    //CONTEXT
    //
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    //QUEUE
    //
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    
    //READ KERNEL AND COMPILE IT
    //
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    // Create and set kernel arguments
    kernel = clCreateKernel(program, "matrixAdd", &err);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*sizeof(float), NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*sizeof(float), NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, READ_BUFF * sizeof(char), NULL, &err);
    cl_mem cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH*HEIGHT*sizeof(float), NULL, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_result);
    
    // Enqueue write operations to copy data to the device buffers
    cl_event event;
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, WIDTH*HEIGHT*sizeof(float), matrix_1, 0, NULL, &event);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, WIDTH*HEIGHT*sizeof(float), matrix_2, 0, NULL, &event);
    
    // run kernel and fill read buffer
    
    size_t global_work_size[2] = { WIDTH, HEIGHT }; // 2D global work size
    size_t local_work_size[2] = { 1, 1 }; // 2D local work size, can be adjusted according to device capabilities

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);

    
    err = clEnqueueTask(queue, kernel, 0, NULL, &event);
    
    err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, READ_BUFF * sizeof(char), result, 0, NULL, &event);
    err = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(float), result_mat_cl, 0, NULL, &event);
    printf("***%s***\n", result);
    
    clFinish(queue);
    
    // Get profiling information
    cl_ulong start, end;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get profiling information\n");
        // Handle error
    }
    // Calculate kernel execution time
    double kernel_exec_time = (end - start) * 1.0e-9; // Convert nanoseconds to seconds

    printf("Kernel execution time: %.10f seconds\n", kernel_exec_time);

    // cross-check results with host calculations
    if (memcmp(result_mat, result_mat_cl, WIDTH*HEIGHT*sizeof(float)) == (int)0)
    {
        printf("Results match\n");
    }
    else
    {
        printf("Results don't match\n");
    }
    
    //Free memory
    clReleaseMemObject(output);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);
    free(matrix_1);
    free(matrix_2);
    free(result_mat);
    free(result_mat_cl);
    clReleaseEvent(event);
    
    return 0;
}

void add_matrix(float* A, float* B, float* result)
{
    struct timeval tv;
    struct timeval start_tv;
    double elapsed = 0.0;
    
    gettimeofday(&start_tv, NULL);
    
    
    for (int i = 0; i < HEIGHT; i++) // rows
    {
        for (int j = 0; j < WIDTH; j++) // columns
        {
            result[HEIGHT*i + j] = A[HEIGHT*i + j] + B[HEIGHT*i + j];
        }
    }
    
    gettimeofday(&tv, NULL);
    elapsed = (tv.tv_sec - start_tv.tv_sec) +
              (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
    printf("Elapsed time: %.10f seconds\n", elapsed);
}