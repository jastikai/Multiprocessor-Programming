/*# -----------------------------------------------------------------------------
# Manuel Lage Cañellas (CMVS - University of Oulu).
# Multiprocessor programming course 2024
#
# -----------------------------------------------------------------------------
*/

/*
Example of where you can find your OpenCL based on your SDK
NVIDIA Linux
Linker:     /usr/local/cuda-11.6/lib64/libOpenCL.so
Directory:  /usr/local/cuda-11.6/include

NVIDIA Windows
Linker:     Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64\OpenCl.lib
Directory:  Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include

Intel CPU Linux
Linker:      /opt/intel/system_studio_2020/opencl/SDK/lib64/libOpenCl.so
Directory:   opt/intel/system_studio_2020/opencl/SDK/include

Intel Graphics Linux
install intel-opencl-icd

*/

#include <stdio.h>
#include <stdlib.h>

//Depending of your installation more includes should be uses, check your particular SDK installation
//
#include <CL/cl.h>


//This is a kernel, a piece of code intended to be executed in a GPU or CPU
const char *kernel_source =
"__kernel void hello(__global char *output) {"
"output[0]='h';"
"output[1]='e';"
"output[2]='l';"
"output[3]='l';"
"output[4]='o';"
"output[5]=',';"
"output[6]=' ';"
"output[7]='w';"
"output[8]='o';"
"output[9]='r';"
"output[10]='l';"
"output[11]='d';"
"output[12]='\\0';"
"}";

//The rest of the code is intended to be executed in the host
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
    cl_mem           output;

    char result[13];

    // PLATFORM
    // In this example we will only consider one platform
    //
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
    queue = clCreateCommandQueue(context, device, 0, &err);

    //READ KERNEL AND COMPILE IT
    //
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);


    //CREATE KERNEL AND KERNEL PARAMETERS
    //
    kernel = clCreateKernel(program, "hello", &err);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 13 * sizeof(char), NULL, &err);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output);

    //EXECUTE KERNEL!
    //
    err = clEnqueueTask(queue, kernel, 0, NULL, NULL);

    //READ KERNEL OUTPUT
    //
    err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, 13 * sizeof(char), result, 0, NULL, NULL);
    printf("***%s***", result);


    //Free your memory please....
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);


    return 0;
}
