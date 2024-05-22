// Jaakko Astikainen 2024

#ifndef PROJECT_H
#define PROJECT_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#define DOWN_RATIO 4 // downscaling ratio
#define KERNEL_SIZE 23 // ZNCC computation kernel size
#define KERNEL_RADIUS (KERNEL_SIZE/2) // kernel radius
#define THRESHOLD 1 // disparity threshold for cross-check
#define MAX_RADIUS 64 // max nearest nonzero neighbor search radius
#define THREADS_PER_IM 4 // amount of pthreads per image
#define THREADS (2*THREADS_PER_IM) // total amount of threads
#define LOCAL_WORK_SIZE_NVIDIA 24
#define LOCAL_WORK_SIZE_INTEL 16

#endif //PROJECT_H