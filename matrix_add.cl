__kernel void matrixAdd(__global float* A,
                        __global float* B,
                        __global char* output,
                        __global float* result_mat_cl){

    int i = get_global_id(0); // Global ID represents the row index
    int j = get_global_id(1); // Global ID represents the column index
    
    int index = i * 100 + j; // Convert 2D indices to 1D index

    result_mat_cl[index] = A[index] + B[index];

    output[0] = 'D';
    output[1] = 'O';
    output[2] = 'N';
    output[3] = 'E';
    output[5] = '\0';
}