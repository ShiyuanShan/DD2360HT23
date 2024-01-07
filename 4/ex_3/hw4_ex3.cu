#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

// Error checking macros
#define gpuCheck(stmt) \
    do { \
        cudaError_t err = stmt; \
        if (err != cudaSuccess) { \
            printf("ERROR. Failed to run stmt %s\n", #stmt); \
            break; \
        } \
    } while (0)

#define cublasCheck(stmt) \
    do { \
        cublasStatus_t err = stmt; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt); \
            break; \
        } \
    } while (0)

#define cusparseCheck(stmt) \
    do { \
        cusparseStatus_t err = stmt; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt); \
            break; \
        } \
    } while (0)

// Timing functions
struct timeval t_start, t_end;
void cputimer_start() {
    gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info) {
    gettimeofday(&t_end, 0);
    double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec);
    printf("Timing - %s. \t\tElapsed %.0f microseconds \n", info, time);
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX, double alpha) {
    // Implementation as in your example...
}

int main(int argc, char **argv) {
    // Variable definitions as in your example...
    int device = 0;
    int dimX;  // Dimension of the metal rod
    int nsteps;  // Number of time steps to perform
    double alpha = 0.4;  // Diffusion coefficient
    double *temp;  // Array to store the final time step
    double *A;  // Sparse matrix A values in the CSR format
    int *ARowPtr;  // Sparse matrix A row pointers in the CSR format
    int *AColIndx;  // Sparse matrix A col values in the CSR format
    int nzv;  // Number of non-zero values in the sparse matrix
    double *tmp;  // Temporal array of dimX for computations
    size_t bufferSize = 0;  // Buffer size needed by some routines
    void *buffer = nullptr;  // Buffer used by some routines in the libraries
    int concurrentAccessQ;  // Check if concurrent access flag is set
    double zero = 0;  // Zero constant
    double one = 1;  // One constant
    double norm;  // Variable for norm values
    double error;  // Variable for storing the relative error
    double tempLeft = 200.;  // Left heat source applied to the rod
    double tempRight = 300.;  // Right heat source applied to the rod
    cublasHandle_t cublasHandle;  // cuBLAS handle
    cusparseHandle_t cusparseHandle;  // cuSPARSE handle

    // Read the command line arguments and print them...
    dimX = atoi(argv[1]);
    nsteps = atoi(argv[2]);
    printf("The X dimension of the grid is %d \n", dimX);
    printf("The number of time steps to perform is %d \n", nsteps);

    // Check for concurrent managed access...
    gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

    // Allocate memory using Unified Memory for temp, tmp, and the sparse matrix...
    cputimer_start();
    gpuCheck(cudaMallocManaged(&temp, dimX * sizeof(double)));
    gpuCheck(cudaMallocManaged(&tmp, dimX * sizeof(double)));
    gpuCheck(cudaMallocManaged(&A, nzv * sizeof(double)));
    gpuCheck(cudaMallocManaged(&ARowPtr, (dimX + 1) * sizeof(int)));
    gpuCheck(cudaMallocManaged(&AColIndx, nzv * sizeof(int)));
    cputimer_stop("Allocating device memory");

    // Prefetch data to the appropriate locations...
    if (concurrentAccessQ) {
        cputimer_start();
        gpuCheck(cudaMemPrefetchAsync(A, nzv * sizeof(double), cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), cudaCpuDeviceId));
        cputimer_stop("Prefetching GPU memory to the host");
    }

    // Initialize the sparse matrix...
    cputimer_start();
    matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
    cputimer_stop("Initializing the sparse matrix on the host");

    // Initialize the boundary conditions...
    cputimer_start();
    memset(temp, 0, sizeof(double) * dimX);
    temp[0] = tempLeft;
    temp[dimX - 1] = tempRight;
    cputimer_stop("Initializing memory on the host");

    // Create the cuBLAS and cuSPARSE handles...
    cublasCheck(cublasCreate(&cublasHandle));
    cusparseCheck(cusparseCreate(&cusparseHandle));

    // Set the cuBLAS pointer mode to CUBLAS_POINTER_MODE_HOST...
    cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

    // Create the matrix descriptor and vector descriptors for temp and tmp...
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCheck(cusparseCreateCsr(&matA, dimX, dimX, nzv, ARowPtr, AColIndx, A, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseCheck(cusparseCreateDnVec(&vecX, dimX, temp, CUDA_R_64F));
    cusparseCheck(cusparseCreateDnVec(&vecY, dimX, tmp, CUDA_R_64F));

// Calculate buffer size and allocate buffer for cuSPARSE operations
cusparseCheck(cusparseSpMV_bufferSize(
    cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX, &zero, vecY, 
    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
gpuCheck(cudaMalloc(&buffer, bufferSize));

for (int it = 0; it < nsteps; ++it) {
    cusparseCheck(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX, &zero, vecY, 
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));
    cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

    if (norm < 1e-4)
        break;
}
    // Calculate the exact solution using thrust...
    thrust::device_ptr<double> thrustPtr(tmp);
    thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft, (tempRight - tempLeft) / (dimX - 1));

    // Calculate the relative approximation error...
    one = -1;
    cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));
    cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));
    error = norm;
    cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));
    error /= norm;
    printf("The relative error of the approximation is %f\n", error);

    // Destroy the matrix descriptor and vector descriptors...
    cusparseCheck(cusparseDestroySpMat(matA));
    cusparseCheck(cusparseDestroyDnVec(vecX));
    cusparseCheck(cusparseDestroyDnVec(vecY));

    // Destroy the cuSPARSE and cuBLAS handles...
    cusparseCheck(cusparseDestroy(cusparseHandle));
    cublasCheck(cublasDestroy(cublasHandle));

    // Deallocate memory...
    cudaFree(temp);
    cudaFree(tmp);
    cudaFree(A);
    cudaFree(ARowPtr);
    cudaFree(AColIndx);
    cudaFree(buffer);

    return 0;
}
