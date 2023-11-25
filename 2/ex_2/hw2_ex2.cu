#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define DataType double
#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    DataType sum = 0.0;
    for (int i = 0; i < numAColumns; i++)
      sum += A[row * numAColumns + i] * B[i * numBColumns + col];
    C[row * numBColumns + col] = sum;
  }
}

//@@ Insert code to implement timer start
double startTime;
//@@ Insert code to implement timer stop
double endTime;

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  DataType *hostA;      // The A matrix
  DataType *hostB;      // The B matrix
  DataType *hostC;      // The output C matrix
  DataType *resultRef;  // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;         // number of rows in the matrix A
  int numAColumns;      // number of columns in the matrix A
  int numBRows;         // number of rows in the matrix B
  int numBColumns;      // number of columns in the matrix B
  int numCRows;         // number of rows in the matrix C
  int numCColumns;      // number of columns in the matrix C

  //@@ Insert code below to read in numARows, numAColumns, numBRows, numBColumns from args
  if (argc != 5) {
    printf("Input invalid!\n");
    return -1;
  }
  numCRows = numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numCColumns = numBColumns = atoi(argv[4]);

  if (numAColumns != numBRows) {
    printf("Input wrong!\n");
    return -1;
  }

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows * numAColumns; i++)
    hostA[i] = (DataType)rand() / RAND_MAX;
  for (int i = 0; i < numBRows * numBColumns; i++)
    hostB[i] = (DataType)rand() / RAND_MAX;

  // Reference computation for verification
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      DataType sum = 0.0;
      for (int k = 0; k < numAColumns; k++)
        sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      resultRef[i * numCColumns + j] = sum;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  startTime = getTime();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  endTime = getTime();
  printf("Time cost of Memcpy from Host to Device: %f seconds\n", endTime - startTime);

  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(16, 16);
  dim3 gridSize((numCColumns - 1) / blockSize.x + 1, (numCRows - 1) / blockSize.y + 1);

  //@@ Launch the GPU Kernel here
  startTime = getTime();
  gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  endTime = getTime();
  printf("Time cost of CUDA Kernel: %f seconds\n", endTime - startTime);

  //@@ Copy the GPU memory back to the CPU here  
  startTime = getTime();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  endTime = getTime();  
  printf("Time cost of Memcpy from Device to Host: %f seconds\n", endTime - startTime);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numCRows * numCColumns; i++) {
    if (fabs(hostC[i] - resultRef[i]) > 1e-9) {
      printf("Mismatch at index %d\n", i);
      break;
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  // Free the CPU memory
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
