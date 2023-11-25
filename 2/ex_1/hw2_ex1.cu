#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) out[i] = in1[i] + in2[i];
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
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    printf("Input invalid!\n");
    return -1;
  }
  inputLength = atoi(argv[1]);
  printf("Input length: %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = (DataType)rand() / RAND_MAX;
    hostInput2[i] = (DataType)rand() / RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  startTime = getTime();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  endTime = getTime();
  printf("Time cost of Memcpy from Host to Device: %f seconds\n", endTime - startTime);

  //@@ Initialize the 1D grid and block dimensions here
  int blockSize = 256;
  int gridSize = (inputLength - 1) / blockSize + 1;

  //@@ Launch the GPU Kernel here
  startTime = getTime();
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  endTime = getTime();
  printf("Time cost of CUDA Kernel: %f seconds\n", endTime - startTime);

  //@@ Copy the GPU memory back to the CPU here
  startTime = getTime();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  endTime = getTime();  
  printf("Time cost of Memcpy from Device to Host: %f seconds\n", endTime - startTime);

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-9) {
      printf("Mismatch at index %d\n", i);
      break;
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  // Free the CPU memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}