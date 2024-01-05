#include <stdio.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define NUM_BINS 4096
#define THREADS_PER_BLOCK 256

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  //@@ Insert code below to compute histogram of input using shared memory and atomics
  extern __shared__ unsigned int shared_bins[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize shared memory bins to 0
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    shared_bins[i] = 0;
  }
  __syncthreads();

  // Compute histogram using shared memory and atomics
  while (tid < num_elements) {
      atomicAdd(&shared_bins[input[tid]], 1);
      tid += stride;
  }
  __syncthreads();

  // Update global memory bins
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    atomicAdd(&bins[i], shared_bins[i]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Insert code below to clean up bins that saturate at 127
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Clean up bins that saturate at 127
  while (tid < num_bins) {
    if (bins[tid] > 127) {
      bins[tid] = 127;
    }
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char **argv) {
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  // Read inputLength from args
  sscanf(argv[1], "%d", &inputLength);

  printf("The input length is %d\n", inputLength);

  // Allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  // Initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; ++i) {
    hostInput[i] = rand() % NUM_BINS;
  }

  // Create reference result in CPU
  for (int i = 0; i < NUM_BINS; ++i) {
    resultRef[i] = 0;
  }
  for (int i = 0; i < inputLength; ++i) {
    resultRef[hostInput[i]]++;
  }

  // Allocate GPU memory
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));

  // Copy memory to the GPU
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  // Initialize GPU results
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim((inputLength + blockDim.x - 1) / blockDim.x, 1, 1);

  // Launch the GPU Kernel for histogram computation
  histogram_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  // Launch the GPU Kernel for cleanup
  convert_kernel<<<gridDim, blockDim>>>(deviceBins, NUM_BINS);

  // Copy the GPU memory back to the CPU
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // Compare the output with the reference
  for (int i = 0; i < NUM_BINS; ++i) {
    if (hostBins[i] != resultRef[i]) {
      printf("Mismatch at bin %d: Expected %u, Got %u\n", i, resultRef[i], hostBins[i]);
    } else printf("%d, %u\n",i,hostBins[i]);
  }

  // Free the GPU memory
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  // Free the CPU memory
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
