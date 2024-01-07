#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int inputLength, numSegments;
    DataType *hostInput1, *hostInput2, *hostOutput, *resultRef;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;

    if (argc != 3) {
        printf("Usage: %s <input length> <number of segments>\n", argv[0]);
        return -1;
    }
    inputLength = atoi(argv[1]);
    numSegments = atoi(argv[2]);
    int S_seg = (inputLength + numSegments - 1) / numSegments; // Calculate segment size
    printf("Input length: %d, Number of segments: %d, Segment size: %d\n", inputLength, numSegments, S_seg);

    hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = (DataType)rand() / RAND_MAX;
        hostInput2[i] = (DataType)rand() / RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }

    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));

    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int blockSize = 256;

    double startTime, endTime;
    double h2dTime = 0;
    double exeTime = 0;
    double d2hTime = 0;

    for (int i = 0; i < numSegments; i++) {
        int offset = i * S_seg;
        int size = min(S_seg, inputLength - offset);
        cudaStream_t stream = streams[i % 4];
        startTime = getTime();
        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, stream);
        endTime = getTime();
        h2dTime += endTime - startTime;
    }    
    for (int i = 0; i < numSegments; i++) {
        int offset = i * S_seg;
        int size = min(S_seg, inputLength - offset);
        cudaStream_t stream = streams[i % 4];
        int gridSize = (size - 1) / blockSize + 1;
        startTime = getTime();
        vecAdd<<<gridSize, blockSize, 0, stream>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, size);
        cudaStreamSynchronize(stream);
        endTime = getTime();
        exeTime += endTime - startTime;
    }
    for (int i = 0; i < numSegments; i++) {
        int offset = i * S_seg;
        int size = min(S_seg, inputLength - offset);
        cudaStream_t stream = streams[i % 4];
        startTime = getTime();
        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, size * sizeof(DataType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        endTime = getTime();
        d2hTime += endTime - startTime;
    }

    printf("Host to Device copy time:   %f seconds\n", h2dTime);
    printf("Kernel execution time:      %f seconds\n", exeTime);
    printf("Device to Host copy time:   %f seconds\n", d2hTime);

    for (int i = 0; i < inputLength; i++) {
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-9) {
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
