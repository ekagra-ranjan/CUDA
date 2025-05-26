#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#

__global__ void myKernel(float* d_data, int hidden_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {  // Bounds check
        d_data[idx] = d_data[idx] * 2;
    }
}

int main(){
    float* h_data1;  // Host data (CPU)
    float* h_data2;  // Host data (CPU)
    float* d_data1;  // Device data (GPU)
    float* d_data2;  // Device data (GPU)
    int hidden_dim = 12288;
    size_t dataSize = hidden_dim * sizeof(float);

    // Allocate memory on host and device
    cudaMalloc(&d_data1, dataSize);
    cudaMalloc(&d_data2, dataSize);
    h_data1 = (float*)malloc(dataSize);
    h_data2 = (float*)malloc(dataSize);

    // Create streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);  // Stream for kernel
    cudaStreamCreate(&stream1);  // Stream for timing

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Transfer data from host to device (this is done on stream1)
    cudaMemcpyAsync(d_data1, h_data1, dataSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data2, h_data2, dataSize, cudaMemcpyHostToDevice, stream1);

    int num_iters = 1e4;
    std::vector<float> bw(num_iters);

    int threadsPerBlock = 256;
    int blocksPerGrid = (hidden_dim + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < num_iters; i++) {
        // Start timing on stream1 (for memory copy)
        cudaEventRecord(start, stream1);

        // Launch kernel on stream0
        // myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data2, hidden_dim);
        myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream0>>>(d_data1, hidden_dim);

        // Stop timing on stream1 (for memory copy)
        cudaEventRecord(stop, stream1);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // printf("Elapsed time: %f ms\n", milliseconds);

        // Store bandwidth
        bw[i] = (dataSize / 1e6) / milliseconds; // GB/s
    }

    // Calculate the average transfer speed
    float transferSpeedGBps = 0;
    for (int i = 0; i < num_iters; i++) {
        transferSpeedGBps += bw[i];
    }
    transferSpeedGBps /= num_iters;

    printf("Transfer speed: %f GB/s\n", transferSpeedGBps);

    return 0;
}
