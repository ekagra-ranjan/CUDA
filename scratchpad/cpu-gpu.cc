#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

int main(){

// Example in C++ using CUDA
float* h_data;  // Host data (CPU)
float* d_data;  // Device data (GPU)
int hidden_dim = 12288;
size_t dataSize = hidden_dim * sizeof(float);

// Allocate memory on host and device
cudaMalloc(&d_data, dataSize);
h_data = (float*)malloc(dataSize);

// Record start and stop events
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Dummy Transfer data from host to device
cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

int num_iters = 1e3;
std::vector<float> bw(num_iters);

for (int i=0; i<num_iters; i++){

    // Generate random data
    for (int j=0; j<hidden_dim; j++){
        h_data[j] = rand() % 100;
    }

    // Start timing
    cudaEventRecord(start, 0);

    // Transfer data from host to device
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the result
    float bw_per_iter = (dataSize / 1e6) / milliseconds; // GB/s

    // printf("Transfer speed: %f GB/s\n", bw_per_iter);

    bw[i] = bw_per_iter;
}

// Calculate the average transfer speed
float transferSpeedGBps = 0;
for (int i=0; i<num_iters; i++){
    // printf("Transfer speed: %f GB/s\n", bw[i]);
    transferSpeedGBps += bw[i];
}
transferSpeedGBps /= num_iters;


printf("Transfer speed: %f GB/s\n", transferSpeedGBps);

return 0;
}
