#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NUM_GPUS 2  // Change this based on available GPUs

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_NCCL(call)                                                      \
    do {                                                                      \
        ncclResult_t err = call;                                              \
        if (err != ncclSuccess) {                                             \
            std::cerr << "NCCL Error: " << ncclGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

void all_gather_example() {
    int numDevices = NUM_GPUS;
    std::vector<int> devices(numDevices);
    std::vector<cudaStream_t> streams(numDevices);
    std::vector<ncclComm_t> comms(numDevices);
    std::vector<float*> d_sendBuf(numDevices);
    std::vector<float*> d_recvBuf(numDevices);

    size_t dataSize = 4; // Each GPU sends 4 floats
    size_t totalDataSize = dataSize * numDevices;

    // Initialize devices
    for (int i = 0; i < numDevices; i++) {
        devices[i] = i;
        CHECK_CUDA(cudaSetDevice(i));

        // Allocate memory for sending and receiving data
        CHECK_CUDA(cudaMalloc(&d_sendBuf[i], dataSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_recvBuf[i], totalDataSize * sizeof(float)));

        // Initialize stream
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // Fill send buffer with unique values for each GPU
        std::vector<float> h_sendBuf(dataSize, i + 1);
        CHECK_CUDA(cudaMemcpy(d_sendBuf[i], h_sendBuf.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize NCCL
    CHECK_NCCL(ncclCommInitAll(comms.data(), numDevices, devices.data()));

    // benchmark time to all gather START
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform All-Gather operation
    for (int i = 0; i < numDevices; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclAllGather(d_sendBuf[i], d_recvBuf[i], dataSize, ncclFloat, comms[i], streams[i]));
    }

    // Synchronize
    for (int i = 0; i < numDevices; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // benchmark time to all gather STOP
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "All-Gather time: " << milliseconds << " ms" << std::endl;

    // Copy data back to host and print results
    for (int i = 0; i < numDevices; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        std::vector<float> h_recvBuf(totalDataSize);
        CHECK_CUDA(cudaMemcpy(h_recvBuf.data(), d_recvBuf[i], totalDataSize * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "GPU " << i << " received data: ";
        for (float val : h_recvBuf) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    for (int i = 0; i < numDevices; i++) {
        CHECK_CUDA(cudaFree(d_sendBuf[i]));
        CHECK_CUDA(cudaFree(d_recvBuf[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }
}

int main() {
    all_gather_example();
    return 0;
}
