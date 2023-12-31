#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCKSIZE 256

/**********/
/* iDivUp */
/**********/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}

/********************************************************/
/* KERNEL FUNCTION FOR TESTING RANDOM NUMBER GENERATION */
/********************************************************/
__global__ void testrand1(unsigned long seed, float *a, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    if (idx < N) {
        curand_init(seed, idx+2, 0, &state);
        a[idx] = curand_uniform(&state);
    }
}

/********/
/* MAIN */
/********/
int main() {

    const int N = 10;

    float *h_a  = (float*)malloc(N*sizeof(float));
    float *d_a; gpuErrchk(cudaMalloc((void**)&d_a, N*sizeof(float)));

    testrand1<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(1234, d_a, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<N; i++) printf("%i %f\n", i, h_a[i]);

    // getchar();
}
