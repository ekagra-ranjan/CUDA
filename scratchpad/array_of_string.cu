#include "cuda.h"
#include "stdio.h"
#include <string>
#include <iostream>

/*
passing array of string to GPU

- string is array of char so array of string is char**
- copying array of string by coping string in loop will copy the host address of string to GPU which is invalid
- we need to copy the content of the string by value to an GPU address
- we cant use the GPU address from host so we need.3 loops as seen below
*/

 __global__ void kernel(char** a, int size){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // a[i] = "i";
    // a[i] = "i";
    // printf("Thread %d: %s\n", i, a[i]);
    // printf("Thread %d\n", i);
    // printf("Thread %d: %s\n", i, a[i]);

    for(int i=0; i<size; i++){
        char c = a[idx][i];
        // if (c == '\0'){
        //     break;
        // }
        if (c == 'a'){
            printf("Thread %d: %s: %c: %i\n", idx, a[idx], c, i);
        }
    }
}

int main(){
    int N = 4;
    char** d_warmup_a;
    cudaMalloc((void ***)&d_warmup_a, N*sizeof(char*));

    printf("start\n");

    // int N = 4;
    char** d_a;
    cudaMalloc((void ***)&d_a, N*sizeof(char*));

    printf("device malloc done\n");

    int size = 8;
    char *h_a[] = {"abcdef", "bc", "ac", "d"};
    char **h_pointer;
    h_pointer = (char**)malloc(N*sizeof(char*));

    printf("host malloc done\n");

    int string_size[N];
    for (int i = 0; i < N; i++){
        for (int j = 0; j < size; j++){
            char c = h_a[i][j];
            if (c == '\0'){
                string_size[i] = j;
                break;
            }
        }
    }

    printf("length done\n");

		// loop 1
    for (int i = 0; i < N; i++){
        cudaMalloc((void **)&h_pointer[i], size * sizeof(char));
    }
	
		// loop 2
    for (int i = 0; i < N; i++){
        printf("string_size[%d]: %d\n", i, string_size[i]);
        cudaMemcpy(h_pointer[i], h_a[i], string_size[i] * sizeof(char), cudaMemcpyHostToDevice);
    }

		// loop 3
    cudaMemcpy(d_a, h_pointer, N*sizeof(char*), cudaMemcpyHostToDevice);

    printf("start kernel\n");
    kernel<<<1, N>>>(d_a, size);
    printf("done\n");

    cudaFree(&d_a);
    // free(h_a);
    cudaDeviceSynchronize();
}
