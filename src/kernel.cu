#include <stdio.h>

void __global__ kernel_add_one(int* a, int length) {
    int gid = threadIdx.x + blockDim.x*blockIdx.x;
    /*
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < length) {
        a[gid] += 1;
    }
    */
    while(gid < length) {
    	a[gid] += 1;
        gid += blockDim.x*gridDim.x;
    }
}
