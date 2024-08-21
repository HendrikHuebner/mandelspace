#include <cuda_runtime.h>
#include <iostream>


__global__ void render(uchar4* devPtr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Simple triangle condition
    if (x > width / 4 && x < 3 * width / 4 && y > height / 4 && y < 3 * height / 4) {
        devPtr[idx] = make_uchar4(255, 0, 0, 255); // Red color
    } else {
        devPtr[idx] = make_uchar4(0, 0, 0, 255); // Black color
    }
}

void cudaDraw(struct cudaGraphicsResource *pboCuda, int width, int height) {
    uchar4* devPtr;
    size_t size;

    cudaGraphicsMapResources(1, &pboCuda, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, pboCuda);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    render<<<grid, block>>>(devPtr, width, height);

    cudaGraphicsUnmapResources(1, &pboCuda, 0);
}
