#include "fractals/mandelbulb.hpp"
#include "../fractal.hpp"
#include "util.hpp"
#include <cuda_runtime.h>
#include <iostream>


__device__ float march(Fractal frac, float3 pos, float3 dir) {
	float total_dist = 0.0;
	int max_ray_steps = 64;
	float min_distance = 0.002;

	int steps;
	for (steps = 0; steps < max_ray_steps; ++steps) {
		float3 p = pos + dir * total_dist;
		float distance = cuda_mandelbulb_de(8.0, 8.0, p);

		total_dist += distance;
		if (distance < min_distance) break;
	}

	return 1.0 - (float) steps / (float) max_ray_steps;
}

__global__ void render(Fractal frac, Scene scene, Camera camera) {
	size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < scene.width && y < scene.height) {
		float min_w_h = (float) min(scene.width, scene.height);

		float ar = (float) scene.width / (float) scene.height;
		float u = (float) x / min_w_h - ar * 0.5f;
		float v = (float) y / min_w_h - 0.5f;

	    float3 dir = normalize(make_float3(1.0, u, v));

		unsigned char c = (unsigned char) (255.0f * march(frac, camera.pos, dir));
	
		scene.buf[y * scene.width + x] = make_uchar4(c, c, c, 255);
	}
}

void cudaDraw(Fractal frac, struct cudaGraphicsResource *pboCuda, int width, int height) {
    Scene scene(NULL, width, height);
    size_t size;

    cudaGraphicsMapResources(1, &pboCuda, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&scene.buf, &size, pboCuda);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    Camera cam(make_float3(-3.0, 0.0, 0.0), make_float3(1.0, 0.0, 0.0));

    render<<<grid, block>>>(frac, scene, cam);

    cudaGraphicsUnmapResources(1, &pboCuda, 0);
}
