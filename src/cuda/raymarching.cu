#include "fractals/mandelbulb.hpp"
#include "fractals/menger_sponge.hpp"
#include "../fractal.hpp"
#include "util.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define EPSILON 0.002
#define MAX_STEPS 64
#define MAX_DIST 8

__device__ float map(float3 ray, double time) {
	return cuda_menger_sponge_de(7, ray);
}

__device__ float3 normal(float3 pos, double time) {
	const float2 b = make_float2(1.0,-1.0);
	const float3 b1 = make_float3(b.x, b.y, b.y); 
	const float3 b2 = make_float3(b.y, b.y, b.x); 
	const float3 b3 = make_float3(b.y, b.x, b.y); 
	const float3 b4 = make_float3(b.x, b.x, b.x);
	const float m = 0.0002;

    float p1 = map(pos + b1 * m, time);
    float p2 = map(pos + b2 * m, time);
    float p3 = map(pos + b3 * m, time);
    float p4 = map(pos + b4 * m, time);
	
	return normalize(b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4);
}

__device__ float shadow(float3 pos, float3 dir, float min, float max, double time) {
	float dist = min;
    for(int i = 0; i < 256 && dist < max; i++) {
		float h = map(pos + dir * dist, time);
        
		if(h < EPSILON)
		return 0.0;
        
		dist += h;
    }
	
    return 1.0;
}

struct HitInfo {
	int steps;
	float dist;
	float3 pos;
};

__device__ HitInfo march(Fractal frac, float3 pos, float3 dir, double time) {
	float distance = 0.0;
	int steps;
	float3 currentPos;

	for (steps = 0; steps < MAX_STEPS && distance < MAX_DIST; steps++) {
		currentPos = pos + dir * distance;
		float d = map(currentPos, time);

		if (d < EPSILON) {
			HitInfo h = {
				.steps = steps,
				.dist = distance,
				.pos = currentPos
			};

			return h;
		}
		
		distance += d;
	}

	// no hit	
	return {
		.steps = -1,
		.dist = distance,
		.pos = pos
	};
}

__global__ void render(Fractal frac, Scene scene, CameraDTO camera, double time) {
	size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < scene.width && y < scene.height) {
		float min_w_h = (float) min(scene.width, scene.height);

		float ar = (float) scene.width / (float) scene.height;
		float u = (float) x / min_w_h - ar * 0.5f;
		float v = (float) y / min_w_h - 0.5f;

	    float3 dir = normalize(to_float3(camera.forward) + u * to_float3(camera.right)  + v * to_float3(camera.up));

		HitInfo h = march(frac, to_float3(camera.pos), dir, time);
		unsigned char color = (unsigned char) (255.0 -  255.0f * (float) h.steps / (float) MAX_STEPS);

		const float3 lightDir = make_float3(-1.0, -1.0, -1.0);
		float s = shadow(h.pos, lightDir, 0, 8, time);
		color = max(0, color - (unsigned char) s * 100);

		scene.buf[y * scene.width + x] = make_uchar4(color, color, color, 255);
	}
}

void cudaDraw(Fractal frac, CameraDTO camera, struct cudaGraphicsResource *pboCuda, int width, int height, double time) {
    Scene scene(NULL, width, height);
    size_t size;

    cudaGraphicsMapResources(1, &pboCuda, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&scene.buf, &size, pboCuda);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    render<<<grid, block>>>(frac, scene, camera, time);

    cudaGraphicsUnmapResources(1, &pboCuda, 0);
}
