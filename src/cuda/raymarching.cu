#include "fractals/mandelbulb.hpp"
#include "fractals/menger_prison.hpp"
#include "fractals/mandelbox.hpp"
#include "../fractal.hpp"
#include "util.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define EPSILON 0.00002
#define MAX_STEPS 512
#define MAX_DIST 8

__device__ float map(float3 ray, float lod, double time) {
	int iterations = (int) (clamp(2.2 + 4 * exp(0.3 - 0.2 * lod), 3.0f, 8.0f));
	// return cuda_mandelbox_de(10, 3, 0.7, 0.2, ray);
	return cuda_mandelbulb_de(5, ray, time);
	//return cuda_menger_prison_de(iterations, ray);
}

__device__ float3 normal(float3 pos, float lod, double time) {
	const float2 b = make_float2(1.0,-1.0);
	const float3 b1 = make_float3(b.x, b.y, b.y); 
	const float3 b2 = make_float3(b.y, b.y, b.x); 
	const float3 b3 = make_float3(b.y, b.x, b.y); 
	const float3 b4 = make_float3(b.x, b.x, b.x);
	const float m = 0.000005;

    float p1 = map(pos + b1 * m, lod, time);
    float p2 = map(pos + b2 * m, lod, time);
    float p3 = map(pos + b3 * m, lod, time);
    float p4 = map(pos + b4 * m, lod, time);
	
	return normalize(b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4);
}

__device__ float shadow(float3 pos, float3 dir, float minDist, float maxDist, float k, float lod, double time) {
	float res = 1.0;
	float dist = minDist;
	
    for(int i = 0; i < 180 && dist < maxDist; i++) {
		float h = map(pos + dir * dist, lod, time);
    
		res = min(res, k * h / dist);
		dist += max(h, 0.0f);

		if(h < 0.0001)
			return 0.3;
    }
	
    return 1.0;
}

__device__ __forceinline__ static float ambientOcclusion(float3 pos, float3 normal, float lod, double time) {
    float AO = 1.0f;
	const float step_dist = 0.015;
	for (int step = 15; step > 0; step--) {
		AO -= pow(step * step_dist - map(pos + normal * step * step_dist, lod, time), 2) / step;
    }

    return clamp(0.3, 1.0, AO);
}

__device__ float3 specular(float3 viewDir, float3 norm, float3 lightDir, float3 lightColor) {
	float3 reflectDir = reflect(-lightDir, norm); 
	float spec = 0.9f * powf(max(dot(viewDir, reflectDir), 0.0), 32);

	return lightColor * spec;
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
		float d = map(currentPos, 1.0, time);
		
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

__device__ float3 renderPixel(Fractal frac, float3 cameraPos, float3 dir, double time) {

	HitInfo h = march(frac, cameraPos, dir, time);
	
	const float3 backgroundColor = make_float3(100, 100, 100) / 255.0f;
	if (h.steps < 0) {
		return backgroundColor;
	}


	const float3 lightDir = normalize(make_float3(-0.12, 0.34, 0.6));
	const float3 lightColor = make_float3(140, 80, 0) / 256.0f;
	const float3 materialColor = make_float3(240, 180, 180) / 256.0f;
	
	float lod = abs(h.dist);
	float fog = pow(1.0 - h.dist / MAX_DIST, 2.0);

	float3 viewDir = normalize(lightDir - dir);
	float3 n = normal(h.pos, lod, time);
	
	const float3 ambient = make_float3(75, 40, 5) / 256.0f;
	float s = shadow(h.pos, lightDir, 0.001, 3.0, 128, lod, time);
	float3 dif = clamp(dot(n, lightDir), 0.0f, 1.0f) * lightColor;
	float dif2 = clamp(dot(n, viewDir), 0.0f, 1.0f) * 0.7f + 0.3f;
	float AO = pow(ambientOcclusion(h.pos, n, lod, time), 40);
	float3 spec = specular(viewDir, n, lightDir, lightColor);


	float3 color = spec + ambient + dif;

	color = clamp(color * fog, 0.0f, 1.0f);
	return 255.0f * color;
}

__global__ void renderScene(Fractal frac, Scene scene, CameraDTO camera, double time) {
	size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < scene.width && y < scene.height) {
		float min_w_h = (float) min(scene.width, scene.height);

		float ar = (float) scene.width / (float) scene.height;

		#define AA 2.0
		
		float3 color = make_float3(0, 0, 0);
		for(int j = 0; j < AA; j++) {
			for(int i = 0; i < AA; i++) {
				
				float u = ((float) i / AA + (float) x) / min_w_h - ar * 0.5f;
				float v = ((float) j / AA + (float) y) / min_w_h - 0.5f;
		
				float3 dir = normalize(to_float3(camera.forward) + u * to_float3(camera.right)  + v * to_float3(camera.up));
				
				color += renderPixel(frac, to_float3(camera.pos), dir, time);
			}
		}

		color /= (float) AA * AA;

		scene.buf[y * scene.width + x] = make_uchar4((unsigned char) color.x, (unsigned char) color.y, (unsigned char) color.z, 255);
	}
}

void cudaDraw(Fractal frac, CameraDTO camera, struct cudaGraphicsResource *pboCuda, int width, int height, double time) {
    Scene scene(NULL, width, height);
    size_t size;

    cudaGraphicsMapResources(1, &pboCuda, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&scene.buf, &size, pboCuda);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    renderScene<<<grid, block>>>(frac, scene, camera, time);

    cudaGraphicsUnmapResources(1, &pboCuda, 0);
}
