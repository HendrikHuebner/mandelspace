#include "fractals/fractals.hpp"
#include "fractals/menger_sponge.hpp"
#include "util.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define EPSILON 0.002

__device__ float map(Config config, float3 ray, float lod, double t) {
	switch (config.fractal) {

		case MENGER_PRISON: {
			int iterations = (int) (clamp(2.2f + 4.0f * exp(0.3 - 0.2 * lod), 3.0f, (float) config.iterations));
			return cuda_menger_prison_de(iterations, ray);
		}
		
		case MENGER_SPONGE: {
			int iterations = (int) (clamp(2.2f + 4.0f * exp(0.3 - 0.2 * lod), 3.0f, (float) config.iterations));
			return cuda_menger_sponge_de(iterations, ray);
		}
		case MANDELBULB: {
			//float pow = (config.mandelbulb_power > 0) ? config.mandelbulb_power : 1.0 + 0.1 * abs(2 * fmod(t, 30.0) - fmod(t, 60.0));
			
			
			return cuda_mandelbulb_de(config.iterations, ray, 0);
		}
		case MANDELBOX: {
			return cuda_mandelbox_de(config.iterations, config.mandelbox_mu, config.mandelbox_mR, config.mandelbox_fR, ray);
		}

	}
}

__device__ float3 normal(Config config, float3 pos, float lod, double time) {
	const float2 b = make_float2(1.0,-1.0);
	const float3 b1 = make_float3(b.x, b.y, b.y); 
	const float3 b2 = make_float3(b.y, b.y, b.x); 
	const float3 b3 = make_float3(b.y, b.x, b.y); 
	const float3 b4 = make_float3(b.x, b.x, b.x);
	const float m = 0.000005;

    float p1 = map(config, pos + b1 * m, lod, time);
    float p2 = map(config, pos + b2 * m, lod, time);
    float p3 = map(config, pos + b3 * m, lod, time);
    float p4 = map(config, pos + b4 * m, lod, time);
	
	return normalize(b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4);
}

__device__ float shadow(Config config, float3 pos, float3 dir, float minDist, float maxDist, float lod, double time) {
	float dist = minDist;
	
    for(int i = 0; i < 180 && dist < maxDist; i++) {
		float h = map(config, pos + dir * dist, lod, time);
		dist += max(h, 0.0f);

		if(h < 0.0001)
			return 0.0f;
    }
	
    return 1.0;
}

__device__ __forceinline__ static float ambientOcclusion(Config config, float3 pos, float3 normal, float lod, double time) {
	float step_dist = 0.015;
	float occlusion = 1.0f;
	
    for (int step = 15; step > 0; step--) {
		occlusion -= pow(step * step_dist - map(config, pos + normal * step * step_dist, lod, time),2) / step;
    }

    return pow(occlusion, 24.0);
}

__device__ float3 specular(float3 viewDir, int specularLightRadius, float3 norm, float3 lightDir, float3 lightColor) {
	float3 reflectDir = reflect(-lightDir, norm); 
	float spec = powf(max(dot(viewDir, reflectDir), 0.0), specularLightRadius);

	return lightColor * spec;
}

struct HitInfo {
	int steps;
	float dist;
	float3 pos;
};


__device__ HitInfo march(Config config, float3 pos, float3 dir, double time) {
	float distance = 0.0;
	int steps;
	float3 currentPos;
	
	for (steps = 0; steps < config.maxMarchingSteps && distance < config.maxMarchingSteps; steps++) {
		currentPos = pos + dir * distance;
		float d = map(config, currentPos, 1.0, time);
		
		if (config.fractal == MANDELBOX) {
			if (d < 0.002) {
				HitInfo h = {
					.steps = steps,
					.dist = distance,
					.pos = currentPos
				};

				return h;
			}
		} else {
			if (d < 0.00002) {
				HitInfo h = {
					.steps = steps,
					.dist = distance,
					.pos = currentPos
				};

				return h;
			}
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

__device__ float3 renderPixel(float3 forward, float3 cameraPos, float3 dir, Config config, double time) {
	HitInfo h = march(config, cameraPos, dir, time);

	if (h.steps < 0) {
		return config.backgroundColor;
	}

	float lod = abs(h.dist);

	float fog = clamp(pow(h.dist / config.fogDistance, 5.0), 0.0f, 1.0f);
	
	const float3 ambient =  clamp(config.ambientColor / 255.0f, 0.0f, 1.0f);
	const float3 lightCol = config.lightColor / 255.0f;
	const float3 lightDir = normalize(config.lightDirection);
	float3 viewDir = normalize(cameraPos - h.pos);

	float3 n = normal(config, h.pos, lod, time);

	float s = config.maxShadowSteps == 0 ? 1.0 : shadow(config, h.pos, lightDir, 0.001, 3.0, lod, time);
	float3 dif = clamp(dot(n, lightDir), 0.0f, 1.0f) * lightCol * config.directionalLightStrength;
	float3 dif2 = clamp(sqrt(dot(n, viewDir)), 0.1f, 1.0f) * lightCol / max(h.dist, 0.8) * config.cameraLightStrength;
	
	float AO = config.ambientOcclusionLevels == 0 ? 1.0 : ambientOcclusion(config, h.pos, n, lod, time);
	
	float3 spec;
	if (config.directionalLightStrength > 0.0)
		spec = specular(viewDir, config.specularLightFocus, n, lightDir, lightCol);
	else
		spec = specular(viewDir, config.specularLightFocus, n, viewDir, lightCol);

	float3 color =  dif2 + (dif + config.specularLightStrength * spec) * s + ambient;
	color = clamp(color * (1.0 - fog) * max(AO, 0.3f) + config.backgroundColor / 255.0f * fog, 0.0f, 1.0f);
	return color * 255.0f;
}

__global__ void renderScene(Scene scene, CameraDTO camera, Config config, double time) {
	size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < scene.width && y < scene.height) {
		float min_w_h = (float) min(scene.width, scene.height);

		float ar = (float) scene.width / (float) scene.height;

		const int AA_X = config.antiAliasing % 2 + 1;
		const int AA_Y = config.antiAliasing / 2 + 1;
		
		float3 color = make_float3(0, 0, 0);
		for(int j = 0; j < AA_X; j++) {
			for(int i = 0; i < AA_Y; i++) {
				
				float u = ((float) i / AA_X + (float) x) / min_w_h - ar * 0.5f;
				float v = ((float) j / AA_Y + (float) y) / min_w_h - 0.5f;

				float3 dir = normalize(to_float3(camera.forward) + u * to_float3(camera.right)  + v * to_float3(camera.up));
				
				color += renderPixel(to_float3(camera.forward), to_float3(camera.pos), dir, config, time);
			}
		}

		color /= (float) AA_X * AA_Y;

		scene.buf[y * scene.width + x] = make_uchar4((unsigned char) color.x, (unsigned char) color.y, (unsigned char) color.z, 255);
	}
}

void cudaDraw(CameraDTO camera, struct cudaGraphicsResource *pboCuda, int width, int height, void *config, double time) {
    Scene scene(NULL, width, height);
    size_t size;

    cudaGraphicsMapResources(1, &pboCuda, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&scene.buf, &size, pboCuda);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	Config cfg = *(Config *) config;
    renderScene<<<grid, block>>>(scene, camera, cfg, time);

    cudaGraphicsUnmapResources(1, &pboCuda, 0);
}
