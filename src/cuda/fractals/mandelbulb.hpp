#pragma once

#include "../vector_ops.hpp"
#include "../util.hpp"

#define THRESHOLD 4
#define ITERATIONS 8
#define POW 8.0

// https://github.com/ichko/cuda-mandelbulb/blob/master/main.cu

__device__ float cuda_mandelbulb_de(float p, float q, float3 pos) {
    
	float3 z = pos;
	float dr = 1.0;
	float r = 0.0;

	for(int i = 0; i < ITERATIONS; i++) {
		r = length(z);

		if (r > THRESHOLD) 
			break;

		float theta = acos(z.z / r);
		float phi = atan2(z.y, z.x);

		dr = powf(r, POW - 1.0) * POW * dr + 1.0;

		float zr = pow(r, POW);
		theta *= POW;
		phi *= POW;

		z = make_float3(
				sin(theta) * cos(phi),
				sin(phi) * sin(theta), 
				cos(theta)
			) * zr + pos;
	}

	return 0.5 * log(r) * r / dr;
}
