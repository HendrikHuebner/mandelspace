#include "fractals.hpp"

#define THRESHOLD 4

// https://github.com/ichko/cuda-mandelbulb/blob/master/main.cu

__device__ float cuda_mandelbulb_de(const int iterations, float3 pos, double p) {

	float3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	const float power = 8;

	for(int i = 0; i < 5; i++) {
		r = length(z);

		if (r > THRESHOLD) 
			break;

		float theta = acos(z.z / r);
		float phi = atan2(z.y, z.x);

		dr = powf(r, power - 1.0) * power * dr + 1.0;

		float zr = pow(r, power);
		theta *= power;
		phi *= power;

		z = make_float3(
				sin(theta) * cos(phi),
				sin(phi) * sin(theta), 
				cos(theta)
			) * zr + pos;

	}

	return 0.5 * log(r) * r / dr;
}
