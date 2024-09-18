#pragma once

#include "../vector_ops.hpp"
#include "../util.hpp"

#define ena true

__device__ float cuda_sierpinski_de(const int iterations, float3 z, double time) {
	// Normal full tetra-fold
	if (ena)
	{
		float temp = 0.0f;
		if (z.x + z.y < 0.0f)
		{
			temp = -z.y;
			z.y = -z.x;
			z.x = temp;
		}
		if (z.x + z.z < 0.0f)
		{
			temp = -z.z;
			z.z = -z.x;
			z.x = temp;
		}
		if (z.y + z.z < 0.0f)
		{
			temp = -z.z;
			z.z = -z.y;
			z.y = temp;
		}
	}

	// Reversed full tetra-fold
	if (!ena)
	{
		if (z.x - z.y < 0.0f)
		{
			float temp = z.y;
			z.y = z.x;
			z.x = temp;
		}
		if (z.x - z.z < 0.0f)
		{
			float temp = z.z;
			z.z = z.x;
			z.x = temp;
		}
		if (z.y - z.z < 0.0f)
		{
			float temp = z.z;
			z.z = z.y;
			z.y = temp;
		}
	}

#define scaleA2 0.8
	z *= scaleA2;

	return z;
}
