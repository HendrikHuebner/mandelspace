#pragma once

#include "../util.hpp"
#include "../vector_ops.hpp"



__device__ __forceinline__ float cuda_mandelbox_de(int iterations,  float mu, float fR, float mR, float3 pos) {
	float3 z = pos;
	float dr = mu;

	for (int n = 0; n < iterations; n++) {

        // box fold
		z = clamp(z, -1.0, 1.0) * 2.0 - z;

		// sphere fold
        const float r2 = dot(z, z);
    
        if (r2 < mR) {
            z *= fR / mR;
            dr *= fR / mR;

        } else if (r2 < fR) {
            z *= fR / r2;
            dr *= fR / r2;
        }

        z = mu * z + pos;
        dr = dr * mu;
	}

	return length(z) / dr;
}
