#pragma once
#include "../util.hpp"

struct Scene {
public:

    uchar4 *buf;
    const unsigned int width;
    const unsigned int height;

    __host__ Scene(uchar4 *buf, unsigned int width, unsigned int height) : 
        buf(buf), width(width), height(height) {}

};


inline __host__ __device__ float3 to_float3(vec3F a)
{
	return make_float3((float) a.x, (float) a.y, (float) a.z);
}

inline __host__ __device__ float3 sign(float3 a)
{
	return make_float3((a.x > 0) ? 1.0 : ((a.x < 0) ? -1.0 : 0.0),
                        (a.y > 0) ? 1.0 : ((a.y < 0) ? -1.0 : 0.0),
                        (a.z > 0) ? 1.0 : ((a.z < 0) ? -1.0 : 0.0));
}


inline __device__ __host__ float3 sin(float3 v)
{
	return make_float3(sin(v.x), sin(v.y), sin(v.z));
}
__device__ struct Config {
    const int iterations;
    const Fractal fractal;
    const float mandelbulb_power;
    const float mandelbox_mu;
    const float mandelbox_mR;
    const float mandelbox_fR;
    const float3 backgroundColor;

    const float3 ambientColor;
    const float3 lightDirection;
    const float3 lightColor;
    const float3 materialColor;
    const float directionalLightStrength;
    const float cameraLightStrength;
    const int ambientOcclusionLevels;
    const float specularLightStrength;
    const int specularLightFocus;
    const float fogDistance;

    // fractal quality
    const int maxMarchingSteps;
    const int maxShadowSteps;
    const int antiAliasing;
};
