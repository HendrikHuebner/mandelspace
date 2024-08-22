#pragma once
#include "../util.hpp"

struct Scene {
public:

    uchar4 *buf;
    unsigned int width;
    unsigned int height;

    Scene(uchar4 *buf, unsigned int width, unsigned int height) : buf(buf), width(width), height(height) {}

};


inline __host__ __device__ float3 to_float3(vec a)
{
	return make_float3((float) a.x, (float) a.y, (float) a.z);
}
