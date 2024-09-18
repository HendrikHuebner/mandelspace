#pragma once

#include <cuda_runtime.h>

#include "../util.hpp"
#include "../vector_ops.hpp"

__device__ float cuda_mandelbox_de(int iterations,  float mu, float fR, float mR, float3 pos);

__device__ float cuda_mandelbulb_de(const int iterations, float3 pos, double pow);

__device__ float cuda_menger_prison_de(int iterations, float3 pos);
