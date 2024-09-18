#include "fractals.hpp"

#define SIZE 2.0

__device__ __forceinline__ static float cube(float3 rayPos) {
    const float3 corner = make_float3(SIZE, SIZE, SIZE);

    float3 foldedPos = fabs(rayPos);
    float3 ctr = foldedPos - corner;
    float3 closestToOutsideRay = fmaxf(ctr, make_float3(0.0));
    float cornerToRayMaxComponent = max(max(ctr.x, ctr.y), ctr.z);
    float distToInsideRay = min(cornerToRayMaxComponent, 0.0);

    return length(closestToOutsideRay) + distToInsideRay;
}

__device__ __forceinline__  static float cross(float3 pos) {
    const float3 corner = make_float3(SIZE, SIZE, SIZE);
    float3 foldedPos = fabs(pos);
    float3 ctr = foldedPos - corner;

    float minComp = min(min(ctr.x, ctr.y), ctr.z);
    float maxComp = max(max(ctr.x, ctr.y), ctr.z);

    float midComp = ctr.x + ctr.y + ctr.z - minComp - maxComp;

    float2 closestOutsidePoint = fmaxf(make_float2(minComp, midComp), make_float2(0, 0));
    float2 closestInsidePoint = fminf(make_float2(midComp, maxComp), make_float2(0, 0));

    return length(closestOutsidePoint) - length(closestInsidePoint);
}

// https://connorahaskins.substack.com/p/ray-marching-menger-sponge-breakdown
__device__ float cuda_menger_prison_de(int iterations, float3 pos) {
    const float cubeWidth = 2 * SIZE;
    const float oneThird = 1.0 / 3.0;

    float3 rayPos = sign(pos) * fmodf(pos, 2 * SIZE) - SIZE;

    float spongeCube = cube(rayPos);
    float mengerSpongeDist = spongeCube;

    float scale = 1.0;

    for (int i = 0; i < iterations; i++) {
        float boxedWidth = cubeWidth / scale;

        float translation = -boxedWidth / 2.0;

        float3 ray = rayPos - translation;

        float3 repeatedPos = make_float3(rayPos.x - boxedWidth * floor(ray.x / boxedWidth),
                                         rayPos.y - boxedWidth * floor(ray.y / boxedWidth),
                                         rayPos.z - boxedWidth * floor(ray.z / boxedWidth));

        scale *= 3.0;
        repeatedPos = repeatedPos * scale;

        float crossDist = cross(repeatedPos) / scale;

        mengerSpongeDist = max(mengerSpongeDist, -crossDist);
    }

    return mengerSpongeDist;
}
