

#define zero3 make_float3(0, 0, 0)
#define one3 make_float3(1, 1, 1)

static __device__ float shadow(float3 pos, float3 dir, float minDist, float maxDist, float k, double time) {
	float res = 1.0;
	float dist = minDist;
    for(int i = 0; i < 256 && dist < maxDist; i++) {
		float h = map(pos + dir * dist, time);
    
		res = min(res, k * h / dist);
		dist += h;

		if(h < 0.0001)
			return 0.3;
    }
	
    return 1.0;
}

__device__ float3 lightScene(float3 hitPos, float3 ray, float3 normal, double time) {
    const float3 lightDir = normalize(make_float3(-0.1, 0.3, 0.6));
    const float3 lightColor = normalize(make_float3(-0.1, 0.3, 0.6));

    const float3 ambient = make_float3(75, 50, 10) / 256.0f;

    float s = shadow(hitPos, lightDir, 0.001, 3.0, 128, time);
    float dif = clamp(dot(normal, normalize(lightDir - ray)), 0.0, 1.0);

    float3 color = dif * s * make_float3(1.0, 0.7, 0.5) + ambient;

    color = 255.0f * clamp(color, zero3, one3);

    return color;
}