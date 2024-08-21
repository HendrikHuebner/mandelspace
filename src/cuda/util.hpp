#pragma once

struct Camera {

public:
    
    float3 pos;
    float3 dir;

    Camera(float3 pos, float3 dir) : pos(pos), dir(dir) {}
};

struct Scene {
public:

    uchar4 *buf;
    unsigned int width;
    unsigned int height;

    Scene(uchar4 *buf, unsigned int width, unsigned int height) : buf(buf), width(width), height(height) {}

};
