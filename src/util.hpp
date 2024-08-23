#pragma once 


struct vec3F {
    float x;
    float y;
    float z;
};

struct Ray {
    vec3F pos;
    vec3F dir;
};

struct CameraDTO {
    vec3F pos;
    vec3F forward;
    vec3F right;
    vec3F up;
};
