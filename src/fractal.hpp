#pragma once

#include <cstdint>

class Fractal {
public:
    float distanceEst(float x, float y, float z) { return 0.0; };
};

class Mandelbulb : public Fractal {
    
public:
    float p;
    float q;

    Mandelbulb(float p, float q) : p(p), q(q) {}

    float distanceEst(float x, float y, float z) {
        return 0.0;
    };
};
