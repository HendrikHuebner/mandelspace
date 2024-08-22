#pragma once 

typedef struct vec {
    float x;
    float y;
    float z;
} vec;

typedef struct  {
    vec pos;
    vec dir;
} Ray;
