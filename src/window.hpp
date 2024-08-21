#pragma once 

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "fractal.hpp"

extern void cudaDraw(Fractal frac, struct cudaGraphicsResource* d_data, int width, int height);

int windowWidth = 1024;
int windowHeight = 1024;

void initPBO(GLuint *pbo) {
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void display(GLuint pbo) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

int initWindow(GLFWwindow **window) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    *window = glfwCreateWindow(windowWidth, windowHeight, "CUDA OpenGL Interop", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(*window);
    glewInit();

    return 0;
}

void teardownWindow(GLFWwindow *window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}


int renderLoop(Fractal frac) {
    GLFWwindow *window;

    if (initWindow(&window)) {
        std::cerr << "Failed to initilize GLFW window" << std::endl;
    }

    // Create PBO
    GLuint pbo;
    struct cudaGraphicsResource* pboCuda;
    initPBO(&pbo);
    cudaGraphicsGLRegisterBuffer(&pboCuda, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // Main loop
    int frameCount = 0;
    double prevTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {

        cudaDraw(frac, pboCuda, windowWidth, windowHeight);
        display(pbo);

        frameCount++;
        double currentTime = glfwGetTime();

        if (currentTime - prevTime >= 1.0) {
            double fps = frameCount / (currentTime - prevTime);
            std::cout << "FPS: " << fps << std::endl;
            prevTime = currentTime;
            frameCount = 0;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(pboCuda);
    glDeleteBuffers(1, &pbo);

    teardownWindow(window);

    return 0;
}
