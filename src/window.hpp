#pragma once 

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "fractal.hpp"
#include "inputs.hpp"
#include "util.hpp"
#include "ui.hpp"

extern void cudaDraw(Fractal frac, CameraDTO camera, struct cudaGraphicsResource* d_data, int width, int height, double time);

const int windowWidth = 1012;
const int windowHeight = 512;

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

    *window = glfwCreateWindow(windowWidth, windowHeight, "Mandeldspace", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(*window);
    glfwSwapInterval(0);

    glewInit();

    glfwInitInputs(*window);

    return 0;
}

void closeWindow(GLFWwindow *window) {
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

    Camera& camera = initCamera();

    double t0 = glfwGetTime();
    int frameCount = 0;
    double prevTime = t0;
    double prevMeasureFPSStart = t0;

    while (!glfwWindowShouldClose(window)) {
    
        frameCount++;
        double currentTime = glfwGetTime();
        float deltaTime = currentTime - prevTime;
        prevTime = currentTime;

        CameraDTO camDto = {
            .pos = {camera.position.x, camera.position.y, camera.position.z },
            .forward = {camera.direction.x, camera.direction.y, camera.direction.z },
            .right = {camera.right.x, camera.right.y, camera.right.z },
            .up = {camera.up.x, camera.up.y, camera.up.z },
        };

        cudaDraw(frac, camDto, pboCuda, windowWidth, windowHeight, currentTime - t0);
        
        display(pbo);
        drawCameraOrientation(camera);

        if (currentTime - prevMeasureFPSStart >= 1.0) {
            double fps = frameCount / (currentTime - prevMeasureFPSStart);
            std::cout << "FPS: " << fps << std::endl;
            prevMeasureFPSStart = currentTime;
            frameCount = 0;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
        handleKeyInputs(window, deltaTime);

    }

    cudaGraphicsUnregisterResource(pboCuda);
    glDeleteBuffers(1, &pbo);

    closeWindow(window);

    return 0;
}
