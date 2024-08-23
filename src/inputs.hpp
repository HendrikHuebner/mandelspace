
#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


struct Camera {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 up;
    glm::vec3 right;
};

// global instance of the camera
static Camera camera;

void handleKeyInputs(GLFWwindow* window, float deltaTime) {
    const float cameraSpeed = 0.005f;

    glm::vec3 forward = glm::normalize(camera.direction);
    camera.right = glm::normalize(glm::cross(forward, camera.up));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.position += forward * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.position -= forward * cameraSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.position -= camera.right * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.position += camera.right * cameraSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.position += glm::vec3(0.0f, 0.0f, 1.0f) * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.position -= glm::vec3(0.0f, 0.0f, 1.0f) * cameraSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void handleMouseMovement(GLFWwindow* window, double xpos, double ypos) {
    const float mouseSensitivity = 0.01f;

    static bool firstMouse = true;
    static float lastX = 400, lastY = 300;
    static float yaw = -90.0f;
    static float pitch = 0.0f;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xOffset = xpos - lastX;
    float yOffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    return;
    camera.direction = glm::normalize(direction);
    camera.right = glm::cross(camera.direction, glm::vec3(0.0f, 0.0f, 1.0f));
    camera.up = glm::cross(camera.right, camera.direction);
}

Camera& initCamera() {
    camera.position = glm::vec3(-3.0f, 0.0f, 0.0f);
    camera.direction = glm::vec3(1.0f, 0.0f, 0.0f);
    camera.right = glm::vec3(0.0f, 1.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 0.0f, 1.0f);

    return camera;
}

void glfwInitInputs(GLFWwindow *window) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, handleMouseMovement );
}
