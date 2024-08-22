
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

const float cameraSpeed = 2.5f;
const float mouseSensitivity = 0.1f;

struct Camera {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 up;
    float yaw;
    float pitch;
};

// global instance of the camera
static Camera camera;

void processInput(GLFWwindow* window, float deltaTime) {
    glm::vec3 forward = glm::normalize(camera.direction);
    glm::vec3 right = glm::normalize(glm::cross(forward, camera.up));
    glm::vec3 up = camera.up;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.position += forward * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.position -= forward * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.position -= right * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.position += right * cameraSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.position += up * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.position -= up * cameraSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

static void handleMouseMovement(GLFWwindow* window, double xpos, double ypos) {
    static bool firstMouse = true;
    static float lastX = 400, lastY = 300;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xOffset = xpos - lastX;
    float yOffset = lastY - xpos;
    lastX = xpos;
    lastY = ypos;

    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    camera.yaw += xOffset;
    camera.pitch += yOffset;

    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    direction.y = sin(glm::radians(camera.pitch));
    direction.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    camera.direction = glm::normalize(direction);
}

Camera initCamera() {
    camera.position = glm::vec3(-3.0f, 0.0f, 0.0f);
    camera.direction = glm::vec3(1.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera.yaw = -90.0f;
    camera.pitch = 0.0f;

    return camera;
}

void glfwInitInputs(GLFWwindow *window) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, handleMouseMovement );
}
