#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "inputs.hpp"
#include "util.hpp"

void drawCameraOrientation(const Camera& camera) {
    return;
    const float length = 0.5f;

    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f); // Red for forward
    glVertex3f(0.0f, 0.0f, 0.0f);
    
    glVertex3f(length, 0.0f, 0.0f);
    glEnd();

    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 0.0f); // Green for right
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, length, 0.0f);
    glEnd();

    glBegin(GL_LINES);
    glColor3f(0.0f, 0.0f, 1.0f); // Blue for up
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, length);
    glEnd();
}
