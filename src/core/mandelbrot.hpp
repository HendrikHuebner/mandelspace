#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

static const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)";

static const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;

    uniform vec2 u_resolution;
    uniform vec2 u_center;
    uniform float u_zoom;

    int mandelbrot(vec2 c) {
        vec2 z = vec2(0.0);
        int iterations = 0;
        const int maxIterations = 1000;

        for (int i = 0; i < maxIterations; i++) {
            if (dot(z, z) > 4.0) break;
            z = vec2(
                z.x * z.x - z.y * z.y + c.x,
                2.0 * z.x * z.y + c.y
            );
            iterations++;
        }
        return iterations;
    }

    void main() {
        vec2 coord = (gl_FragCoord.xy - u_resolution / 2.0) / u_zoom + u_center;
        int iterations = mandelbrot(coord);
        float brightness = 1 / float(iterations);
        
        FragColor = vec4(vec3(color, color, color), 1.0);
    }
)";

static GLuint createShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed\n" << infoLog << std::endl;
    }
    return shader;
}

static GLuint createProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program Linking Failed\n" << infoLog << std::endl;
    }
    return program;
}

int render_mandelbrot() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Mandelbrot Set", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Vertex shader
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSource);
    
    // Fragment shader
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Shader program
    GLuint shaderProgram = createProgram(vertexShader, fragmentShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // single output square for the viewport
    float vertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f,
    };

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Set uniforms
        glUniform2f(glGetUniformLocation(shaderProgram, "u_resolution"), 1000, 800);
        glUniform2f(glGetUniformLocation(shaderProgram, "u_center"), -0.5f, 0.0f);
        glUniform1f(glGetUniformLocation(shaderProgram, "u_zoom"), 200.0f);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
