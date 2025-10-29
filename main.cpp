#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

// window size callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// input handler
void processInput(GLFWwindow* window) {
    // press 'esc', then close window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main(void) {
    // GLFW init
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL 4.5 Core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "GPUDestruction Project - Hello Window!", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // window context <= current thread context
    glfwMakeContextCurrent(window);

    // window size callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // GLAD init
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Render Loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // screen color
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        // clear color buffer
        glClear(GL_COLOR_BUFFER_BIT);

        // some code

        // swap double buffer
        glfwSwapBuffers(window);
        // check event
        glfwPollEvents();
    }

    // Clear resources
    glfwTerminate();
    return 0;
}