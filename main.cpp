#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Shader.h"
#include "PhysicsEngine.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern const int NUM_INSTANCES;

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
    // init
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "GPUDestruction Project - Modular Engine!", NULL, NULL);
    if (window == NULL) { /* ... error check ... */ return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    Shader ourShader("shader.vert", "shader.frag");
    PhysicsEngine physicsEngine; // 建立物理引擎

    // cude vertex data
    float vertices[] = {
        // 後面
        -0.5f, -0.5f, -0.5f,  0.5f, 0.5f, 0.5f, // 0
         0.5f, -0.5f, -0.5f,  0.5f, 0.5f, 0.5f, // 1
         0.5f,  0.5f, -0.5f,  0.5f, 0.5f, 0.5f, // 2
        -0.5f,  0.5f, -0.5f,  0.5f, 0.5f, 0.5f, // 3

        // 前面
        -0.5f, -0.5f,  0.5f,  0.5f, 0.5f, 0.5f, // 4
         0.5f, -0.5f,  0.5f,  0.5f, 0.5f, 0.5f, // 5
         0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f, // 6
        -0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f  // 7
    };

    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0,
        4, 5, 6,  6, 7, 4,
        7, 6, 2,  2, 3, 7,
        4, 5, 1,  1, 0, 4,
        4, 7, 3,  3, 0, 4,
        5, 6, 2,  2, 1, 5 
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // VBO (頂點資料)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // EBO (索引資料)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 頂點屬性 (location 0: pos, location 1: color)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Instance VBO
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_INSTANCES * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);

    // Instance 屬性 (location 2, 3, 4, 5: model matrix)
    std::size_t vec4Size = sizeof(glm::vec4);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(vec4Size));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * vec4Size));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * vec4Size));

    glVertexAttribDivisor(2, 1);
    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);

    glBindVertexArray(0);
    
    // Render Loop
    float lastFrameTime = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        processInput(window);

        //  呼叫物理引擎來更新所有物體
        physicsEngine.update(currentTime, deltaTime);

        // get matrix
        const std::vector<glm::mat4>& modelMatrices = physicsEngine.getModelMatrices();
        // 將資料上傳到 GPU
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_INSTANCES * sizeof(glm::mat4), modelMatrices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // render command
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 啟用 Shader
        ourShader.use();

        // view matrix
        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 30.0f, 50.0f), // 攝影機位置 (更高、更遠)
            glm::vec3(0.0f, 0.0f, 0.0f),   // 攝影機看向原點
            glm::vec3(0.0f, 1.0f, 0.0f)    // 上方向
        );
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 200.0f);
        ourShader.setMat4("view", view);
        ourShader.setMat4("projection", projection);

        // draw all instances
        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, NUM_INSTANCES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &instanceVBO);

    glfwTerminate();
    return 0;
}