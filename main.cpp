#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern const int NUM_INSTANCES;

// vertex shader
const char* vertexShaderSource = R"glsl(
    #version 450 core
    
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    layout (location = 2) in mat4 model;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 ourColor;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor;
    }
)glsl";

// fragment shader
const char* fragmentShaderSource = R"glsl(
    #version 450 core
    in vec3 ourColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(ourColor, 1.0);
    }
)glsl";

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

void cpu_update_physics(std::vector<glm::mat4>& modelMatrices, float time)
{
    for (int i = 0; i < NUM_INSTANCES; ++i)
    {
        // simple animation
        float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f;
        float angle = time * 0.5f + (float)i * 0.01f;

        float x = radius * cos(angle);
        float y = 0.0f;
        float z = radius * sin(angle);

        glm::vec3 position = glm::vec3(x, y, z);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::rotate(model, time * 2.0f + (float)i, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(0.1f));

        // 將計算好的矩陣存入 CPU 的 vector
        modelMatrices[i] = model;
    }
}

int main() {
    // init
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "GPUDestruction Step 5: CUDA + OpenGL Interop!", NULL, NULL);
    if (window == NULL) { /* ... error check ... */ }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    unsigned int shaderProgram = glCreateProgram();
    glLinkProgram(shaderProgram);

    // error check
    int success; char infoLog[512];
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL); glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success); if (!success) { glGetShaderInfoLog(vertexShader, 512, NULL, infoLog); std::cerr << "VS COMPILE ERROR: " << infoLog << std::endl; }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL); glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success); if (!success) { glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog); std::cerr << "FS COMPILE ERROR: " << infoLog << std::endl; }
    glAttachShader(shaderProgram, vertexShader); glAttachShader(shaderProgram, fragmentShader); glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success); if (!success) { glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog); std::cerr << "SHADER LINK ERROR: " << infoLog << std::endl; }
    glDeleteShader(vertexShader); glDeleteShader(fragmentShader);


    // cude vertex data
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f, // 0
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f, // 1
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f, // 2
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f, // 3
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f, // 4
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f, // 5
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f, // 6
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f  // 7
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

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Instance VBO
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    glBufferData(GL_ARRAY_BUFFER, NUM_INSTANCES * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);

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
    std::vector<glm::mat4> modelMatrices(NUM_INSTANCES);

    // Render Loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        //  在 CPU 上計算
        cpu_update_physics(modelMatrices, (float)glfwGetTime());

        // 將資料從 CPU 上傳到 GPU
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_INSTANCES * sizeof(glm::mat4), modelMatrices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // render command
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 15.0f, 30.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 100.0f);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, NUM_INSTANCES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}