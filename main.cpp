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

// 陰影貼圖的解析度
const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

// FBO 和 紋理 ID
unsigned int depthMapFBO;
unsigned int depthMapTexture;

// 光源位置
glm::vec3 lightPos(0.0f, 40.0f, 20.0f); // 調整這個來改變陰影方向
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

	Shader ourShader("shader.vert", "shader.frag"); // main shader
    Shader depthShader("depth.vert", "depth.frag"); // depth shader
    PhysicsEngine physicsEngine; // 建立物理引擎

    // vertex data
    float vertices[] = {
        // positions          // colors           // texture coords
        // 後面
        -0.5f, -0.5f, -0.5f,  0.5f, 0.5f, 0.5f,   0.0f, 0.0f, // 0
         0.5f, -0.5f, -0.5f,  0.5f, 0.5f, 0.5f,   1.0f, 0.0f, // 1
         0.5f,  0.5f, -0.5f,  0.5f, 0.5f, 0.5f,   1.0f, 1.0f, // 2
        -0.5f,  0.5f, -0.5f,  0.5f, 0.5f, 0.5f,   0.0f, 1.0f, // 3

        // 前面
        -0.5f, -0.5f,  0.5f,  0.5f, 0.5f, 0.5f,   0.0f, 0.0f, // 4
         0.5f, -0.5f,  0.5f,  0.5f, 0.5f, 0.5f,   1.0f, 0.0f, // 5
         0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f,   1.0f, 1.0f, // 6
        -0.5f,  0.5f,  0.5f,  0.5f, 0.5f, 0.5f,   0.0f, 1.0f  // 7
    };

    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0,
        4, 5, 6,  6, 7, 4,
        7, 6, 2,  2, 3, 7,
        4, 5, 1,  1, 0, 4,
        4, 7, 3,  3, 0, 4,
        5, 6, 2,  2, 1, 5 
    };

    // 設定陰影貼圖 FBO
    glGenFramebuffers(1, &depthMapFBO);
    // 建立深度紋理
    glGenTextures(1, &depthMapTexture);
    glBindTexture(GL_TEXTURE_2D, depthMapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
        SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // 設定「超出紋理」的部分為白色 (代表不在陰影中)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // 將深度紋理附加到 FBO
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTexture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // 解綁

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // 每個頂點的資料大小現在是 8 個 float (3 pos + 3 color + 2 texcoord)
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 頂點屬性 (location 0: pos, location 1: color)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 紋理座標屬性 (location 6)
    glVertexAttribPointer(6, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(6);


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

    // ground setup
    unsigned int groundVAO, groundVBO;
    glGenVertexArrays(1, &groundVAO);
    glGenBuffers(1, &groundVBO);
    glBindVertexArray(groundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, groundVBO);

    // 地面方塊的頂點資料 (一個 100x1x100 的巨大方塊)
    float groundVertices[] = {
        // positions          // colors           // texture coords
        // 背面 (Z 負軸)
        -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,

        // 正面 (Z 正軸)
        -0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,

        // 左面 (X 負軸)
        -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,

        // 右面 (X 正軸)
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,

         // 底面 (Y 負軸)
         -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
          0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
          0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
          0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
         -0.5f, -0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
         -0.5f, -0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,

         // 頂面 (Y 正軸)
         -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
          0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
          0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
          0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
         -0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
         -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 1.0f,   0.0f, 1.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(groundVertices), groundVertices, GL_STATIC_DRAW);

    // 設定頂點屬性指標
    // 位置 (aPos)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 顏色 (aColor)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // 紋理座標 (aTexCoord)
    glVertexAttribPointer(6, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(6);

    glBindVertexArray(0); // 解綁 VAO

    // Load Texture
    unsigned int texture1;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    // 設定紋理環繞和過濾選項
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load rock texture
    int width, height, nrChannels;
    unsigned char* data = stbi_load("rock_texture.jpg", &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format = GL_RGB;
        if (nrChannels == 4) format = GL_RGBA;

        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cerr << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // load ground texture
    unsigned int texture2;
    glGenTextures(1, &texture2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture2);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width2, height2, nrChannels2;
    unsigned char* data2 = stbi_load("ground_texture.png", &width2, &height2, &nrChannels2, 0);
    if (data2)
    {
        GLenum format;
        if (nrChannels2 == 1)
            format = GL_RED;
        else if (nrChannels2 == 3)
            format = GL_RGB;
        else if (nrChannels2 == 4)
            format = GL_RGBA;

        glTexImage2D(GL_TEXTURE_2D, 0, format, width2, height2, 0, format, GL_UNSIGNED_BYTE, data2);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load ground texture" << std::endl;
    }
    stbi_image_free(data2);
    
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
        // 計算光源矩陣
        // 使用正交投影來模擬平行光
        float near_plane = 1.0f, far_plane = 70.0f;
        glm::mat4 lightProjection = glm::ortho(-30.0f, 30.0f, -30.0f, 30.0f, near_plane, far_plane);
        glm::mat4 lightView = glm::lookAt(lightPos,
            glm::vec3(0.0f, 0.0f, 0.0f), // 光源看向原點
            glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 lightSpaceMatrix = lightProjection * lightView;

        // PASS 1 深度通道 (渲染到 FBO)
        depthShader.use();
        depthShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);

        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(VAO);
        ourShader.setInt("isGround", 0);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, NUM_INSTANCES);

        // 繪製地面方塊
        glm::mat4 groundModel = glm::mat4(1.0f);
        groundModel = glm::translate(groundModel, glm::vec3(0.0f, -2.5f, 0.0f)); // 放置在 Y=-2.5
        groundModel = glm::scale(groundModel, glm::vec3(100.0f, 1.0f, 100.0f)); // 100x1x100 巨大方塊
        ourShader.setInt("isGround", 1);
        depthShader.setMat4("model", groundModel); // 將 Model 矩陣傳給 Shader

        glBindVertexArray(groundVAO); // 綁定地面 VAO
        glDrawArrays(GL_TRIANGLES, 0, 36); // 繪製單一個方塊 (36 個頂點)
        glBindVertexArray(0); // 解綁
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // PASS 2 顏色通道 (渲染到螢幕)
        glViewport(0, 0, 1280, 720); // 恢復螢幕解析度
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 啟用 Shader
        ourShader.use();

        // 設定攝影機
        glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 30.0f, 50.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.1f, 200.0f);
        ourShader.setMat4("view", view);
        ourShader.setMat4("projection", projection);

        // 傳送光照 uniforms
        ourShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
        ourShader.setVec3("lightPos", lightPos);
        ourShader.setVec3("viewPos", glm::vec3(0.0f, 30.0f, 50.0f));

        // 綁定紋理
        ourShader.setInt("ourTexture", 0);
        ourShader.setInt("shadowMap", 1);
        ourShader.setInt("groundTexture", 2);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);       // 石塊紋理
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthMapTexture); // 陰影貼圖
        glActiveTexture(GL_TEXTURE2);                  // 地板貼圖
        glBindTexture(GL_TEXTURE_2D, texture2);

        // 石塊實例
        glBindVertexArray(VAO);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, NUM_INSTANCES);
        ourShader.setMat4("model", groundModel);

        glBindVertexArray(groundVAO); // 綁定地面 VAO
        glDrawArrays(GL_TRIANGLES, 0, 36); // 繪製單一個方塊
        glBindVertexArray(0); // 解綁

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteFramebuffers(1, &depthMapFBO);
    glDeleteTextures(1, &depthMapTexture);
    glDeleteVertexArrays(1, &groundVAO);
    glDeleteBuffers(1, &groundVBO);

    glfwTerminate();
    return 0;
}