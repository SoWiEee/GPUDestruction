#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// vertex shader
const char* vertexShaderSource = R"glsl(
    #version 450 core
    
    layout (location = 0) in vec3 aPos;     // 頂點位置
    layout (location = 1) in vec3 aColor;   // 頂點顏色

    out vec3 ourColor;

    // receive MVP matrix
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        // 最終位置 = 投影 * 視圖 * 模型 * 原始位置
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor; // 將顏色傳遞下去
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

int main() {
    // GLFW init
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GPUDestruction Project - Step 4: 3D Cube!", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // z-buffer
    glEnable(GL_DEPTH_TEST);

    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // link shader
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // 定義方塊的 8 個頂點
    // 一個方塊有 6 個面，每個面 2 個三角形，共 12 個三角形
    // 12 * 3 = 36 個頂點。我們使用索引來重複使用 8 個頂點。
    float vertices[] = {
        // 後面
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f, // 0
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f, // 1
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f, // 2
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f, // 3
        // 前面
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f, // 4
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f, // 5
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f, // 6
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f  // 7
    };

    // 2. 定義 12 個三角形 (36 個索引)
    // 這些索引指向上面 'vertices' 陣列的索引 (0 到 7)
    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0, // 後面
        4, 5, 6,  6, 7, 4, // 前面
        7, 6, 2,  2, 3, 7, // 上面
        4, 5, 1,  1, 0, 4, // 下面
        4, 7, 3,  3, 0, 4, // 左面
        5, 6, 2,  2, 1, 5  // 右面
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // VAO settings
    glBindVertexArray(VAO);

    // VBO (儲存頂點資料)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // EBO (儲存索引資料)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Render Loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // render command
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 啟用 Shader
        glUseProgram(shaderProgram);


        // Model Matrix
        // 讓方塊隨時間旋轉
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, (float)glfwGetTime() * glm::radians(50.0f), glm::vec3(0.5f, 1.0f, 0.0f));

        // View Matrix
        // 在 (0, 0, 3) 的位置，看著原點 (0, 0, 0)
        glm::mat4 view = glm::mat4(1.0f);
        view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 3.0f), // 攝影機位置
            glm::vec3(0.0f, 0.0f, 0.0f), // 攝影機看向的目標
            glm::vec3(0.0f, 1.0f, 0.0f)  // 攝影機的 "上" 方向
        );

        // Projection Matrix
        // 設定 45 度的視野 (FOV)
        glm::mat4 projection = glm::mat4(1.0f);
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        // 將矩陣傳送到 Shader
        // 首先找到 uniform 變數在 shader 中的位置
        int modelLoc = glGetUniformLocation(shaderProgram, "model");
        int viewLoc = glGetUniformLocation(shaderProgram, "view");
        int projLoc = glGetUniformLocation(shaderProgram, "projection");

        // 然後上傳矩陣資料
        // glUniformMatrix4fv(location, 1, GL_FALSE, 矩陣資料的指標)
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));


        // 繪製方塊 (使用索引繪製)
        glBindVertexArray(VAO); // 綁定 "配方" (VAO 會自動綁定 EBO)

        // 使用 glDrawElements 來繪製
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

        // swap buffers, event handle
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clean resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}