#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

// vertex shader
const char* vertexShaderSource = R"glsl(
    #version 450 core
    
    // layout (location = 0)
    layout (location = 0) in vec3 aPos;     // (x, y, z)
    
    // layout (location = 1)
    layout (location = 1) in vec3 aColor;   // (r, g, b)

    out vec3 ourColor;  // 把顏色資料傳遞給片段著色器

    void main() {
        // gl_Position 是 GLSL 的內建變數，代表頂點的最終位置
        gl_Position = vec4(aPos, 1.0);
        ourColor = aColor; // 將顏色傳遞下去
    }
)glsl";

// fragment shader
const char* fragmentShaderSource = R"glsl(
    #version 450 core

    in vec3 ourColor;   // 從頂點著色器接收 (已插值) 的顏色
    out vec4 FragColor; // 最終輸出到螢幕的顏色

    void main() {
        FragColor = vec4(ourColor, 1.0); // 輸出顏色
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

int main(void) {
    // GLFW init
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "GPUDestruction Project - Hello Triangle!", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // GLAD init
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // compile shaders

    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check compiler error
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check compiler error
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // link shader program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check linker error
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // vertex and buffer

    // vertex data (x, y, z, r, g, b)
    float vertices[] = {
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // 右下 (紅)
        -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // 左下 (綠)
         0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // 頂部 (藍)
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO); // create VAO
    glGenBuffers(1, &VBO);      // create VBO

    // VAO setting
    // Vertex Array Object
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // 將頂點資料從 CPU 複製到 GPU 上的 VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Vertex Attribute Pointers

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); // 啟用屬性 0
    
    // color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1); // 啟用屬性 1

    // unbind VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Render Loop
    while (!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        // render command
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw triangle
        glUseProgram(shaderProgram);      // shader
        glBindVertexArray(VAO);           // bind VAO, VBO
        glDrawArrays(GL_TRIANGLES, 0, 3); // draw vertex
        // glBindVertexArray(0);          // unbind (optional)

        // swap buffer, event handle
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clear resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}