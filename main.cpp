#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// vertex shader
const char* vertexShaderSource = R"glsl(
    #version 450 core
    
    layout (location = 0) in vec3 aPos;     // ���I��m
    layout (location = 1) in vec3 aColor;   // ���I�C��

    out vec3 ourColor;

    // receive MVP matrix
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        // �̲צ�m = ��v * ���� * �ҫ� * ��l��m
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor; // �N�C��ǻ��U�h
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

    // �w�q����� 8 �ӳ��I
    // �@�Ӥ���� 6 �ӭ��A�C�ӭ� 2 �ӤT���ΡA�@ 12 �ӤT����
    // 12 * 3 = 36 �ӳ��I�C�ڭ̨ϥί��ިӭ��ƨϥ� 8 �ӳ��I�C
    float vertices[] = {
        // �᭱
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f, 0.0f, // 0
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f, // 1
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f, // 2
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f, 0.0f, // 3
        // �e��
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f, // 4
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f, // 5
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f, // 6
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f, 1.0f  // 7
    };

    // 2. �w�q 12 �ӤT���� (36 �ӯ���)
    // �o�ǯ��ޫ��V�W�� 'vertices' �}�C������ (0 �� 7)
    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0, // �᭱
        4, 5, 6,  6, 7, 4, // �e��
        7, 6, 2,  2, 3, 7, // �W��
        4, 5, 1,  1, 0, 4, // �U��
        4, 7, 3,  3, 0, 4, // ����
        5, 6, 2,  2, 1, 5  // �k��
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // VAO settings
    glBindVertexArray(VAO);

    // VBO (�x�s���I���)
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // EBO (�x�s���޸��)
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

        // �ҥ� Shader
        glUseProgram(shaderProgram);


        // Model Matrix
        // ������H�ɶ�����
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, (float)glfwGetTime() * glm::radians(50.0f), glm::vec3(0.5f, 1.0f, 0.0f));

        // View Matrix
        // �b (0, 0, 3) ����m�A�ݵۭ��I (0, 0, 0)
        glm::mat4 view = glm::mat4(1.0f);
        view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 3.0f), // ��v����m
            glm::vec3(0.0f, 0.0f, 0.0f), // ��v���ݦV���ؼ�
            glm::vec3(0.0f, 1.0f, 0.0f)  // ��v���� "�W" ��V
        );

        // Projection Matrix
        // �]�w 45 �ת����� (FOV)
        glm::mat4 projection = glm::mat4(1.0f);
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        // �N�x�}�ǰe�� Shader
        // ������� uniform �ܼƦb shader ������m
        int modelLoc = glGetUniformLocation(shaderProgram, "model");
        int viewLoc = glGetUniformLocation(shaderProgram, "view");
        int projLoc = glGetUniformLocation(shaderProgram, "projection");

        // �M��W�ǯx�}���
        // glUniformMatrix4fv(location, 1, GL_FALSE, �x�}��ƪ�����)
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));


        // ø�s��� (�ϥί���ø�s)
        glBindVertexArray(VAO); // �j�w "�t��" (VAO �|�۰ʸj�w EBO)

        // �ϥ� glDrawElements ��ø�s
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