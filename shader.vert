#version 450 core

// �Ӧ� VBO
layout (location = 0) in vec3 aPos;     // position
layout (location = 1) in vec3 aColor;   // color

// �Ӧ� instanceVBO
layout (location = 2) in mat4 model;

// �Ӧ� C++ �� uniform
uniform mat4 view;
uniform mat4 projection;

out vec3 ourColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor; // �N�C��ǻ��U�h
}