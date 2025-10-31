#version 450 core

// 來自 VBO
layout (location = 0) in vec3 aPos;     // position
layout (location = 1) in vec3 aColor;   // color

// 來自 instanceVBO
layout (location = 2) in mat4 model;

// 來自 C++ 的 uniform
uniform mat4 view;
uniform mat4 projection;

out vec3 ourColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor; // 將顏色傳遞下去
}