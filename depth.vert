#version 450 core

layout (location = 0) in vec3 aPos;
layout (location = 2) in mat4 model;

// 光源的視角
uniform mat4 lightSpaceMatrix;

void main() {
    // 計算在光源視角下的位置
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}