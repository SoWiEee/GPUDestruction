#version 450 core

layout (location = 0) in vec3 aPos;
layout (location = 2) in mat4 instanceModel; // 實例 Model 矩陣

uniform mat4 model; // 單一物體 Model 矩陣

uniform mat4 lightSpaceMatrix;

void main()
{
    mat4 finalModel;
    if (gl_InstanceID == 0) {
        finalModel = model;
    } else {
        finalModel = instanceModel;
    }
    gl_Position = lightSpaceMatrix * finalModel * vec4(aPos, 1.0);
}