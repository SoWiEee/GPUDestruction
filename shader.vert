#version 450 core

layout (location = 0) in vec3 aPos;     // position
layout (location = 1) in vec3 aColor;   // color
layout (location = 6) in vec2 aTexCoord; // 紋理座標
layout (location = 2) in mat4 model;   // 實例 Model 矩陣

// camera matrices
uniform mat4 view;
uniform mat4 projection;

uniform mat4 lightSpaceMatrix;  // 光源矩陣

out vec3 ourColor;
out vec2 TexCoord;
out vec3 FragPos; //  傳遞世界座標 (用於光照)
out vec4 FragPosLightSpace; // 在光源視角下的座標

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);

    gl_Position = projection * view * worldPos;

    FragPosLightSpace = lightSpaceMatrix * worldPos;

    ourColor = aColor;
    TexCoord = aTexCoord;
}