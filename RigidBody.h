#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

// 剛體的物理狀態
struct RigidBody {
    // --- 動態屬性 ---
    glm::vec3 position;
    glm::quat rotation; // 使用四元數 (w, x, y, z)

    glm::vec3 linearVelocity;  // 線速度
    glm::vec3 angularVelocity; // 角速度

    // --- 靜態屬性 ---
    float mass;

    // --- 渲染用 ---
    glm::mat4 modelMatrix; // 每幀計算一次，給 OpenGL 用

    // 建構子
    RigidBody() :
        position(0.0f),
        rotation(1.0f, 0.0f, 0.0f, 0.0f), // (w, x, y, z)
        linearVelocity(0.0f),
        angularVelocity(0.0f),
        mass(1.0f),
        modelMatrix(1.0f)
    {
    }

    // 根據 position 和 rotation 更新 modelMatrix
    void updateModelMatrix() {
        // 1. 從四元數轉換為 4x4 旋轉矩陣
        glm::mat4 rotMat = glm::mat4_cast(rotation);
        // 2. 建立平移矩陣
        glm::mat4 transMat = glm::translate(glm::mat4(1.0f), position);
        // 3. 結合 (平移 * 旋轉)
        // [!] 注意：縮放 (scale) 應該在渲染時才套用
        // 因為它會影響物理碰撞偵測 (AABB)
        modelMatrix = transMat * rotMat;
    }
};