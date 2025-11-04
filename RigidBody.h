#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

// 軸對齊包圍盒 (Axis-Aligned Bounding Box)
struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB(glm::vec3 _min = glm::vec3(-0.5f), glm::vec3 _max = glm::vec3(0.5f))
        : min(_min), max(_max) {
    }
};

// 剛體的物理狀態
struct RigidBody {
    // --- 動態屬性 ---
    glm::vec3 position;
    glm::quat rotation; // 使用四元數 (w, x, y, z)
    glm::vec3 linearVelocity;  // 線速度
    glm::vec3 angularVelocity; // 角速度

    // 力與衝量的累加器
    glm::vec3 force;
    glm::vec3 impulse;

    // --- 靜態屬性 ---
    float mass;
    float inverseMass;

    // 碰撞體 (AABB)
    AABB localAABB; // 物件本地座標 AABB (例如 -0.5 到 0.5)

    // flag
    bool isStatic; // 是否為靜態物體 (例如地板)

    // 建構子
    RigidBody() :
        position(0.0f),
        rotation(1.0f, 0.0f, 0.0f, 0.0f),
        linearVelocity(0.0f),
        angularVelocity(0.0f),
        force(0.0f),
        impulse(0.0f),
        mass(1.0f),
        inverseMass(1.0f / 1.0f),
        localAABB(glm::vec3(-0.5f), glm::vec3(0.5f)), // 預設 1x1x1 方塊
        isStatic(false)
    {
        // 靜態物體有 "無限大" 的質量
        if (isStatic) {
            mass = 0.0f;
            inverseMass = 0.0f;
        }
    }
};