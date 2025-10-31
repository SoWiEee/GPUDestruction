#include "PhysicsEngine.h"

const int NUM_INSTANCES = 10000;

PhysicsEngine::PhysicsEngine() : cellSize(2.0f) {
    // 調整 vector 大小以容納所有物體
    bodies.resize(NUM_INSTANCES);
    modelMatrices.resize(NUM_INSTANCES);

    // [!] 初始化：我們把動畫邏輯移到這裡作為 "初始狀態"
    // (未來這會被 "載入關卡" 或 "破壞" 邏輯取代)
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f;
        float angle = (float)i * 0.01f; // 初始角度
        bodies[i].position = glm::vec3(radius * cos(angle), 0.0f, radius * sin(angle));

        // (可以設定一些初始速度)
        // bodies[i].linearVelocity = glm::vec3(0.0f, 1.0f, 0.0f); 
    }
}

// 物理引擎的主更新函式
void PhysicsEngine::update(float time, float deltaTime) {
    // [!] 為了保持動畫效果，我們先暫時在這裡更新
    // (未來，這個 "動畫" 邏輯會被 "積分" 和 "碰撞反應" 取代)
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f;
        float angle = time * 0.5f + (float)i * 0.01f;
        bodies[i].position = glm::vec3(radius * cos(angle), 0.0f, radius * sin(angle));
        bodies[i].rotation = glm::angleAxis(time * 2.0f + (float)i, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    // --- 真正的物理步驟 ---
    // (目前是空的，但架構已經準備好了)
    applyForces();
    broadPhaseCollision();
    narrowPhaseCollision();
    integrate(deltaTime);

    // (這一步是必須的)
    updateModelMatrices();
}

void PhysicsEngine::applyForces() {
    // glm::vec3 gravity(0.0f, -9.8f, 0.0f);
    // for (auto& body : bodies) {
    //    body.linearVelocity += gravity * deltaTime; // (需要傳入 deltaTime)
    // }
}

void PhysicsEngine::broadPhaseCollision() {
    // 1. 清除網格
    spatialGrid.clear();

    // 2. 填入網格
    for (int i = 0; i < bodies.size(); ++i) {
        GridKey key = getGridKey(bodies[i].position);
        spatialGrid[key].push_back(i);
    }

    // 3. 找出潛在碰撞對
    // (未來實作...)
}

void PhysicsEngine::narrowPhaseCollision() {
    // (未來實作...)
}

void PhysicsEngine::integrate(float deltaTime) {
    // (未來實作... 例如：)
    // for (auto& body : bodies) {
    //    body.position += body.linearVelocity * deltaTime;
    //    body.rotation = ... (更複雜的四元數積分)
    // }
}

void PhysicsEngine::updateModelMatrices() {
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        bodies[i].updateModelMatrix();

        // 套用視覺上的縮放 (不影響物理)
        modelMatrices[i] = glm::scale(bodies[i].modelMatrix, glm::vec3(0.1f));
    }
}

// 空間網格 Key 的計算
GridKey PhysicsEngine::getGridKey(const glm::vec3& pos) {
    int x = static_cast<int>(floor(pos.x / cellSize));
    int y = static_cast<int>(floor(pos.y / cellSize));
    int z = static_cast<int>(floor(pos.z / cellSize));

    return (static_cast<GridKey>(x) << 42) |
        (static_cast<GridKey>(y & 0x1FFFFF) << 21) |
        static_cast<GridKey>(z & 0x1FFFFF);
}