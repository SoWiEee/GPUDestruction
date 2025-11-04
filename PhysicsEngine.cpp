#include "PhysicsEngine.h"
#include <iostream>

const int NUM_INSTANCES = 10000;

PhysicsEngine::PhysicsEngine() : cellSize(1.0f) {
    // 調整 vector 大小以容納所有物體
    bodies.resize(NUM_INSTANCES);
    modelMatrices.resize(NUM_INSTANCES);

    // 初始化一個 "盒子" 裡的物體，讓它們落下
    int size = 22; // 22*22*22 = 10648, 接近 10000
    float spacing = 1.0f; // 物體間距
    float startX = -size * spacing / 2.0f;
    float startY = 10.0f; // 從 10 公尺高的地方開始
    float startZ = -size * spacing / 2.0f;

    int n = 0;
    for (int y = 0; y < size; ++y) {
        for (int z = 0; z < size; ++z) {
            for (int x = 0; x < size; ++x) {
                if (n >= NUM_INSTANCES) break;

                bodies[n].position = glm::vec3(
                    startX + x * spacing,
                    startY + y * spacing,
                    startZ + z * spacing
                );
                // 增加一點隨機性
                bodies[n].position += glm::vec3(rand() % 10 / 50.0f, 0, 0);

                n++;
            }
            if (n >= NUM_INSTANCES) break;
        }
        if (n >= NUM_INSTANCES) break;
    }
}

// main update
void PhysicsEngine::update(float time, float deltaTime) {
    const float dt = 1.0f / 60.0f; // 60Hz

    clearForces();      // 清除上一幀的力
    applyForces();      // 套用這一幀的力
    handleCollisions(); // 碰撞偵測與反應
    integrate(dt);
    updateModelMatrices();
}

void PhysicsEngine::clearForces() {
    for (auto& body : bodies) {
        body.force = glm::vec3(0.0f);
        body.impulse = glm::vec3(0.0f);
    }
}

void PhysicsEngine::applyForces() {
    glm::vec3 gravity(0.0f, -9.8f, 0.0f);
    for (auto& body : bodies) {
        if (body.isStatic) continue;

        // F = m * g
        body.force += gravity * body.mass;
    }
}

void PhysicsEngine::handleCollisions() {
    // 與地板碰撞
    float groundY = -2.0f; // 地板高度
    float restitution = 0.1f; // 彈性 (0.1 = 幾乎不彈)

    for (auto& body : bodies) {
        if (body.isStatic) continue;

        // 簡單的 AABB 碰撞
        float bodyMinY = body.position.y - 0.5f; // 假設方塊大小為 1

        if (bodyMinY < groundY) {
            body.position.y = groundY + 0.5f;   // 防止穿透
            float velocityY = body.linearVelocity.y;    // 衝量反應
            if (velocityY < 0) {
                // F = m * (v_final - v_initial) / dt
                // 簡化為：施加一個反向的力來抵銷速度
                body.linearVelocity.y = -velocityY * restitution;
            }
        }
    }

    // Broad Phase
    spatialGrid.clear();
    for (int i = 0; i < bodies.size(); ++i) {
        GridKey key = getGridKey(bodies[i].position);
        spatialGrid[key].push_back(i);
    }

    // Narrow Phase
    for (auto const& [key, cell] : spatialGrid) {
        if (cell.size() < 2) continue;

        // 檢查格子內的所有物體對 (N^2)
        for (auto it1 = cell.begin(); it1 != cell.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != cell.end(); ++it2) {
                int id1 = *it1;
                int id2 = *it2;

                RigidBody& b1 = bodies[id1];
                RigidBody& b2 = bodies[id2];

                // --- 簡易 AABB 測試 ---
                // (同樣，先不考慮旋轉)
                glm::vec3 p1 = b1.position;
                glm::vec3 p2 = b2.position;
                float size = 1.0f; // 假設方塊大小為 1

                if (abs(p1.x - p2.x) < size &&
                    abs(p1.y - p2.y) < size &&
                    abs(p1.z - p2.z) < size)
                {
                    // 碰撞了！
                    // [!] 實作一個非常簡化的「分離」
                    // glm::vec3 diff = p2.position - p1.position;
                    glm::vec3 diff = b2.position - b1.position;
                    if (glm::length(diff) < 0.001f) diff = glm::vec3(0, 1, 0);

                    glm::vec3 normal = glm::normalize(diff);
                    float overlap = size - glm::length(diff);

                    if (overlap > 0) {
                        // 1. 位置校正
                        b1.position -= normal * overlap * 0.5f;
                        b2.position += normal * overlap * 0.5f;

                        // 2. 速度反應 (非常簡化，只處理 y 軸)
                        float avgVelY = (b1.linearVelocity.y + b2.linearVelocity.y) * 0.5f;
                        b1.linearVelocity.y = avgVelY * (1.0f - restitution);
                        b2.linearVelocity.y = avgVelY * (1.0f - restitution);
                    }
                }
            }
        }
    }
}

// 4. 積分 (半隱式歐拉法)
void PhysicsEngine::integrate(float deltaTime) {
    for (auto& body : bodies) {
        if (body.isStatic) continue;

        // a = F / m
        glm::vec3 acceleration = body.force * body.inverseMass;

        // v_new = v_old + a * dt
        body.linearVelocity += acceleration * deltaTime;

        // p_new = p_old + v_new * dt
        body.position += body.linearVelocity * deltaTime;
    }
}

// 更新渲染矩陣
void PhysicsEngine::updateModelMatrices() {
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        RigidBody& body = bodies[i];

        // 1. 從四元數轉換為 4x4 旋轉矩陣
        glm::mat4 rotMat = glm::mat4_cast(body.rotation);
        // 2. 建立平移矩陣
        glm::mat4 transMat = glm::translate(glm::mat4(1.0f), body.position);

        // [!] 縮放 (Scale) 是渲染屬性，不是物理屬性
        // 我們假設所有方塊的物理 AABB 是 1x1x1
        // 但渲染時，我們可以把它們縮小
        glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.9f)); // [!] 縮小到 0.9，讓方塊間有縫隙

        modelMatrices[i] = transMat * rotMat * scaleMat;
    }
}

// 空間網格 Key 的計算
GridKey PhysicsEngine::getGridKey(const glm::vec3& pos) {
    int x = static_cast<int>(floor(pos.x / cellSize));
    int y = static_cast<int>(floor(pos.y / cellSize));
    int z = static_cast<int>(floor(pos.z / cellSize));
    return (static_cast<GridKey>(x) << 42) | (static_cast<GridKey>(y & 0x1FFFFF) << 21) | static_cast<GridKey>(z & 0x1FFFFF);
}