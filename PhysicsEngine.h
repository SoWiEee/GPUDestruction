#pragma once

#include "RigidBody.h"
#include <vector>
#include <map>
#include <list>

extern const int NUM_INSTANCES;

// 空間網格的 Key
typedef long long GridKey;

class PhysicsEngine {
public:
    // 儲存所有物理實體
    std::vector<RigidBody> bodies;

    // 儲存計算完畢的模型矩陣，準備給 OpenGL
    std::vector<glm::mat4> modelMatrices;

public:
    // 建構子：初始化所有物體
    PhysicsEngine();

    // 物理引擎的主更新函式
    void update(float time, float deltaTime);

    // 取得渲染所需的矩陣
    const std::vector<glm::mat4>& getModelMatrices() const {
        return modelMatrices;
    }

private:
    // --- 空間網格 (Broad Phase) ---
    std::map<GridKey, std::list<int>> spatialGrid;
    float cellSize;

    // 將 3D 座標轉換為 64-bit key
    GridKey getGridKey(const glm::vec3& pos);

    // --- 物理步驟 ---
    void applyForces();
    void broadPhaseCollision();
    void narrowPhaseCollision(); // (未來實作)
    void integrate(float deltaTime); // (未來實作)
    void updateModelMatrices();
};