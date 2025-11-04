#pragma once

#include "RigidBody.h"
#include <vector>
#include <map>
#include <list>

extern const int NUM_INSTANCES;
typedef long long GridKey;

class PhysicsEngine {
public:
    std::vector<RigidBody> bodies;          // 儲存所有物理實體
    std::vector<glm::mat4> modelMatrices;   // 儲存計算完畢的模型矩陣，準備給 OpenGL

public:
    PhysicsEngine();

    // 物理引擎的主更新函式
    void update(float time, float deltaTime);

    // 取得渲染所需的矩陣
    const std::vector<glm::mat4>& getModelMatrices() const {
        return modelMatrices;
    }

private:
    // Broad Phase
    std::map<GridKey, std::list<int>> spatialGrid;
    float cellSize;
    GridKey getGridKey(const glm::vec3& pos);   // 將 3D 座標轉換為 64-bit key

    // --- 物理步驟 ---
    void clearForces();
    void applyForces();
    void handleCollisions();
    // 4. 積分 (更新速度和位置)
    void integrate(float deltaTime);
    void updateModelMatrices();
};