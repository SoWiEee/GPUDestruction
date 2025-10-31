#pragma once

#include "RigidBody.h"
#include <vector>
#include <map>
#include <list>

extern const int NUM_INSTANCES;

// �Ŷ����檺 Key
typedef long long GridKey;

class PhysicsEngine {
public:
    // �x�s�Ҧ����z����
    std::vector<RigidBody> bodies;

    // �x�s�p�⧹�����ҫ��x�}�A�ǳƵ� OpenGL
    std::vector<glm::mat4> modelMatrices;

public:
    // �غc�l�G��l�ƩҦ�����
    PhysicsEngine();

    // ���z�������D��s�禡
    void update(float time, float deltaTime);

    // ���o��V�һݪ��x�}
    const std::vector<glm::mat4>& getModelMatrices() const {
        return modelMatrices;
    }

private:
    // --- �Ŷ����� (Broad Phase) ---
    std::map<GridKey, std::list<int>> spatialGrid;
    float cellSize;

    // �N 3D �y���ഫ�� 64-bit key
    GridKey getGridKey(const glm::vec3& pos);

    // --- ���z�B�J ---
    void applyForces();
    void broadPhaseCollision();
    void narrowPhaseCollision(); // (���ӹ�@)
    void integrate(float deltaTime); // (���ӹ�@)
    void updateModelMatrices();
};