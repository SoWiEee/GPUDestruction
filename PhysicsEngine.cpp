#include "PhysicsEngine.h"

const int NUM_INSTANCES = 10000;

PhysicsEngine::PhysicsEngine() : cellSize(2.0f) {
    // �վ� vector �j�p�H�e�ǩҦ�����
    bodies.resize(NUM_INSTANCES);
    modelMatrices.resize(NUM_INSTANCES);

    // [!] ��l�ơG�ڭ̧�ʵe�޿貾��o�̧@�� "��l���A"
    // (���ӳo�|�Q "���J���d" �� "�}�a" �޿���N)
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f;
        float angle = (float)i * 0.01f; // ��l����
        bodies[i].position = glm::vec3(radius * cos(angle), 0.0f, radius * sin(angle));

        // (�i�H�]�w�@�Ǫ�l�t��)
        // bodies[i].linearVelocity = glm::vec3(0.0f, 1.0f, 0.0f); 
    }
}

// ���z�������D��s�禡
void PhysicsEngine::update(float time, float deltaTime) {
    // [!] ���F�O���ʵe�ĪG�A�ڭ̥��Ȯɦb�o�̧�s
    // (���ӡA�o�� "�ʵe" �޿�|�Q "�n��" �M "�I������" ���N)
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f;
        float angle = time * 0.5f + (float)i * 0.01f;
        bodies[i].position = glm::vec3(radius * cos(angle), 0.0f, radius * sin(angle));
        bodies[i].rotation = glm::angleAxis(time * 2.0f + (float)i, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    // --- �u�������z�B�J ---
    // (�ثe�O�Ū��A���[�c�w�g�ǳƦn�F)
    applyForces();
    broadPhaseCollision();
    narrowPhaseCollision();
    integrate(deltaTime);

    // (�o�@�B�O������)
    updateModelMatrices();
}

void PhysicsEngine::applyForces() {
    // glm::vec3 gravity(0.0f, -9.8f, 0.0f);
    // for (auto& body : bodies) {
    //    body.linearVelocity += gravity * deltaTime; // (�ݭn�ǤJ deltaTime)
    // }
}

void PhysicsEngine::broadPhaseCollision() {
    // 1. �M������
    spatialGrid.clear();

    // 2. ��J����
    for (int i = 0; i < bodies.size(); ++i) {
        GridKey key = getGridKey(bodies[i].position);
        spatialGrid[key].push_back(i);
    }

    // 3. ��X��b�I����
    // (���ӹ�@...)
}

void PhysicsEngine::narrowPhaseCollision() {
    // (���ӹ�@...)
}

void PhysicsEngine::integrate(float deltaTime) {
    // (���ӹ�@... �Ҧp�G)
    // for (auto& body : bodies) {
    //    body.position += body.linearVelocity * deltaTime;
    //    body.rotation = ... (��������|���ƿn��)
    // }
}

void PhysicsEngine::updateModelMatrices() {
    for (int i = 0; i < NUM_INSTANCES; ++i) {
        bodies[i].updateModelMatrix();

        // �M�ε�ı�W���Y�� (���v�T���z)
        modelMatrices[i] = glm::scale(bodies[i].modelMatrix, glm::vec3(0.1f));
    }
}

// �Ŷ����� Key ���p��
GridKey PhysicsEngine::getGridKey(const glm::vec3& pos) {
    int x = static_cast<int>(floor(pos.x / cellSize));
    int y = static_cast<int>(floor(pos.y / cellSize));
    int z = static_cast<int>(floor(pos.z / cellSize));

    return (static_cast<GridKey>(x) << 42) |
        (static_cast<GridKey>(y & 0x1FFFFF) << 21) |
        static_cast<GridKey>(z & 0x1FFFFF);
}