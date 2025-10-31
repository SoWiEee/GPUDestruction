#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ���骺���z���A
struct RigidBody {
    // --- �ʺA�ݩ� ---
    glm::vec3 position;
    glm::quat rotation; // �ϥΥ|���� (w, x, y, z)

    glm::vec3 linearVelocity;  // �u�t��
    glm::vec3 angularVelocity; // ���t��

    // --- �R�A�ݩ� ---
    float mass;

    // --- ��V�� ---
    glm::mat4 modelMatrix; // �C�V�p��@���A�� OpenGL ��

    // �غc�l
    RigidBody() :
        position(0.0f),
        rotation(1.0f, 0.0f, 0.0f, 0.0f), // (w, x, y, z)
        linearVelocity(0.0f),
        angularVelocity(0.0f),
        mass(1.0f),
        modelMatrix(1.0f)
    {
    }

    // �ھ� position �M rotation ��s modelMatrix
    void updateModelMatrix() {
        // 1. �q�|�����ഫ�� 4x4 ����x�}
        glm::mat4 rotMat = glm::mat4_cast(rotation);
        // 2. �إߥ����x�}
        glm::mat4 transMat = glm::translate(glm::mat4(1.0f), position);
        // 3. ���X (���� * ����)
        // [!] �`�N�G�Y�� (scale) ���Ӧb��V�ɤ~�M��
        // �]�����|�v�T���z�I������ (AABB)
        modelMatrix = transMat * rotMat;
    }
};