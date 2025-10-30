#include "kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

extern const int NUM_INSTANCES = 10000;
struct cudaGraphicsResource* vbo_cuda_resource;

// Kernel)

/**
 * @brief GPU �W���æ�禡�C�C�Ӱ������s�@�Ӥ���C
 * @param modelMatrices ���V VBO �O���骺 GPU ���� (glm::mat4 �}�C)
 * @param time �ثe���ʵe�ɶ�
 */
__global__ void static update_kernel(glm::mat4* modelMatrices, float time) {

    // thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_INSTANCES) {
        return;
    }

    // ���Ҧ�����b�@�Ӥj��L�W����
    float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f; // �b�|
    float angle = time * 0.5f + (float)i * 0.01f;     // ����

    float x = radius * cos(angle);
    float y = 0.0f;
    float z = radius * sin(angle);

    glm::vec3 position = glm::vec3(x, y, z);

    // �إ� Model �x�}
    glm::mat4 model = glm::mat4(1.0f);
    // 1. ����
    model = glm::translate(model, position);
    // 2. ���� (������ۤv�]��)
    model = glm::rotate(model, time * 2.0f + (float)i, glm::vec3(0.0f, 1.0f, 0.0f));
    // 3. �Y�� (������p�@�I)
    model = glm::scale(model, glm::vec3(0.1f));

    // �N�p��n���x�}�g�^ VBO
    modelMatrices[i] = model;
}

// -------------------------------------------------
// GPU �֤� (Kernel) - [!! ����²�ƪ� !!]
// -------------------------------------------------
// ���Ҧ� 10,000 �Ӥ�����R��a�X�{�b (0, 0, 5) ����m
__global__ void update_kernel_simple(glm::mat4* modelMatrices) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_INSTANCES) {
        return;
    }

    // [!! SIMPLIFIED !!]
    // �إߤ@���R� Model �x�}
    // �N���ڭ̦b�u�������աv�������@��
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 5.0f)); // ��b (0,0,5)
    model = glm::scale(model, glm::vec3(0.1f)); // �Y�p�@�I (�]���� 10000 ��)

    modelMatrices[i] = model;
}

// C++ interface

void cuda_init(GLuint vbo_id) {
    // ���U OpenGL VBO �� CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&vbo_cuda_resource, vbo_id, cudaGraphicsRegisterFlagsWriteDiscard);

    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void cuda_run(float time) {
    // 1. �M�g (Map) VBO
    cudaCheckError(cudaGraphicsMapResources(1, &vbo_cuda_resource, 0));

    // 2. ���o�]�ƫ���
    glm::mat4* device_ptr;
    size_t num_bytes;
    cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &num_bytes, vbo_cuda_resource));

    // 3. �Ұ� (Launch) Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_INSTANCES + threadsPerBlock - 1) / threadsPerBlock;

    // [!! UPDATED !!] �I�s²�ƪ��� Kernel
    update_kernel_simple<<<blocksPerGrid, threadsPerBlock>>>(device_ptr);

    // [!! UPDATED !!] 
    // �ˬd Kernel �ҰʬO�_�����~
    // cudaGetLastError() �|�^�� kernel ������������~
    cudaCheckError(cudaGetLastError());

    // 4. �����M�g (Unmap) VBO
    cudaCheckError(cudaGraphicsUnmapResources(1, &vbo_cuda_resource, 0));
}

void cuda_cleanup() {
    cudaCheckError(cudaGraphicsUnregisterResource(vbo_cuda_resource));
}