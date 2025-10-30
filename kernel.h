#pragma once
#include <iostream> // ���F std::cerr �M std::cin
#include <cuda_runtime.h> // ���F cudaError_t

// [!! NEW !!] �P�R���~�ˬd��
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) {
            std::cerr << "Press ENTER to exit..." << std::endl;
            std::cin.get(); // [!] �Ȱ��{��
            exit(code);
        }
    }
}

#include <glad/glad.h>
extern const int NUM_INSTANCES;

// --- C++ �I�s CUDA ������ ---

/**
 * @brief ��l�� CUDA�A�õ��U OpenGL VBO�C
 * @param vbo_id �n�� CUDA �g�J�� VBO �� OpenGL ID�C
 */
void cuda_init(GLuint vbo_id);

/**
 * @brief ���� CUDA �֤ߡA��s�Ҧ���Ҫ� Model �x�}�C
 * @param time �ثe���ʵe�ɶ� (�Ҧp glfwGetTime())�C
 */
void cuda_run(float time);

/**
 * @brief �M�z CUDA �귽�C
 */
void cuda_cleanup();