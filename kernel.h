#pragma once
#include <iostream> // 為了 std::cerr 和 std::cin
#include <cuda_runtime.h> // 為了 cudaError_t

// [!! NEW !!] 致命錯誤檢查宏
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) {
            std::cerr << "Press ENTER to exit..." << std::endl;
            std::cin.get(); // [!] 暫停程式
            exit(code);
        }
    }
}

#include <glad/glad.h>
extern const int NUM_INSTANCES;

// --- C++ 呼叫 CUDA 的介面 ---

/**
 * @brief 初始化 CUDA，並註冊 OpenGL VBO。
 * @param vbo_id 要給 CUDA 寫入的 VBO 的 OpenGL ID。
 */
void cuda_init(GLuint vbo_id);

/**
 * @brief 執行 CUDA 核心，更新所有實例的 Model 矩陣。
 * @param time 目前的動畫時間 (例如 glfwGetTime())。
 */
void cuda_run(float time);

/**
 * @brief 清理 CUDA 資源。
 */
void cuda_cleanup();