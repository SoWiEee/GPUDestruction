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
 * @brief GPU 上的並行函式。每個執行緒更新一個方塊。
 * @param modelMatrices 指向 VBO 記憶體的 GPU 指標 (glm::mat4 陣列)
 * @param time 目前的動畫時間
 */
__global__ void static update_kernel(glm::mat4* modelMatrices, float time) {

    // thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_INSTANCES) {
        return;
    }

    // 讓所有方塊在一個大圓盤上旋轉
    float radius = 5.0f + (float)i / NUM_INSTANCES * 10.0f; // 半徑
    float angle = time * 0.5f + (float)i * 0.01f;     // 角度

    float x = radius * cos(angle);
    float y = 0.0f;
    float z = radius * sin(angle);

    glm::vec3 position = glm::vec3(x, y, z);

    // 建立 Model 矩陣
    glm::mat4 model = glm::mat4(1.0f);
    // 1. 平移
    model = glm::translate(model, position);
    // 2. 旋轉 (讓方塊自己也轉)
    model = glm::rotate(model, time * 2.0f + (float)i, glm::vec3(0.0f, 1.0f, 0.0f));
    // 3. 縮放 (讓方塊小一點)
    model = glm::scale(model, glm::vec3(0.1f));

    // 將計算好的矩陣寫回 VBO
    modelMatrices[i] = model;
}

// -------------------------------------------------
// GPU 核心 (Kernel) - [!! 極度簡化版 !!]
// -------------------------------------------------
// 讓所有 10,000 個方塊都靜止地出現在 (0, 0, 5) 的位置
__global__ void update_kernel_simple(glm::mat4* modelMatrices) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_INSTANCES) {
        return;
    }

    // [!! SIMPLIFIED !!]
    // 建立一個靜止的 Model 矩陣
    // 就像我們在「剝離測試」中做的一樣
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 5.0f)); // 放在 (0,0,5)
    model = glm::scale(model, glm::vec3(0.1f)); // 縮小一點 (因為有 10000 個)

    modelMatrices[i] = model;
}

// C++ interface

void cuda_init(GLuint vbo_id) {
    // 註冊 OpenGL VBO 到 CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&vbo_cuda_resource, vbo_id, cudaGraphicsRegisterFlagsWriteDiscard);

    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void cuda_run(float time) {
    // 1. 映射 (Map) VBO
    cudaCheckError(cudaGraphicsMapResources(1, &vbo_cuda_resource, 0));

    // 2. 取得設備指標
    glm::mat4* device_ptr;
    size_t num_bytes;
    cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &num_bytes, vbo_cuda_resource));

    // 3. 啟動 (Launch) Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_INSTANCES + threadsPerBlock - 1) / threadsPerBlock;

    // [!! UPDATED !!] 呼叫簡化版的 Kernel
    update_kernel_simple<<<blocksPerGrid, threadsPerBlock>>>(device_ptr);

    // [!! UPDATED !!] 
    // 檢查 Kernel 啟動是否有錯誤
    // cudaGetLastError() 會回報 kernel 執行期間的錯誤
    cudaCheckError(cudaGetLastError());

    // 4. 取消映射 (Unmap) VBO
    cudaCheckError(cudaGraphicsUnmapResources(1, &vbo_cuda_resource, 0));
}

void cuda_cleanup() {
    cudaCheckError(cudaGraphicsUnregisterResource(vbo_cuda_resource));
}