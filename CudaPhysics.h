#pragma once

#include <vector>
#include <glm/glm.hpp>

extern const int CUDA_NUM_INSTANCES;

// 初始化 GPU 記憶體
void CudaPhysics_Init();

// 在 GPU 上執行物理模擬 (目前先用動畫代替)
void CudaPhysics_Update(float time, float deltaTime);

// 從 GPU 取回計算好的模型矩陣
void CudaPhysics_GetModelMatrices(std::vector<glm::mat4>& h_modelMatrices);

// 釋放 GPU 記憶體
void CudaPhysics_Cleanup();