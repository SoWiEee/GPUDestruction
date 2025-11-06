#define GLM_FORCE_MESSAGES
#define GLM_FORCE_CXX17
#define GLM_FORCE_SIMD_AVX2
#include "CudaPhysics.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) { std::cerr << "Press ENTER to exit..." << std::endl; std::cin.get(); exit(code); }
    }
}

using Mat4 = glm::mat4;
using Vec3 = glm::vec3;

const int CUDA_NUM_INSTANCES = 10000;

// GPU 記憶體指標
Mat4* d_modelMatrices = nullptr; // Kernel output
Vec3* d_positions = nullptr;     // 位置
Vec3* d_velocities = nullptr;    // 速度

__device__ void setTranslation(Mat4* mat, const Vec3& pos) {
    mat->operator[](0)[0] = 1.0f; mat->operator[](1)[0] = 0.0f; mat->operator[](2)[0] = 0.0f; mat->operator[](3)[0] = pos.x;
    mat->operator[](0)[1] = 0.0f; mat->operator[](1)[1] = 1.0f; mat->operator[](2)[1] = 0.0f; mat->operator[](3)[1] = pos.y;
    mat->operator[](0)[2] = 0.0f; mat->operator[](1)[2] = 0.0f; mat->operator[](2)[2] = 1.0f; mat->operator[](3)[2] = pos.z;
    mat->operator[](0)[3] = 0.0f; mat->operator[](1)[3] = 0.0f; mat->operator[](2)[3] = 0.0f; mat->operator[](3)[3] = 1.0f;
}

__device__ void setScale(Mat4* mat, const Vec3& scale) {
    mat->operator[](0)[0] = scale.x; mat->operator[](1)[0] = 0.0f;    mat->operator[](2)[0] = 0.0f;    mat->operator[](3)[0] = 0.0f;
    mat->operator[](0)[1] = 0.0f;    mat->operator[](1)[1] = scale.y; mat->operator[](2)[1] = 0.0f;    mat->operator[](3)[1] = 0.0f;
    mat->operator[](0)[2] = 0.0f;    mat->operator[](1)[2] = 0.0f;    mat->operator[](2)[2] = scale.z; mat->operator[](3)[2] = 0.0f;
    mat->operator[](0)[3] = 0.0f;    mat->operator[](1)[3] = 0.0f;    mat->operator[](2)[3] = 0.0f;    mat->operator[](3)[3] = 1.0f;
}

// CUDA Kernel
__global__ void physics_kernel(
    Mat4* modelMatrices, // output
    Vec3* positions,     // state
    Vec3* velocities,    // state
    float deltaTime
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CUDA_NUM_INSTANCES) {
        return;
    }

    // 物理常數
    const float mass = 1.0f;
    const Vec3 gravity = Vec3(0.0f, -9.8f, 0.0f);
    const float groundY = -2.0f;
    const float restitution = 0.3f; // 彈性

    // --- 1. 讀取目前狀態 ---
    Vec3 pos = positions[i];
    Vec3 vel = velocities[i];

    // --- 2. 套用外力 (重力) ---
    Vec3 force = gravity * mass;

    // --- 3. 積分 (半隱式歐拉法) ---
    // v_new = v_old + (F/m) * dt
    vel += (force / mass) * deltaTime;
    // p_new = p_old + v_new * dt
    pos += vel * deltaTime;

    // --- 4. 碰撞偵測 (與地板) ---
    float bodyMinY = pos.y - 0.5f; // 假設方塊大小為 1x1x1 (中心點在 pos)
    if (bodyMinY < groundY) {
        // 4a. 位置校正 (防止穿透)
        pos.y = groundY + 0.5f;

        // 4b. 速度反應 (反彈)
        if (vel.y < 0.0f) {
            vel.y = -vel.y * restitution;
        }
    }

    // --- 5. 寫回狀態 (供下一幀使用) ---
    positions[i] = pos;
    velocities[i] = vel;

    // --- 6. 產生「輸出」的模型矩陣 ---
    Mat4 transMat, scaleMat;
    setTranslation(&transMat, pos);
    setScale(&scaleMat, Vec3(0.9f)); // 縮小到 0.9f 讓方塊間有縫隙

    // [!] 我們的輸出是 (T * S)
    // (我們手動寫 T * S 的矩陣乘法)
    modelMatrices[i] = transMat; // 先複製平移
    // 再乘上縮放 (只需修改對角線)
    modelMatrices[i][0][0] *= 0.9f;
    modelMatrices[i][1][1] *= 0.9f;
    modelMatrices[i][2][2] *= 0.9f;
}


// C++ Interface

void CudaPhysics_Init() {
    // allocate output
    cudaCheckError(cudaMalloc((void**)&d_modelMatrices, CUDA_NUM_INSTANCES * sizeof(Mat4)));

    // allocate state
    cudaCheckError(cudaMalloc((void**)&d_positions, CUDA_NUM_INSTANCES * sizeof(Vec3)));
    cudaCheckError(cudaMalloc((void**)&d_velocities, CUDA_NUM_INSTANCES * sizeof(Vec3)));

    // state init-
    std::vector<Vec3> h_positions(CUDA_NUM_INSTANCES);
    std::vector<Vec3> h_velocities(CUDA_NUM_INSTANCES, Vec3(0.0f)); // v = 0

    int size = 22; // 22*22*22 = 10648
    float spacing = 1.0f;
    float startX = -size * spacing / 2.0f;
    float startY = 10.0f; // 從 10 公尺高開始
    float startZ = -size * spacing / 2.0f;

    int n = 0;
    for (int y = 0; y < size; ++y) {
        for (int z = 0; z < size; ++z) {
            for (int x = 0; x < size; ++x) {
                if (n >= CUDA_NUM_INSTANCES) break;
                h_positions[n] = Vec3(
                    startX + x * spacing,
                    startY + y * spacing,
                    startZ + z * spacing
                );
                n++;
            }
            if (n >= CUDA_NUM_INSTANCES) break;
        }
        if (n >= CUDA_NUM_INSTANCES) break;
    }

    // 4. 將初始狀態從 CPU (h_...) 複製到 GPU (d_...)
    cudaCheckError(cudaMemcpy(d_positions, h_positions.data(), CUDA_NUM_INSTANCES * sizeof(Vec3), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_velocities, h_velocities.data(), CUDA_NUM_INSTANCES * sizeof(Vec3), cudaMemcpyHostToDevice));

    // (h_positions 和 h_velocities 會在這裡自動被釋放，非常方便)
}

void CudaPhysics_Update(float time, float deltaTime) {
    const float dt = 1.0f / 60.0f;

    int threadsPerBlock = 256;
    int blocksPerGrid = (CUDA_NUM_INSTANCES + threadsPerBlock - 1) / threadsPerBlock;

    physics_kernel << <blocksPerGrid, threadsPerBlock >> > (
        (Mat4*)d_modelMatrices,
        (Vec3*)d_positions,
        (Vec3*)d_velocities,
        dt
        );

    cudaCheckError(cudaGetLastError());
}

void CudaPhysics_GetModelMatrices(std::vector<glm::mat4>& h_modelMatrices) {
    cudaCheckError(cudaMemcpy(
        h_modelMatrices.data(),
        d_modelMatrices,
        CUDA_NUM_INSTANCES * sizeof(glm::mat4),
        cudaMemcpyDeviceToHost
    ));
}

void CudaPhysics_Cleanup() {
    cudaCheckError(cudaFree(d_modelMatrices));
    cudaCheckError(cudaFree(d_positions));
    cudaCheckError(cudaFree(d_velocities));
}