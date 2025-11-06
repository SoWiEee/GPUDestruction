#define GLM_FORCE_MESSAGES
#define GLM_FORCE_CXX17
#define GLM_FORCE_SIMD_AVX2
#include "CudaPhysics.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <vector>
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

// GPU ptr
Mat4* d_modelMatrices = nullptr; // kernel output
Vec3* d_positions = nullptr;     // 位置
Vec3* d_velocities = nullptr;    // 速度

//  雜湊網格 (Hashed Grid)
const int HASH_GRID_SIZE = 19997;
// d_grid_heads: 雜湊表，儲存每個網格「鏈結串列」的頭部 (物體 ID)
int* d_grid_heads = nullptr;
// d_grid_next: 儲存下一個物體的 ID (組成鏈結串列)
int* d_grid_next = nullptr;
// d_grid_keys: (偵錯用) 儲存每個物體計算出的網格 Key
int* d_grid_keys = nullptr;

// [!! UPDATED !!]
// 將 CELL_SIZE 宣告在 __constant__ (常數) 記憶體中
// 讓 GPU 上的所有 __device__ 和 __global__ 函式都能讀取它
__constant__ float CELL_SIZE = 1.0f;


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

__device__ Mat4 matrixMultiply(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[k][row] * b[col][k];
            }
            result[col][row] = sum;
        }
    }
    return result;
}

// CUDA Kernel
__global__ void physics_kernel(
    Mat4* modelMatrices, // 輸出
    Vec3* positions,     // 狀態 (讀/寫)
    Vec3* velocities,    // 狀態 (讀/寫)
    float deltaTime
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CUDA_NUM_INSTANCES) {
        return;
    }

    // --- 物理常數 ---
    const float mass = 1.0f;
    // [!! FIX !!] 手動初始化 Vec3，避免 C++ 建構子
    const Vec3 gravity = Vec3(0.0f, -9.8f, 0.0f);
    const float groundY = -2.0f;
    const float restitution = 0.3f;

    // --- 1. 讀取目前狀態 ---
    Vec3 pos = positions[i];
    Vec3 vel = velocities[i];

    // --- 2. 套用外力 (重力) ---
    // [!! FIX !!] 手動計算 F = m * g (逐分量)
    Vec3 force;
    force.x = gravity.x * mass;
    force.y = gravity.y * mass;
    force.z = gravity.z * mass;

    // --- 3. 積分 (半隱式歐拉法) ---
    // 手動計算 a = F / m
    Vec3 acceleration;
    acceleration.x = force.x / mass;
    acceleration.y = force.y / mass;
    acceleration.z = force.z / mass;

    //  手動計算 v_new = v_old + a * dt
    vel.x = vel.x + acceleration.x * deltaTime;
    vel.y = vel.y + acceleration.y * deltaTime;
    vel.z = vel.z + acceleration.z * deltaTime;

    // 手動計算 p_new = p_old + v_new * dt
    pos.x = pos.x + vel.x * deltaTime;
    pos.y = pos.y + vel.y * deltaTime;
    pos.z = pos.z + vel.z * deltaTime;

    // --- 4. 碰撞偵測 (與地板) ---
    float bodyMinY = pos.y - 0.5f;
    if (bodyMinY < groundY) {
        pos.y = groundY + 0.5f;
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
    setScale(&scaleMat, Vec3(0.9f));
    modelMatrices[i] = matrixMultiply(transMat, scaleMat);
}

__device__ int getGridKey(const Vec3& pos) {
    int x = (int)floorf(pos.x / CELL_SIZE);
    int y = (int)floorf(pos.y / CELL_SIZE);
    int z = (int)floorf(pos.z / CELL_SIZE);
    // (x*p1 ^ y*p2 ^ z*p3) % N
    // 73856093, 19349663, 83492791 是有名的 LCG 質數
    int key = (x * 73856093 ^ y * 19349663 ^ z * 83492791);
    // [!] 必須是正數
    return abs(key);
}

// 根據 3D 座標和網格索引取得網格 Key
__device__ int getGridKeyFromCoords(int x, int y, int z) {
    int key = (x * 73856093 ^ y * 19349663 ^ z * 83492791);
    return abs(key);
}

// Kernel 1: 建立雜湊網格
// 每個執行緒 (物體 i) 將自己插入到雜湊表的鏈結串列中
__global__ void build_grid_kernel(
    int* grid_heads,
    int* grid_next,
    int* grid_keys,
    const Vec3* positions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CUDA_NUM_INSTANCES) return;

    // 1. 計算我 (i) 的網格 Key
    int key = getGridKey(positions[i]);
    grid_keys[i] = key; // 儲存 (偵錯用)

    // 2. 計算雜湊表索引
    int hash = key % HASH_GRID_SIZE;

    // 3. [!! 原子操作 !!] 將我 (i) 插入到鏈結串列的頭部
    // atomicExch 會「原子地」執行以下兩步：
    // a) 取得 grid_heads[hash] 的「舊」值 (可能是 -1 或另一個物體 ID)
    // b) 將 grid_heads[hash] 的值「設定」為 i (我現在是新的頭)
    int old_head = atomicExch(&grid_heads[hash], i);

    // 4. 將「舊」的頭，連接到我的「下一個」
    grid_next[i] = old_head;
}


// Kernel 2: 物理計算與碰撞
__global__ void collide_kernel(
    Mat4* modelMatrices,
    Vec3* positions,
    Vec3* velocities,
    const int* grid_heads,
    const int* grid_next,
    const Vec3* all_positions,
    float deltaTime
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CUDA_NUM_INSTANCES) return;

    // --- 物理常數 ---
    const float mass = 1.0f;
    const Vec3 gravity = Vec3(0.0f, -9.8f, 0.0f);
    const float groundY = -2.0f;
    const float restitution = 0.3f; // 彈性
    const float size = 1.0f;        // 碰撞大小 (0.5 + 0.5)

    // --- 1. 讀取 & 積分 (不變) ---
    Vec3 pos = positions[i];
    Vec3 vel = velocities[i];
    Vec3 force; force.x = gravity.x * mass; force.y = gravity.y * mass; force.z = gravity.z * mass;
    Vec3 acceleration; acceleration.x = force.x / mass; acceleration.y = force.y / mass; acceleration.z = force.z / mass;
    vel.x += acceleration.x * deltaTime; vel.y += acceleration.y * deltaTime; vel.z += acceleration.z * deltaTime;
    pos.x += vel.x * deltaTime; pos.y += vel.y * deltaTime; pos.z += vel.z * deltaTime;

    // --- 2. 碰撞偵測 (地板) (不變) ---
    float bodyMinY = pos.y - 0.5f;
    if (bodyMinY < groundY) {
        pos.y = groundY + 0.5f;
        if (vel.y < 0.0f) vel.y = -vel.y * restitution;
    }

    // --- 3. 物體 vs 物體 碰撞 (Broad + Narrow Phase) ---
    int my_grid_x = (int)floorf(pos.x / CELL_SIZE);
    int my_grid_y = (int)floorf(pos.y / CELL_SIZE);
    int my_grid_z = (int)floorf(pos.z / CELL_SIZE);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_grid_x = my_grid_x + dx;
                int neighbor_grid_y = my_grid_y + dy;
                int neighbor_grid_z = my_grid_z + dz;
                int neighbor_key = getGridKeyFromCoords(neighbor_grid_x, neighbor_grid_y, neighbor_grid_z);
                int neighbor_hash = neighbor_key % HASH_GRID_SIZE;
                int neighbor_id = grid_heads[neighbor_hash];

                while (neighbor_id != -1) {
                    // [!] 修正：我們應該要能和 ID > i 的鄰居碰撞
                    // 但我們只「校正」自己的位置，並只在 neighbor_id < i 時處理
                    // 這樣可以避免資料競爭

                    if (neighbor_id == i) {
                        neighbor_id = grid_next[neighbor_id];
                        continue;
                    }

                    Vec3 neighbor_pos = all_positions[neighbor_id];
                    if (fabsf(pos.x - neighbor_pos.x) < size &&
                        fabsf(pos.y - neighbor_pos.y) < size &&
                        fabsf(pos.z - neighbor_pos.z) < size)
                    {
                        // 碰撞了！
                        Vec3 diff;
                        diff.x = pos.x - neighbor_pos.x;
                        diff.y = pos.y - neighbor_pos.y;
                        diff.z = pos.z - neighbor_pos.z;

                        float length = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                        if (length < 0.0001f) {
                            length = 0.0001f;
                            diff.x = 0.0f; diff.y = 0.0001f; diff.z = 0.0f; // 預設往上推
                        }

                        float overlap = size - length;
                        if (overlap > 0) {
                            // 1. 位置校正 (只推開「我」)
                            Vec3 normal;
                            normal.x = diff.x / length;
                            normal.y = diff.y / length;
                            normal.z = diff.z / length;

                            pos.x += normal.x * overlap;
                            pos.y += normal.y * overlap;
                            pos.z += normal.z * overlap;

                            // [!! NEW !!] 2. 3D 速度反射
                            // 這是模擬「散開」的關鍵
                            // v_new = v_old - (1 + e) * dot(v_old, n) * n

                            float velDotNormal = vel.x * normal.x + vel.y * normal.y + vel.z * normal.z;

                            // 只有當物體相向運動時才反彈
                            if (velDotNormal < 0) {
                                float bounceFactor = -(1.0f + restitution) * velDotNormal;

                                // [!] 將所有速度分量都反射出去
                                vel.x += bounceFactor * normal.x;
                                vel.y += bounceFactor * normal.y;
                                vel.z += bounceFactor * normal.z;
                            }
                        }
                    }
                    neighbor_id = grid_next[neighbor_id];
                }
            }
        }
    }

    // --- 4. 寫回狀態 (不變) ---
    positions[i] = pos;
    velocities[i] = vel;

    // --- 5. 產生「輸出」的模型矩陣 (不變) ---
    Mat4 transMat, scaleMat;
    setTranslation(&transMat, pos);
    setScale(&scaleMat, Vec3(0.9f));
    modelMatrices[i] = matrixMultiply(transMat, scaleMat);
}

__global__ void init_kernel(int* array, int size, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        array[i] = value;
    }
}

// C++ Interface

void CudaPhysics_Init() {
    // allocate output
    cudaCheckError(cudaMalloc((void**)&d_modelMatrices, CUDA_NUM_INSTANCES * sizeof(Mat4)));

    // allocate state
    cudaCheckError(cudaMalloc((void**)&d_positions, CUDA_NUM_INSTANCES * sizeof(Vec3)));
    cudaCheckError(cudaMalloc((void**)&d_velocities, CUDA_NUM_INSTANCES * sizeof(Vec3)));

    // 分配「網格」記憶體
    cudaCheckError(cudaMalloc((void**)&d_grid_heads, HASH_GRID_SIZE * sizeof(int)));
    cudaCheckError(cudaMalloc((void**)&d_grid_next, CUDA_NUM_INSTANCES * sizeof(int)));
    cudaCheckError(cudaMalloc((void**)&d_grid_keys, CUDA_NUM_INSTANCES * sizeof(int)));

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

    cudaCheckError(cudaMemcpy(d_positions, h_positions.data(), CUDA_NUM_INSTANCES * sizeof(Vec3), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_velocities, h_velocities.data(), CUDA_NUM_INSTANCES * sizeof(Vec3), cudaMemcpyHostToDevice));
}

void CudaPhysics_Update(float time, float deltaTime) {
    const float dt = 1.0f / 60.0f;

    int threadsPerBlock = 256;
    // 步驟 1: 清空/初始化網格 ---
    // (我們必須每幀都把網格頭部設為 -1)
    int blocks_grid = (HASH_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel << <blocks_grid, threadsPerBlock >> > (d_grid_heads, HASH_GRID_SIZE, -1);
    cudaCheckError(cudaGetLastError()); // 檢查

    // 步驟 2: 建立網格 ---
    int blocks_particles = (CUDA_NUM_INSTANCES + threadsPerBlock - 1) / threadsPerBlock;
    build_grid_kernel << <blocks_particles, threadsPerBlock >> > (
        d_grid_heads,
        d_grid_next,
        d_grid_keys,
        (const Vec3*)d_positions
        );
    cudaCheckError(cudaGetLastError());

    collide_kernel << <blocks_particles, threadsPerBlock >> > (
        (Mat4*)d_modelMatrices,
        (Vec3*)d_positions,
        (Vec3*)d_velocities,
        (const int*)d_grid_heads,
        (const int*)d_grid_next,
        (const Vec3*)d_positions,
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