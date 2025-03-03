#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <random>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void mykernel(float* r, const float* d, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= n || j >= n)
        return;
    float v = HUGE_VALF;
    for (int k = 0; k < n; ++k) {
        float x = d[n*i + k];
        float y = d[n*k + j];
        float z = x + y;
        v = min(v, z);
    }
    r[n*i + j] = v;
}

void step(float* r, const float* d, int n) {
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}


int main() {
    constexpr int n = 4000;
    static float d[n*n];
    std::srand( ( unsigned int )std::time( nullptr ) );
    for (int i = 0; i < n*n; ++i) {
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Use Mersenne Twister engine
        std::uniform_real_distribution<> dis(0.0, 1.0); // Define the range
        float random_float = dis(gen);
        d[i] = random_float;
    }

    static float r[n*n];

    auto start = std::chrono::high_resolution_clock::now();
    step(r, d, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << ms_int.count() << " ms";
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << r[i*n + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
}