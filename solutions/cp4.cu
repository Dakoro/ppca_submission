#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

// Kernel to precompute means and sum of squares for each row
__global__ void computeMeansAndSumSq(int ny, int nx, const float* data, float* means, float* sum_sq) {
    int i = blockIdx.x; // Each block processes one row
    if (i < ny) {
        const float* row = &data[i * nx];
        __shared__ float partial_sum[256]; // Shared memory for reduction, assuming blockDim.x <= 256

        // Step 1: Compute sum for mean
        float sum = 0.0f;
        for (int k = threadIdx.x; k < nx; k += blockDim.x) {
            sum += row[k];
        }
        partial_sum[threadIdx.x] = sum;
        __syncthreads();

        // Reduction within block to compute total sum
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Compute mean and store
        if (threadIdx.x == 0) {
            means[i] = partial_sum[0] / nx;
        }
        __syncthreads();

        float mean = means[i];

        // Step 2: Compute sum of squares
        float sq_sum = 0.0f;
        for (int k = threadIdx.x; k < nx; k += blockDim.x) {
            float val = row[k] - mean;
            sq_sum += val * val;
        }
        partial_sum[threadIdx.x] = sq_sum;
        __syncthreads();

        // Reduction for sum of squares
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            sum_sq[i] = partial_sum[0];
        }
    }
}

// Optimized kernel to compute correlations using precomputed means and sum of squares
__global__ void correlateKernel(int ny, int nx, const float* data, const float* means, 
                              const float* sum_sq, float* result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j <= i && i < ny && j < ny) { // Compute upper triangle
        const float* row_i = &data[i * nx];
        const float* row_j = &data[j * nx];
        float mean_i = means[i];
        float mean_j = means[j];

        // Compute numerator
        float numerator = 0.0f;
        for (int k = 0; k < nx; k++) {
            float val_i = row_i[k] - mean_i;
            float val_j = row_j[k] - mean_j;
            numerator += val_i * val_j;
        }

        // Compute correlation using precomputed sum of squares
        float denominator = sqrt(sum_sq[i] * sum_sq[j]);
        float correlation = (denominator > 1e-10f) ? (numerator / denominator) : 0.0f;

        // Store result
        result[i + j * ny] = correlation;
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
    // Allocate device memory
    float *d_data = NULL, *d_result = NULL, *d_means = NULL, *d_sum_sq = NULL;

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    int threadsPerBlock = 256; // For means and sum_sq computation
    int blocksPerGrid = ny;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_data, ny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_result, ny * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_means, ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_sum_sq, ny * sizeof(float)));

    // Copy input data to device
    CHECK(cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_result, 0, ny * ny * sizeof(float)));

    // Step 1: Precompute means and sum of squares
    computeMeansAndSumSq<<<blocksPerGrid, threadsPerBlock>>>(ny, nx, d_data, d_means, d_sum_sq);
    CHECK(cudaGetLastError());

    // Step 2: Compute correlations
    correlateKernel<<<gridDim, blockDim>>>(ny, nx, d_data, d_means, d_sum_sq, d_result);
    CHECK(cudaGetLastError());

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_means);
    cudaFree(d_sum_sq);
}

int main() {
    // Example parameters
    int ny = 5;  // Number of samples (rows)
    int nx = 2;   // Number of features (columns)
    
    // Create sample data
    std::vector<float> data(ny * nx);
    for (int i = 0; i < ny * nx; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "First few values of the correlation matrix:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << data[i * nx + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Allocate memory for result
    std::vector<float> result(ny * ny);
    
    // Calculate correlations
    std::cout << "Calculating PCC correlation matrix..." << std::endl;

    auto start= high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Runtime1: " << duration.count() << "\n";

    // auto start2= high_resolution_clock::now();
    // correlate_v2(ny, nx, data.data(), result.data());
    // auto end2 = high_resolution_clock::now();
    // auto duration2 = duration_cast<milliseconds>(end2 - start2);
    // std::cout << "Runtime2: " << duration2.count() << "\n";
    
    // Print a small part of the result for verification

    std::cout << "First few values of the correlation matrix:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << result[i * nx + j] << " ";
        }
        std::cout << std::endl;
    }
    
    
    return 0;
}