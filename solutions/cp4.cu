#include <cuda_runtime.h>
#include <iostream>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

// Single kernel to compute all correlations
__global__ void correlateKernel(int ny, int nx, const float* data, float* result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j <= i && i < ny) {  // Only compute upper triangle
        // Get pointers to the two rows
        const float* row_i = &data[i * nx];
        const float* row_j = &data[j * nx];
        
        // Step 1: Calculate means of both rows
        double mean_i = 0.0;
        double mean_j = 0.0;
        for (int k = 0; k < nx; k++) {
            mean_i += row_i[k];
            mean_j += row_j[k];
        }
        mean_i /= nx;
        mean_j /= nx;
        
        // Step 2: Calculate normalized dot product
        double numerator = 0.0;
        double sum_sq_i = 0.0;
        double sum_sq_j = 0.0;
        
        for (int k = 0; k < nx; k++) {
            double val_i = row_i[k] - mean_i;
            double val_j = row_j[k] - mean_j;
            
            numerator += val_i * val_j;
            sum_sq_i += val_i * val_i;
            sum_sq_j += val_j * val_j;
        }
        
        // Step 3: Compute correlation
        double denominator = sqrt(sum_sq_i * sum_sq_j);
        double correlation = 0.0;
        
        if (denominator > 1e-10) {
            correlation = numerator / denominator;
        }
        
        // Store result
        result[i + j * ny] = correlation;
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
    // Allocate device memory
    float *d_data = NULL, *d_result = NULL;
    cudaError_t err;
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((ny + blockDim.x - 1) / blockDim.x, 
                 (ny + blockDim.y - 1) / blockDim.y);
                 
    // Allocate memory for input data
    CHECK(cudaMalloc((void**)&d_data, ny * nx * sizeof(float)));
    
    // Allocate memory for result
    CHECK(cudaMalloc((void**)&d_result, ny * ny * sizeof(float)));
    
    // Copy input data to device
    CHECK(cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize result matrix with zeros to avoid uninitialized memory issues
    CHECK(cudaMemset(d_result, 0, ny * ny * sizeof(float)));
    
    // Launch kernel
    correlateKernel<<<gridDim, blockDim>>>(ny, nx, d_data, d_result);
    
    // Check for kernel errors
    CHECK(cudaGetLastError());
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Copy result back to host
    CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    if (d_data) cudaFree(d_data);
    if (d_result) cudaFree(d_result);
}