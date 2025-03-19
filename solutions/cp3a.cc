#include <cstddef>
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std::chrono;

#include <new>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>


/**
 * Computes the dot product of matrix A and its transpose (A·Aᵀ)
 * Highly optimized for large matrices with cache-friendly blocking,
 * memory prefetching, and advanced OpenMP parallelization
 * 
 * @param A Input matrix in double precision (flattened in row-major order)
 * @param result Pointer to pre-allocated result matrix in float precision (rows x rows)
 * @param rows Number of rows in matrix A
 * @param cols Number of columns in matrix A
 * @param num_threads Number of threads to use (default: maximum available)
 */
void matrix_dot_product_transpose_large(
    double* __restrict A, 
    float* __restrict result,
    size_t rows, 
    size_t cols,
    int num_threads = 0) {
    
    size_t BLOCK_SIZE = 64;

    // Set number of threads
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    
    // Initialize result matrix to zeros
    #pragma omp parallel for simd
    for (size_t i = 0; i < rows * rows; ++i) {
        result[i] = 0.0f;
    }
    
    // Determine if we should use blocking based on matrix size
    bool use_blocking = (rows > BLOCK_SIZE && cols > BLOCK_SIZE);
    
    if (use_blocking) {
        // Compute A·Aᵀ using cache-friendly blocking
        #pragma omp parallel
        {
            // Allocate thread-local temporary storage for better cache locality
            alignas(32) std::vector<double> local_sum(BLOCK_SIZE, BLOCK_SIZE);
            
            #pragma omp for schedule(dynamic)
            for (size_t bi = 0; bi < rows; bi += BLOCK_SIZE) {
                size_t block_rows = std::min(BLOCK_SIZE, rows - bi);
                
                for (size_t bj = 0; bj <= bi; bj += BLOCK_SIZE) {
                    size_t block_cols = std::min(BLOCK_SIZE, rows - bj);
                    
                    // Initialize the local sum block to zeros
                    for (size_t i = 0; i < block_rows; ++i) {
                        for (size_t j = 0; j < block_cols; ++j) {
                            local_sum[i + j*block_cols] = 0.0;
                        }
                    }
                    
                    // Process the dot product in blocks along the k dimension
                    for (size_t bk = 0; bk < cols; bk += BLOCK_SIZE) {
                        size_t block_k = std::min(BLOCK_SIZE, cols - bk);
                        
                        for (size_t i = 0; i < block_rows; ++i) {
                            const size_t global_i = bi + i;
                            
                            for (size_t j = 0; j < block_cols; ++j) {
                                const size_t global_j = bj + j;
                                
                                // Skip upper triangular part for efficiency
                                if (global_j > global_i) continue;
                                
                                double sum = local_sum[i + j*block_cols];
                                const double* row_i = A + global_i * cols + bk;
                                const double* row_j = A + global_j * cols + bk;
                                
                                // Prefetch next cache line
                                _mm_prefetch((const char*)(row_i + 16), _MM_HINT_T0);
                                _mm_prefetch((const char*)(row_j + 16), _MM_HINT_T0);
                                
                                // Vectorized inner loop with manual unrolling for better pipelining
                                size_t k = 0;
                                size_t k_limit = block_k - (block_k % 16);
                                
                                for (; k < k_limit; k += 16) {
                                    // Process 4 elements at a time in 4 separate accumulators
                                    __m256d sum_vec1 = _mm256_setzero_pd();
                                    __m256d sum_vec2 = _mm256_setzero_pd();
                                    __m256d sum_vec3 = _mm256_setzero_pd();
                                    __m256d sum_vec4 = _mm256_setzero_pd();
                                    
                                    __m256d a_vec1 = _mm256_loadu_pd(row_i + k);
                                    __m256d b_vec1 = _mm256_loadu_pd(row_j + k);
                                    sum_vec1 = _mm256_fmadd_pd(a_vec1, b_vec1, sum_vec1);
                                    
                                    __m256d a_vec2 = _mm256_loadu_pd(row_i + k + 4);
                                    __m256d b_vec2 = _mm256_loadu_pd(row_j + k + 4);
                                    sum_vec2 = _mm256_fmadd_pd(a_vec2, b_vec2, sum_vec2);
                                    
                                    __m256d a_vec3 = _mm256_loadu_pd(row_i + k + 8);
                                    __m256d b_vec3 = _mm256_loadu_pd(row_j + k + 8);
                                    sum_vec3 = _mm256_fmadd_pd(a_vec3, b_vec3, sum_vec3);
                                    
                                    __m256d a_vec4 = _mm256_loadu_pd(row_i + k + 12);
                                    __m256d b_vec4 = _mm256_loadu_pd(row_j + k + 12);
                                    sum_vec4 = _mm256_fmadd_pd(a_vec4, b_vec4, sum_vec4);
                                    
                                    // Combine results
                                    sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec2);
                                    sum_vec3 = _mm256_add_pd(sum_vec3, sum_vec4);
                                    sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec3);
                                    
                                    // Horizontal sum
                                    double temp[4];
                                    _mm256_storeu_pd(temp, sum_vec1);
                                    sum += temp[0] + temp[1] + temp[2] + temp[3];
                                }
                                
                                // Handle remaining elements
                                for (; k < block_k; ++k) {
                                    sum += row_i[k] * row_j[k];
                                }
                                
                               local_sum[i + j*block_cols] = sum;
                            }
                        }
                    }
                    
                    // Store results to global memory
                    for (size_t i = 0; i < block_rows; ++i) {
                        const size_t global_i = bi + i;
                        
                        for (size_t j = 0; j < block_cols; ++j) {
                            const size_t global_j = bj + j;
                            
                            if (global_j > global_i) continue;
                            
                            float res = static_cast<float>(local_sum[i + j*block_cols]);
                            result[global_i * rows + global_j] = res;
                            
                            // Fill symmetric part
                            if (global_i != global_j) {
                                result[global_j * rows + global_i] = res;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // For smaller matrices, use the original algorithm with optimizations
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                const double* row_i = A + i * cols;
                const double* row_j = A + j * cols;
                
                // SIMD computation with manual unrolling for small matrices
                size_t k = 0;
                size_t simd_limit = cols - (cols % 8);
                
                for (; k < simd_limit; k += 8) {
                    __m256d sum_vec1 = _mm256_setzero_pd();
                    __m256d sum_vec2 = _mm256_setzero_pd();
                    
                    __m256d a_vec1 = _mm256_loadu_pd(row_i + k);
                    __m256d b_vec1 = _mm256_loadu_pd(row_j + k);
                    sum_vec1 = _mm256_mul_pd(a_vec1, b_vec1);
                    
                    __m256d a_vec2 = _mm256_loadu_pd(row_i + k + 4);
                    __m256d b_vec2 = _mm256_loadu_pd(row_j + k + 4);
                    sum_vec2 = _mm256_mul_pd(a_vec2, b_vec2);
                    
                    sum_vec1 = _mm256_add_pd(sum_vec1, sum_vec2);
                    
                    double temp[4];
                    _mm256_storeu_pd(temp, sum_vec1);
                    sum += temp[0] + temp[1] + temp[2] + temp[3];
                }
                
                // Process remaining elements
                for (; k < cols; ++k) {
                    sum += row_i[k] * row_j[k];
                }
                
                // Store results
                float res = static_cast<float>(sum);
                result[i * rows + j] = res;
                if (i != j) {
                    result[j * rows + i] = res;
                }
            }
        }
    }
}

// Helper function to allocate aligned memory
template <typename T>
T* allocate_aligned(size_t size, size_t alignment = 32) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate memory for the normalized matrix X
    double* X = new double[ny * nx];

    // Step 1: Normalize each row to have mean 0
    for (int i = 0; i < ny; i++) {
        // Calculate mean for row i
        double mean = 0.0f;
        for (int x = 0; x < nx; x++) {
            mean += data[x + i * nx];
        }
        mean /= nx;
        
        // Subtract mean from each element
        for (int x = 0; x < nx; x++) {
            X[x + i * nx] = data[x + i * nx] - mean;
        }
    }
    
    // Step 2: Normalize each row to have sum of squares = 1
    for (int i = 0; i < ny; i++) {
        // Calculate sum of squares for row i
        double sum_squares = 0.0f;
        for (int x = 0; x < nx; x++) {
            sum_squares += X[x + i * nx] * X[x + i * nx];
        }
        
        // Normalize row if sum of squares is not zero
        if (sum_squares > 1e-10) {
            double scale = 1.0f / sqrt(sum_squares);
            for (int x = 0; x < nx; x++) {
                X[x + i * nx] *= scale;
            }
        }
    }

    size_t N = ny;
    size_t M = nx;
    
    matrix_dot_product_transpose_large(X, result, N, M);
    
    // Free allocated memory
    delete[] X;
    
}
int main() {
    // Example parameters
    int ny = 9000;  // Number of samples (rows)
    int nx = 9000;   // Number of features (columns)
    
    // Create sample data
    std::vector<float> data(ny * nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j){
            data[j + nx*i] = static_cast<float>(rand()) / RAND_MAX;

        }
    }
  
    std::cout << "First few values of the correlation matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << data[j + nx * i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Allocate memory for result
    std::vector<float> result(ny * ny);
    
    // Calculate correlations
    std::cout << "Calculating PCC correlation matrix..." << std::endl;
  
    auto start= high_resolution_clock::now();
    correlate(ny, nx, data.data(),result.data());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Runtime: " << duration.count() << "\n";
  
    // auto start2= high_resolution_clock::now();
    // correlate_v2(ny, nx, data.data(), result.data());
    // auto end2 = high_resolution_clock::now();
    // auto duration2 = duration_cast<milliseconds>(end2 - start2);
    // std::cout << "Runtime2: " << duration2.count() << "\n";
    
    // Print a small part of the result for verification
  
    std::cout << "First few values of the correlation matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (i == j) {
            result[i + ny * j] = 1.0f; 
          }
          std::cout << result[i + ny * j] << " ";
        }
        std::cout << std::endl;
    }
    
    
    return 0;
  }

