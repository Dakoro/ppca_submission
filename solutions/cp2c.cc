
#include <cmath>
#include <immintrin.h> // For AVX/AVX2 intrinsics

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate memory for the normalized matrix X (aligned for AVX)
    double* X = static_cast<double*>(aligned_alloc(4, ny * nx * sizeof(double)));
    
    // Step 1: Normalize each row to have mean 0
    for (int i = 0; i < ny; i++) {
        // Calculate mean for row i using vectorization
        double mean = 0.0;
        int x = 0;
        
        // Use AVX for double precision (4 doubles per vector)
        __m256d sum_vec = _mm256_setzero_pd();
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            // Load 4 floats and convert to double
            __m128 data_vec = _mm_loadu_ps(&data[x + i * nx]);
            __m256d data_double = _mm256_cvtps_pd(data_vec);
            
            // Accumulate sum
            sum_vec = _mm256_add_pd(sum_vec, data_double);
        }
        
        // Horizontal sum of vector
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_vec);
        double vector_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        
        // Handle remaining elements
        for (; x < nx; x++) {
            mean += static_cast<double>(data[x + i * nx]);
        }
        
        mean = (vector_sum + mean) / nx;
        
        // Subtract mean from each element using vectorization
        x = 0;
        __m256d mean_vec = _mm256_set1_pd(mean);
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            // Load 4 floats and convert to double
            __m128 data_vec = _mm_loadu_ps(&data[x + i * nx]);
            __m256d data_double = _mm256_cvtps_pd(data_vec);
            
            // Subtract mean
            __m256d result_vec = _mm256_sub_pd(data_double, mean_vec);
            
            // Store result
            _mm256_storeu_pd(&X[x + i * nx], result_vec);
        }
        
        // Handle remaining elements
        for (; x < nx; x++) {
            X[x + i * nx] = static_cast<double>(data[x + i * nx]) - mean;
        }
    }
    
    // Step 2: Normalize each row to have sum of squares = 1
    for (int i = 0; i < ny; i++) {
        // Calculate sum of squares for row i using vectorization
        double sum_squares = 0.0;
        int x = 0;
        
        // Use AVX for double precision
        __m256d sum_sq_vec = _mm256_setzero_pd();
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            // Load 4 doubles
            __m256d x_vec = _mm256_loadu_pd(&X[x + i * nx]);
            
            // Square and accumulate
            sum_sq_vec = _mm256_fmadd_pd(x_vec, x_vec, sum_sq_vec);
        }
        
        // Horizontal sum of vector
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_sq_vec);
        double vector_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        
        // Handle remaining elements
        for (; x < nx; x++) {
            sum_squares += X[x + i * nx] * X[x + i * nx];
        }
        
        sum_squares = vector_sum + sum_squares;
        
        // Normalize row if sum of squares is not zero
        if (sum_squares > 1e-10) {
            double scale = 1.0 / sqrt(sum_squares);
            x = 0;
            
            // Use AVX for scaling
            __m256d scale_vec = _mm256_set1_pd(scale);
            
            // Process 4 elements at a time
            for (; x <= nx - 4; x += 4) {
                // Load 4 doubles
                __m256d x_vec = _mm256_loadu_pd(&X[x + i * nx]);
                
                // Scale
                __m256d result_vec = _mm256_mul_pd(x_vec, scale_vec);
                
                // Store result
                _mm256_storeu_pd(&X[x + i * nx], result_vec);
            }
            
            // Handle remaining elements
            for (; x < nx; x++) {
                X[x + i * nx] *= scale;
            }
        }
    }
    
    // Step 3: Calculate upper triangle of matrix product Y = X * X^T
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            // Calculate dot product of row i and row j using vectorization
            double dot_product = 0.0;
            int x = 0;
            
            // Use AVX for double precision
            __m256d dot_vec = _mm256_setzero_pd();
            
            // Process 4 elements at a time
            for (; x <= nx - 4; x += 4) {
                // Load 4 doubles from row i
                __m256d x_i_vec = _mm256_loadu_pd(&X[x + i * nx]);
                
                // Load 4 doubles from row j
                __m256d x_j_vec = _mm256_loadu_pd(&X[x + j * nx]);
                
                // Multiply and accumulate
                dot_vec = _mm256_fmadd_pd(x_i_vec, x_j_vec, dot_vec);
            }
            
            // Horizontal sum of vector
            double sum_arr[4];
            _mm256_storeu_pd(sum_arr, dot_vec);
            double vector_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            
            // Handle remaining elements
            for (; x < nx; x++) {
                dot_product += X[x + i * nx] * X[x + j * nx];
            }
            
            dot_product = vector_sum + dot_product;
            
            // Store in result matrix
            result[i + j * ny] = static_cast<float>(dot_product);
        }
    }
    
    // Free aligned memory
    free(X);
}