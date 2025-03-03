#include <cmath>
#include <immintrin.h> // For AVX/SSE intrinsics

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate memory for the normalized matrix X with aligned allocation
    double* X = static_cast<double*>(aligned_alloc(4, ny * nx * sizeof(double)));
    
    // Step 1: Normalize each row to have mean 0
    for (int i = 0; i < ny; i++) {
        // Calculate mean for row i using loop unrolling
        double mean = 0.0;
        int x = 0;
        
        // Four accumulators for better instruction-level parallelism
        double mean0 = 0.0, mean1 = 0.0, mean2 = 0.0, mean3 = 0.0;
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            mean0 += data[x + i * nx];
            mean1 += data[x + 1 + i * nx];
            mean2 += data[x + 2 + i * nx];
            mean3 += data[x + 3 + i * nx];
        }
        
        // Handle remaining elements
        for (; x < nx; x++) {
            mean0 += data[x + i * nx];
        }
        
        // Combine partial sums
        mean = (mean0 + mean1 + mean2 + mean3) / nx;
        
        // Subtract mean from each element with loop unrolling
        x = 0;
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            X[x + i * nx] = data[x + i * nx] - mean;
            X[x + 1 + i * nx] = data[x + 1 + i * nx] - mean;
            X[x + 2 + i * nx] = data[x + 2 + i * nx] - mean;
            X[x + 3 + i * nx] = data[x + 3 + i * nx] - mean;
        }
        
        // Handle remaining elements
        for (; x < nx; x++) {
            X[x + i * nx] = data[x + i * nx] - mean;
        }
    }
    
    // Step 2: Normalize each row to have sum of squares = 1
    for (int i = 0; i < ny; i++) {
        // Calculate sum of squares for row i with loop unrolling
        double sum_squares = 0.0;
        int x = 0;
        
        // Four accumulators for better instruction-level parallelism
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        
        // Process 4 elements at a time
        for (; x <= nx - 4; x += 4) {
            sum0 += X[x + i * nx] * X[x + i * nx];
            sum1 += X[x + 1 + i * nx] * X[x + 1 + i * nx];
            sum2 += X[x + 2 + i * nx] * X[x + 2 + i * nx];
            sum3 += X[x + 3 + i * nx] * X[x + 3 + i * nx];
        }
        
        // Handle remaining elements
        for (; x < nx; x++) {
            sum0 += X[x + i * nx] * X[x + i * nx];
        }
        
        // Combine partial sums
        sum_squares = sum0 + sum1 + sum2 + sum3;
        
        // Normalize row if sum of squares is not zero
        if (sum_squares > 1e-10) {
            double scale = 1.0 / sqrt(sum_squares);
            
            x = 0;
            // Process 4 elements at a time
            for (; x <= nx - 4; x += 4) {
                X[x + i * nx] *= scale;
                X[x + 1 + i * nx] *= scale;
                X[x + 2 + i * nx] *= scale;
                X[x + 3 + i * nx] *= scale;
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
            // Calculate dot product of row i and row j with loop unrolling
            double dot_product = 0.0;
            int x = 0;
            
            // Multiple accumulators for better instruction-level parallelism
            double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;
            
            // Process 4 elements at a time
            for (; x <= nx - 4; x += 4) {
                dot0 += X[x + i * nx] * X[x + j * nx];
                dot1 += X[x + 1 + i * nx] * X[x + 1 + j * nx];
                dot2 += X[x + 2 + i * nx] * X[x + 2 + j * nx];
                dot3 += X[x + 3 + i * nx] * X[x + 3 + j * nx];
            }
            
            // Handle remaining elements
            for (; x < nx; x++) {
                dot0 += X[x + i * nx] * X[x + j * nx];
            }
            
            // Combine partial sums
            dot_product = dot0 + dot1 + dot2 + dot3;
            
            // Store in result matrix
            result[i + j * ny] = dot_product;
        }
    }
    
    // Free aligned memory
    free(X);
}


