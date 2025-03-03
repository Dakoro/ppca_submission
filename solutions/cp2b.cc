#include <cmath>
#include <omp.h>

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate memory for the normalized matrix X
    double* X = new double[ny * nx];

    // Step 1: Normalize each row to have mean 0
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        // Calculate mean for row i
        double mean = 0.0;
        for (int x = 0; x < nx; x++) {
            mean += static_cast<double>(data[x + i * nx]);
        }
        mean /= nx;
        
        // Subtract mean from each element
        for (int x = 0; x < nx; x++) {
            X[x + i * nx] = static_cast<double>(data[x + i * nx]) - mean;
        }
    }
    
    // Step 2: Normalize each row to have sum of squares = 1
    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        // Calculate sum of squares for row i
        double sum_squares = 0.0;
        for (int x = 0; x < nx; x++) {
            sum_squares += X[x + i * nx] * X[x + i * nx];
        }
        
        // Normalize row if sum of squares is not zero
        if (sum_squares > 1e-10) {
            double scale = 1.0 / sqrt(sum_squares);
            for (int x = 0; x < nx; x++) {
                X[x + i * nx] *= scale;
            }
        }
    }
    
    // Step 3: Calculate upper triangle of matrix product Y = X * X^T
    // This step is parallelized with a guided schedule to handle load imbalance
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            // Calculate dot product of row i and row j
            double dot_product = 0.0;
            for (int x = 0; x < nx; x++) {
                dot_product += X[x + i * nx] * X[x + j * nx];
            }
            
            // Store in result matrix
            result[i + j * ny] = static_cast<float>(dot_product);
        }
    }
    
    // Free allocated memory
    delete[] X;
}
