#include "mm.h"
#include <cstdint>
#include <cstdlib>

void gemm(int m, int n, int k, const int8_t* A, const int8_t* B, int32_t* C) {
    // Allocate memory for the transposed B matrix
    int8_t* B_transposed = new int8_t[k * n];
    
    // Transpose B matrix - fix indexing error
    // Original B[i, j] = B[i*n + j]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            // Corrected: B_transposed[j, i] = B[i, j]
            B_transposed[j * k + i] = B[i * n + j];
        }
    }
    
    // Perform matrix multiplication with transposed B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            // A[i, :] starts at i*k
            const int8_t* A_row = &A[i * k];
            // B_transposed[j, :] starts at j*k
            const int8_t* B_col = &B_transposed[j * k];
            
            // Now we can access both matrices in a row-wise fashion
            for (int l = 0; l < k; l++) {
                sum += (int32_t)A_row[l] * (int32_t)B_col[l];
            }
            
            C[i * n + j] = sum;
        }
    }
    
    // Free the allocated memory
    delete[] B_transposed;
}