#include <immintrin.h>
#include <cmath>
#include <cstdlib>

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate aligned memory for intermediate matrix X
    double *X = static_cast<double*>(_mm_malloc(ny * nx * sizeof(double), 32));
    if (!X) return; // Handle allocation failure if necessary

    // Step 1: Normalize each row to mean zero
    for (int i = 0; i < ny; i++) {
        // Compute sum of the row using AVX
        __m256d sum_vec = _mm256_setzero_pd();
        int x;
        for (x = 0; x <= nx - 4; x += 4) {
            // Load four floats and convert to doubles
            __m128 float_vec = _mm_loadu_ps(&data[i * nx + x]);
            __m256d double_vec = _mm256_cvtps_pd(float_vec);
            sum_vec = _mm256_add_pd(sum_vec, double_vec);
        }
        // Horizontal sum of sum_vec
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        double sum = temp[0] + temp[1] + temp[2] + temp[3];
        // Handle remaining elements
        for (; x < nx; x++) {
            sum += static_cast<double>(data[i * nx + x]);
        }
        double mean = sum / nx;

        // Subtract mean from each element using AVX
        __m256d mean_vec = _mm256_set1_pd(mean);
        for (x = 0; x <= nx - 4; x += 4) {
            __m128 float_vec = _mm_loadu_ps(&data[i * nx + x]);
            __m256d double_vec = _mm256_cvtps_pd(float_vec);
            __m256d result_vec = _mm256_sub_pd(double_vec, mean_vec);
            _mm256_store_pd(&X[i * nx + x], result_vec);
        }
        // Handle remaining elements
        for (; x < nx; x++) {
            X[i * nx + x] = static_cast<double>(data[i * nx + x]) - mean;
        }
    }

    // Step 2: Normalize each row to unit length
    for (int i = 0; i < ny; i++) {
        // Compute sum of squares using AVX
        __m256d sum_sq_vec = _mm256_setzero_pd();
        int x;
        for (x = 0; x <= nx - 4; x += 4) {
            __m256d x_vec = _mm256_load_pd(&X[i * nx + x]);
            __m256d sq_vec = _mm256_mul_pd(x_vec, x_vec);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_vec);
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum_sq_vec);
        double sum_squares = temp[0] + temp[1] + temp[2] + temp[3];
        for (; x < nx; x++) {
            double val = X[i * nx + x];
            sum_squares += val * val;
        }

        // Scale if sum_squares is significant
        if (sum_squares > 1e-10) {
            double scale = 1.0 / std::sqrt(sum_squares);
            __m256d scale_vec = _mm256_set1_pd(scale);
            for (x = 0; x <= nx - 4; x += 4) {
                __m256d x_vec = _mm256_load_pd(&X[i * nx + x]);
                __m256d result_vec = _mm256_mul_pd(x_vec, scale_vec);
                _mm256_store_pd(&X[i * nx + x], result_vec);
            }
            for (; x < nx; x++) {
                X[i * nx + x] *= scale;
            }
        }
    }

    // Step 3: Compute upper triangle of correlation matrix
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            // Compute dot product using AVX
            __m256d dot_vec = _mm256_setzero_pd();
            int x;
            for (x = 0; x <= nx - 4; x += 4) {
                __m256d xi_vec = _mm256_load_pd(&X[i * nx + x]);
                __m256d xj_vec = _mm256_load_pd(&X[j * nx + x]);
                __m256d prod_vec = _mm256_mul_pd(xi_vec, xj_vec);
                dot_vec = _mm256_add_pd(dot_vec, prod_vec);
            }
            double temp[4];
            _mm256_storeu_pd(temp, dot_vec);
            double dot_product = temp[0] + temp[1] + temp[2] + temp[3];
            for (; x < nx; x++) {
                dot_product += X[i * nx + x] * X[j * nx + x];
            }
            result[i + j * ny] = static_cast<float>(dot_product);
        }
    }

    // Free aligned memory
    _mm_free(X);
}