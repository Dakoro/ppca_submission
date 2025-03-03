#include <cmath>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result best_result;
    float best_error = INFINITY;

    // For each possible rectangle
    for (int y0 = 0; y0 < ny; y0++) {
        for (int x0 = 0; x0 < nx; x0++) {
            for (int y1 = y0 + 1; y1 <= ny; y1++) {
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    // Count pixels inside and outside the rectangle
                    int inner_count = (y1 - y0) * (x1 - x0);
                    int outer_count = ny * nx - inner_count;

                    // Initialize sums for inner and outer regions
                    float inner_sum[3] = {0, 0, 0};
                    float outer_sum[3] = {0, 0, 0};

                    // Compute sum of color components for inner and outer regions
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            bool is_inner = (y >= y0 && y < y1 && x >= x0 && x < x1);
                            
                            for (int c = 0; c < 3; c++) {
                                float val = data[c + 3 * x + 3 * nx * y];
                                if (is_inner) {
                                    inner_sum[c] += val;
                                } else {
                                    outer_sum[c] += val;
                                }
                            }
                        }
                    }

                    // Calculate average colors
                    float inner_avg[3], outer_avg[3];
                    for (int c = 0; c < 3; c++) {
                        inner_avg[c] = inner_count > 0 ? inner_sum[c] / inner_count : 0;
                        outer_avg[c] = outer_count > 0 ? outer_sum[c] / outer_count : 0;
                    }

                    // Compute sum of squared errors
                    float error = 0;
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            bool is_inner = (y >= y0 && y < y1 && x >= x0 && x < x1);
                            float* avg = is_inner ? inner_avg : outer_avg;
                            
                            for (int c = 0; c < 3; c++) {
                                float val = data[c + 3 * x + 3 * nx * y];
                                float diff = val - avg[c];
                                error += diff * diff;
                            }
                        }
                    }

                    // Update best result if this rectangle has lower error
                    if (error < best_error) {
                        best_error = error;
                        best_result.y0 = y0;
                        best_result.x0 = x0;
                        best_result.y1 = y1;
                        best_result.x1 = x1;
                        for (int c = 0; c < 3; c++) {
                            best_result.inner[c] = inner_avg[c];
                            best_result.outer[c] = outer_avg[c];
                        }
                    }
                }
            }
        }
    }

    return best_result;
}
