#include <algorithm> // for std::nth_element
#include <vector>

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    // For each pixel in the image
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            // Create a vector to store values within the window
            std::vector<float> window;
            window.reserve((2*hx+1) * (2*hy+1)); // Reserve maximum possible size
            
            // Gather values within the window
            for (int j = y - hy; j <= y + hy; j++) {
                for (int i = x - hx; i <= x + hx; i++) {
                    // Check if the coordinates are within image bounds
                    if (0 <= i && i < nx && 0 <= j && j < ny) {
                        window.push_back(in[i + nx*j]);
                    }
                }
            }
            
            int window_size = window.size();
            
            // Calculate median using std::nth_element
            if (window_size % 2 == 1) {
                // Odd number of elements - find the middle element
                int middle = window_size / 2;
                std::nth_element(window.begin(), window.begin() + middle, window.end());
                out[x + nx*y] = window[middle];
            } else {
                // Even number of elements - find and average the two middle elements
                int middle_right = window_size / 2;
                int middle_left = middle_right - 1;
                
                // Find the middle_left element (k-th smallest)
                std::nth_element(window.begin(), window.begin() + middle_left, window.end());
                float left_median = window[middle_left];
                
                // Find the middle_right element (k+1-th smallest)
                std::nth_element(window.begin() + middle_left + 1, window.begin() + middle_right, window.end());
                float right_median = window[middle_right];
                
                // Calculate the median as the average of the two middle elements
                out[x + nx*y] = (left_median + right_median) / 2.0f;
            }
        }
    }
}