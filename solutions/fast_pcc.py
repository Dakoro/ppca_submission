import numpy as np
import cupy as cp  # GPU-accelerated array library

def fast_gpu_pcc(X, Y=None):
    """
    Fast GPU-based Pearson Correlation Coefficient calculation
    
    Parameters:
    -----------
    X : numpy.ndarray or cupy.ndarray
        Input matrix with shape (n_samples, n_features)
        If Y is None, calculate correlation between columns of X
    Y : numpy.ndarray or cupy.ndarray, optional
        Second input matrix with shape (n_samples, m_features)
        If provided, calculate correlation between columns of X and Y
    
    Returns:
    --------
    corr_matrix : cupy.ndarray
        Correlation matrix with shape:
        - (n_features, n_features) if Y is None
        - (n_features, m_features) if Y is provided
    """
    # Transfer data to GPU if not already there
    if isinstance(X, np.ndarray):
        X_gpu = cp.asarray(X)
    else:
        X_gpu = X
    
    # Compute correlation between columns of X
    if Y is None:
        # Get dimensions
        n_samples, n_features = X_gpu.shape
        
        # Center the data (subtract mean of each column)
        X_centered = X_gpu - cp.mean(X_gpu, axis=0)
        
        # Compute standard deviation for each column
        X_std = cp.std(X_gpu, axis=0)
        
        # Handle zero standard deviation
        X_std[X_std == 0] = 1.0
        
        # Normalize the centered data
        X_normalized = X_centered / X_std
        
        # Compute correlation matrix using matrix multiplication
        # (X'X)/(n-1) for normalized data gives the correlation matrix
        corr_matrix = cp.dot(X_normalized.T, X_normalized) / (n_samples - 1)
        
        # Ensure the diagonal is exactly 1.0 (floating point precision issues)
        cp.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix
    
    # Compute correlation between columns of X and columns of Y
    else:
        # Transfer Y to GPU if needed
        if isinstance(Y, np.ndarray):
            Y_gpu = cp.asarray(Y)
        else:
            Y_gpu = Y
        
        # Get dimensions
        n_samples, n_features_x = X_gpu.shape
        _, n_features_y = Y_gpu.shape
        
        # Center the data
        X_centered = X_gpu - cp.mean(X_gpu, axis=0)
        Y_centered = Y_gpu - cp.mean(Y_gpu, axis=0)
        
        # Compute standard deviations
        X_std = cp.std(X_gpu, axis=0)
        Y_std = cp.std(Y_gpu, axis=0)
        
        # Handle zero standard deviations
        X_std[X_std == 0] = 1.0
        Y_std[Y_std == 0] = 1.0
        
        # Normalize the centered data
        X_normalized = X_centered / X_std
        Y_normalized = Y_centered / Y_std
        
        # Compute correlation matrix
        corr_matrix = cp.dot(X_normalized.T, Y_normalized) / (n_samples - 1)
        
        return corr_matrix

def fast_gpu_pcc_batch(X, batch_size=1000):
    """
    Memory-efficient implementation for large matrices using batching
    
    Parameters:
    -----------
    X : numpy.ndarray or cupy.ndarray
        Input matrix with shape (n_samples, n_features)
    batch_size : int
        Size of batches to use when n_features is large
        
    Returns:
    --------
    corr_matrix : cupy.ndarray
        Correlation matrix with shape (n_features, n_features)
    """
    # Transfer to GPU if not already there
    if isinstance(X, np.ndarray):
        X_gpu = cp.asarray(X)
    else:
        X_gpu = X
    
    n_samples, n_features = X_gpu.shape
    
    # Center and normalize the data
    X_centered = X_gpu - cp.mean(X_gpu, axis=0)
    X_std = cp.std(X_gpu, axis=0)
    X_std[X_std == 0] = 1.0
    X_normalized = X_centered / X_std
    
    # For small feature sets, use direct method
    if n_features <= batch_size:
        return cp.dot(X_normalized.T, X_normalized) / (n_samples - 1)
    
    # For large feature sets, compute correlation matrix in batches
    corr_matrix = cp.zeros((n_features, n_features), dtype=cp.float32)
    
    for i in range(0, n_features, batch_size):
        i_end = min(i + batch_size, n_features)
        batch_i = X_normalized[:, i:i_end]
        
        # Compute correlation between current batch and all features
        for j in range(0, n_features, batch_size):
            j_end = min(j + batch_size, n_features)
            batch_j = X_normalized[:, j:j_end]
            
            # Compute batch correlation
            batch_corr = cp.dot(batch_i.T, batch_j) / (n_samples - 1)
            
            # Update the correlation matrix
            corr_matrix[i:i_end, j:j_end] = batch_corr
    
    # Ensure diagonal elements are exactly 1.0
    cp.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix

# Example usage
if __name__ == "__main__":
    # Generate random data
    n_samples = 10000
    n_features = 500
    
    # Create sample data on CPU
    X_cpu = np.random.randn(n_samples, n_features)
    Y_cpu = np.random.randn(n_samples, 200)
    
    # Transfer to GPU
    X_gpu = cp.asarray(X_cpu)
    Y_gpu = cp.asarray(Y_cpu)
    
    # Calculate PCC
    print("Calculating X self-correlation...")
    corr_X = fast_gpu_pcc(X_gpu)
    
    print("Calculating X-Y correlation...")
    corr_XY = fast_gpu_pcc(X_gpu, Y_gpu)
    
    print("Calculating large matrix correlation with batching...")
    # Create larger test data
    large_X = cp.random.randn(5000, 5000).astype(cp.float32)
    corr_large = fast_gpu_pcc_batch(large_X, batch_size=1000)
    
    print("Correlation matrix shapes:")
    print(f"X self-correlation: {corr_X.shape}")
    print(f"X-Y correlation: {corr_XY.shape}")
    print(f"Large matrix correlation: {corr_large.shape}")
    
    # Optionally transfer results back to CPU if needed
    # corr_X_cpu = cp.asnumpy(corr_X)