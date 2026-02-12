import numpy as np
from typing import Tuple


def sample_gp_functions(
    num_samples: int,
    num_points: int,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_variance: float = 1.0,
    noise_variance: float = 0.01,
    x_dim: int = 1,
    y_dim: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample functions from a Gaussian Process.
    
    Args:
        num_samples: Number of function samples to generate
        num_points: Number of points per function
        x_range: Range of x values (min, max)
        kernel_length_scale: Length scale of RBF kernel
        kernel_variance: Variance of RBF kernel
        noise_variance: Observation noise variance
        x_dim: Dimension of input (1 for 1D functions)
        y_dim: Dimension of output (1 for 1D functions)
        
    Returns:
        x: Input locations [num_samples, num_points, x_dim]
        y: Output values [num_samples, num_points, y_dim]
    """
    x_min, x_max = x_range
    
    # Sample random x locations
    x = np.random.uniform(x_min, x_max, size=(num_samples, num_points, x_dim))
    
    # Initialize y
    y = np.zeros((num_samples, num_points, y_dim))
    
    # Sample from GP for each batch and output dimension
    for i in range(num_samples):
        for d in range(y_dim):
            # Compute RBF kernel matrix
            x_i = x[i, :, 0]  # [num_points]
            x_diff = x_i[:, None] - x_i[None, :]  # [num_points, num_points]
            kernel_matrix = kernel_variance * np.exp(
                -0.5 * x_diff ** 2 / kernel_length_scale ** 2
            )
            
            # Add noise to diagonal for numerical stability
            kernel_matrix += (noise_variance + 1e-6) * np.eye(num_points)
            
            # Sample from multivariate normal
            mean = np.zeros(num_points)
            y[i, :, d] = np.random.multivariate_normal(mean, kernel_matrix)
    
    return x, y
