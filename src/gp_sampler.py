import numpy as np
from typing import Callable, Tuple


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
            # Compute RBF kernel matrix using all input dimensions
            x_i = x[i, :, :]  # [num_points, x_dim]
            # Compute pairwise squared Euclidean distances across all dimensions
            x_diff_sq = np.sum((x_i[:, None, :] - x_i[None, :, :]) ** 2, axis=2)  # [num_points, num_points]
            kernel_matrix = kernel_variance * np.exp(
                -0.5 * x_diff_sq / kernel_length_scale ** 2
            )
            
            # Add noise to diagonal for numerical stability
            kernel_matrix += (noise_variance + 1e-6) * np.eye(num_points)
            
            # Sample from multivariate normal
            mean = np.zeros(num_points)
            y[i, :, d] = np.random.multivariate_normal(mean, kernel_matrix)
    
    return x, y


def make_gp_sampler(
    batch_size: int = 16,
    num_context_range: Tuple[int, int] = (3, 50),
    num_total_points: int = 100,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_variance: float = 1.0,
    noise_variance: float = 0.01,
    x_dim: int = 1,
    y_dim: int = 1
) -> Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a sampler that returns context/target batches from GP samples.

    Returns:
        Callable that yields (context_x, context_y, target_x, target_y)
    """
    def sample_batch() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y = sample_gp_functions(
            num_samples=batch_size,
            num_points=num_total_points,
            x_range=x_range,
            kernel_length_scale=kernel_length_scale,
            kernel_variance=kernel_variance,
            noise_variance=noise_variance,
            x_dim=x_dim,
            y_dim=y_dim
        )

        num_context = np.random.randint(num_context_range[0], num_context_range[1] + 1)
        indices = np.random.permutation(num_total_points)
        context_indices = indices[:num_context]
        target_indices = indices[num_context:]

        context_x = x[:, context_indices, :]
        context_y = y[:, context_indices, :]
        target_x = x[:, target_indices, :]
        target_y = y[:, target_indices, :]

        return context_x, context_y, target_x, target_y

    return sample_batch
