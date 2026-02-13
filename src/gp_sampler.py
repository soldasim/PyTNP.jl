import numpy as np
from typing import Callable, Optional, Tuple, Union

PriorSpec = Optional[Union[Callable[[], float], Tuple[float, float]]]
KernelFn = Callable[[np.ndarray, float, float], np.ndarray]


def _sample_prior(prior: PriorSpec, fallback: float) -> float:
    if prior is None:
        return float(fallback)
    if callable(prior):
        return float(prior())
    if isinstance(prior, tuple) and len(prior) == 2:
        low, high = prior
        return float(np.random.uniform(low, high))
    raise ValueError("Prior must be a callable or a (low, high) tuple.")


def _rbf_kernel(
    x_diff_sq: np.ndarray,
    length_scale: float,
    kernel_std: float
) -> np.ndarray:
    return (kernel_std ** 2) * np.exp(-0.5 * x_diff_sq / length_scale ** 2)


def _matern32_kernel(
    x_diff_sq: np.ndarray,
    length_scale: float,
    kernel_std: float
) -> np.ndarray:
    r = np.sqrt(x_diff_sq) / length_scale
    sqrt3_r = np.sqrt(3.0) * r
    return (kernel_std ** 2) * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)


def _matern52_kernel(
    x_diff_sq: np.ndarray,
    length_scale: float,
    kernel_std: float
) -> np.ndarray:
    r = np.sqrt(x_diff_sq) / length_scale
    sqrt5_r = np.sqrt(5.0) * r
    return (kernel_std ** 2) * (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * np.exp(-sqrt5_r)


def _build_kernel_matrix(
    x_diff_sq: np.ndarray,
    length_scale: float,
    kernel_std: float,
    kernel_fn: KernelFn = _rbf_kernel
) -> np.ndarray:
    return kernel_fn(x_diff_sq, length_scale, kernel_std)


def sample_gp_functions(
    num_samples: int,
    num_points: int,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_std: float = 1.0,
    noise_std: float = 0.1,
    kernel_fn: KernelFn = _rbf_kernel,
    kernel_length_scale_prior: PriorSpec = None,
    kernel_std_prior: PriorSpec = None,
    noise_std_prior: PriorSpec = None,
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
        kernel_std: Std dev of RBF kernel
        noise_std: Observation noise std dev
        kernel_fn: Optional custom kernel callable (defaults to RBF)
        kernel_length_scale_prior: Prior for length scale (callable or (low, high))
        kernel_std_prior: Prior for kernel std dev (callable or (low, high))
        noise_std_prior: Prior for noise std dev (callable or (low, high))
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
        # Sample hyperparameters per function if priors are provided.
        length_scale_i = _sample_prior(kernel_length_scale_prior, kernel_length_scale)
        kernel_std_i = _sample_prior(kernel_std_prior, kernel_std)
        noise_std_i = _sample_prior(noise_std_prior, noise_std)
        if length_scale_i <= 0 or kernel_std_i <= 0 or noise_std_i < 0:
            raise ValueError("Hyperparameters must be positive (noise_std can be zero).")
        
        for d in range(y_dim):
            # Compute RBF kernel matrix using all input dimensions
            x_i = x[i, :, :]  # [num_points, x_dim]
            # Compute pairwise squared Euclidean distances across all dimensions
            x_diff_sq = np.sum((x_i[:, None, :] - x_i[None, :, :]) ** 2, axis=2)  # [num_points, num_points]
            kernel_matrix = _build_kernel_matrix(
                x_diff_sq,
                length_scale_i,
                kernel_std_i,
                kernel_fn=kernel_fn
            )
            
            # Add noise to diagonal for numerical stability
            kernel_matrix += (noise_std_i ** 2) * np.eye(num_points)
            
            # Sample from multivariate normal
            mean = np.zeros(num_points)
            y[i, :, d] = np.random.multivariate_normal(mean, kernel_matrix)
    
    return x, y


def make_gp_sampler(
    batch_size: int = 32,
    num_total_points_range: Tuple[int, int] = (2, 101),
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_std: float = 1.0,
    noise_std: float = 0.1,
    kernel_fn: KernelFn = _rbf_kernel,
    kernel_length_scale_prior: PriorSpec = None,
    kernel_std_prior: PriorSpec = None,
    noise_std_prior: PriorSpec = None,
    x_dim: int = 1,
    y_dim: int = 1
) -> Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a sampler that returns context/target batches from GP samples.

    Returns:
        Callable that yields (context_x, context_y, target_x, target_y)
    """
    def sample_batch() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_total_points = np.random.randint(num_total_points_range[0], num_total_points_range[1] + 1)

        x, y = sample_gp_functions(
            num_samples=batch_size,
            num_points=num_total_points,
            x_range=x_range,
            kernel_length_scale=kernel_length_scale,
            kernel_std=kernel_std,
            noise_std=noise_std,
            kernel_fn=kernel_fn,
            kernel_length_scale_prior=kernel_length_scale_prior,
            kernel_std_prior=kernel_std_prior,
            noise_std_prior=noise_std_prior,
            x_dim=x_dim,
            y_dim=y_dim
        )

        num_context = np.random.randint(1, num_total_points - 1 + 1)
        indices = np.random.permutation(num_total_points)
        context_indices = indices[:num_context]
        target_indices = indices[num_context:]

        context_x = x[:, context_indices, :]
        context_y = y[:, context_indices, :]
        target_x = x[:, target_indices, :]
        target_y = y[:, target_indices, :]

        return context_x, context_y, target_x, target_y

    return sample_batch
