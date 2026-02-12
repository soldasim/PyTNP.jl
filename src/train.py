import torch
import numpy as np
from typing import Optional, Tuple

from tnp import TransformerNeuralProcess
from gp_sampler import sample_gp_functions


def train_tnp(
    model: Optional[TransformerNeuralProcess] = None,
    x_dim: int = 1,
    y_dim: int = 1,
    dim_model: int = 128,
    num_heads: int = 4,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    num_iterations: int = 10000,
    batch_size: int = 16,
    num_context_range: Tuple[int, int] = (3, 50),
    num_total_points: int = 100,
    learning_rate: float = 1e-4,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_variance: float = 1.0,
    noise_variance: float = 0.01,
    print_freq: int = 1000,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> list:
    """
    Train the Transformer Neural Process model on GP samples.
    
    Args:
        model: TNP model to train (optional; initialized from hyperparameters when None)
        x_dim: Input dimension for a new model
        y_dim: Output dimension for a new model
        dim_model: Model hidden size for a new model
        num_heads: Number of attention heads for a new model
        num_encoder_layers: Number of encoder layers for a new model
        dim_feedforward: Feedforward dimension for a new model
        dropout: Dropout rate for a new model
        num_iterations: Number of training iterations
        batch_size: Batch size for training
        num_context_range: Range of context points (min, max)
        num_total_points: Total number of points per function sample
        learning_rate: Learning rate for optimizer
        x_range: Range of x values for GP sampling
        kernel_length_scale: Length scale of GP kernel
        kernel_variance: Variance of GP kernel
        noise_variance: Noise variance for GP
        print_freq: Print loss every N iterations
        device: Device to use for training
        model_path: Optional path to load initial weights
        save_path: Optional path to save trained weights
        
    Returns:
        List of losses during training
    """
    if model is None:
        model = TransformerNeuralProcess(
            x_dim=x_dim,
            y_dim=y_dim,
            dim_model=dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for iteration in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        
        # Sample functions from GP
        x, y = sample_gp_functions(
            num_samples=batch_size,
            num_points=num_total_points,
            x_range=x_range,
            kernel_length_scale=kernel_length_scale,
            kernel_variance=kernel_variance,
            noise_variance=noise_variance,
            x_dim=model.x_dim,
            y_dim=model.y_dim
        )
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        
        # Randomly select number of context points
        num_context = np.random.randint(num_context_range[0], num_context_range[1] + 1)
        
        # Split into context and target
        indices = torch.randperm(num_total_points, device=device)
        context_indices = indices[:num_context]
        target_indices = indices[num_context:]
        
        context_x = x[:, context_indices, :]
        context_y = y[:, context_indices, :]
        target_x = x[:, target_indices, :]
        target_y = y[:, target_indices, :]
        
        # Forward pass
        pred_mean, pred_std = model(context_x, context_y, target_x)
        
        # Compute negative log-likelihood loss
        loss = 0.5 * (
            torch.log(pred_std ** 2) +
            (target_y - pred_mean) ** 2 / (pred_std ** 2)
        ).mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        
        # Print progress
        if (iteration + 1) % print_freq == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss_value:.4f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    
    return losses
