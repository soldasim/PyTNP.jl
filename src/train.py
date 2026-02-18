import torch
from typing import Callable, Optional, Tuple

from weights import load_state_dict, save_model
from tnp import TransformerNeuralProcess


def _to_tensor(value: object, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.device.type != device:
            tensor = tensor.to(device)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor
    return torch.tensor(value, dtype=torch.float32, device=device)


def _gaussian_nll_loss(pred_mean: torch.Tensor, pred_logstd: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian negative log-likelihood loss directly from log-standard deviation.
    
    Args:
        pred_mean: Predicted mean [batch_size, ..., num_outputs]
        target: Target values [batch_size, ..., num_outputs]
        pred_logstd: Predicted log-standard deviation [batch_size, ..., num_outputs]
    
    Returns:
        Loss (scalar, mean over all elements)
    """
    # Gaussian NLL: 0.5 * log(2*pi) + 0.5 * log_var + 0.5 * (target - mean)^2 / exp(log_var)
    # The last term can be written as: 0.5 * (target - mean)^2 * exp(-log_var)
    diff_squared = (target - pred_mean) ** 2
    nll = 0.5 * ((2 * pred_logstd) + (diff_squared * torch.exp(-2 * pred_logstd)))
    return nll.mean()


def train_tnp(
    model: TransformerNeuralProcess,
    sample_batch: Callable[[], Tuple[object, object, object, object]],
    num_iterations: int = 10_000,
    lr_start: float = 5e-4,
    lr_end: float = 0.0,
    warmup_ratio: float = 0.05,
    start_factor: float = 0.05,
    rolling_window: int = 100,
    print_freq: int = 500,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[list, list, list]:
    """
    Train the Transformer Neural Process model on sampled data.
    
    Args:
        model: TNP model to train
        sample_batch: Callable that returns (context_x, context_y, target_x, target_y)
        num_iterations: Number of training iterations
        lr_start: Starting learning rate
        lr_end: Ending learning rate for cosine annealing
        warmup_ratio: Fraction of total iterations to use for linear warmup (0.0-1.0)
        start_factor: Initial learning rate factor during warmup (fraction of lr_start)
        rolling_window: Window size for computing rolling average of loss
        print_freq: Print loss every N iterations
        device: Device to use for training
        model_path: Optional[str] = None,
        save_path: Optional[str] = None
        
    Returns:
        Tuple of (losses, learning_rates, rolling_avg_losses) lists during training
    """
    print("model_path: ", model_path) # TODO rem
    
    if model_path is not None:
        load_state_dict(model, model_path, strict=True)

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start)
    
    # Calculate warmup iterations based on ratio
    warmup_iter = int(warmup_ratio * num_iterations)
    
    # Linear warmup followed by cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_iter
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_iterations - warmup_iter,
        eta_min=lr_end
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iter]
    )
    
    # Gaussian negative log-likelihood loss
    loss_fn = _gaussian_nll_loss
    
    losses = []
    learning_rates = []
    rolling_avg_losses = []
    
    for iteration in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        
        context_x, context_y, target_x, target_y = sample_batch()

        # Convert to tensors
        context_x = _to_tensor(context_x, device)
        context_y = _to_tensor(context_y, device)
        target_x = _to_tensor(target_x, device)
        target_y = _to_tensor(target_y, device)
        
        # Forward pass
        pred_mean, pred_log_std = model(context_x, context_y, target_x)
        
        # Compute negative log-likelihood loss
        loss = loss_fn(pred_mean, pred_log_std, target_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        
        # Compute rolling average
        window_start = max(0, len(losses) - rolling_window)
        rolling_avg = sum(losses[window_start:]) / (len(losses) - window_start)
        rolling_avg_losses.append(rolling_avg)
        
        # Print progress
        if (iteration + 1) % print_freq == 0:
            print(
                f"Iteration {iteration + 1}/{num_iterations}, "
                f"Loss: {loss_value:.4f}, Rolling Avg: {rolling_avg:.4f}, LR: {current_lr:.6g}"
            )

    model.eval()

    if save_path is not None:
        save_model(model, save_path)
    
    return learning_rates, losses, rolling_avg_losses
