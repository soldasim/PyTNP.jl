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


def train_tnp(
    model: TransformerNeuralProcess,
    sample_batch: Callable[[], Tuple[object, object, object, object]],
    num_iterations: int = 10000,
    learning_rate: float = 1e-4,
    print_freq: int = 1000,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> list:
    """
    Train the Transformer Neural Process model on sampled data.
    
    Args:
        model: TNP model to train
        sample_batch: Callable that returns (context_x, context_y, target_x, target_y)
        num_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        print_freq: Print loss every N iterations
        device: Device to use for training
        model_path: Optional path to load initial weights
        save_path: Optional path to save trained weights
        
    Returns:
        List of losses during training
    """
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
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

    model.eval()

    if save_path is not None:
        save_model(model, save_path)
    
    return losses
