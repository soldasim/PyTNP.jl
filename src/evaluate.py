import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional, List

from tnp import TransformerNeuralProcess
from gp_sampler import sample_gp_functions, PriorSpec


def _plot_1d(
    num_test_functions: int,
    context_x_list: List[np.ndarray],
    context_y_list: List[np.ndarray],
    target_x_list: List[np.ndarray],
    target_y_list: List[np.ndarray],
    pred_mean_list: List[np.ndarray],
    pred_std_list: List[np.ndarray],
    save_path: Optional[str] = None
):
    """Plot predictions for 1D input functions."""
    fig, axes = plt.subplots(1, num_test_functions, figsize=(15, 4))
    if num_test_functions == 1:
        axes = [axes]
    
    for i in range(num_test_functions):
        context_x_np = context_x_list[i][:, 0]
        context_y_np = context_y_list[i][:, 0]
        target_x_np = target_x_list[i][:, 0]
        target_y_np = target_y_list[i][:, 0]
        pred_mean_np = pred_mean_list[i][:, 0]
        pred_std_np = pred_std_list[i][:, 0]
        
        # Sort by x for plotting
        sort_idx = np.argsort(target_x_np)
        target_x_sorted = target_x_np[sort_idx]
        target_y_sorted = target_y_np[sort_idx]
        pred_mean_sorted = pred_mean_np[sort_idx]
        pred_std_sorted = pred_std_np[sort_idx]
        
        ax = axes[i]
        
        # Plot ground truth
        ax.plot(target_x_sorted, target_y_sorted, 'k-', label='True function', linewidth=2, alpha=0.7)
        
        # Plot context points
        ax.scatter(context_x_np, context_y_np, c='blue', s=100, marker='o', 
                  label='Context points', zorder=5, edgecolors='black', linewidths=1.5)
        
        # Plot predictions with uncertainty
        ax.plot(target_x_sorted, pred_mean_sorted, 'r-', label='Prediction', linewidth=2)
        ax.fill_between(target_x_sorted, 
                       pred_mean_sorted - 2*pred_std_sorted, 
                       pred_mean_sorted + 2*pred_std_sorted, 
                       color='red', alpha=0.2, label='±2σ uncertainty')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Test Function {i+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def _plot_2d(
    num_test_functions: int,
    context_x_list: List[np.ndarray],
    context_y_list: List[np.ndarray],
    target_x_list: List[np.ndarray],
    target_y_list: List[np.ndarray],
    pred_mean_list: List[np.ndarray],
    pred_std_list: List[np.ndarray],
    save_path: Optional[str] = None
):
    """Plot predictions for 2D input functions using scatter plots."""
    fig = plt.figure(figsize=(15, 4 * num_test_functions))
    
    for i in range(num_test_functions):
        context_x_np = context_x_list[i]  # Shape: [num_context, 2]
        context_y_np = context_y_list[i][:, 0]  # Shape: [num_context]
        target_x_np = target_x_list[i]  # Shape: [num_target, 2]
        target_y_np = target_y_list[i][:, 0]  # Shape: [num_target]
        pred_mean_np = pred_mean_list[i][:, 0]  # Shape: [num_target]
        pred_std_np = pred_std_list[i][:, 0]  # Shape: [num_target]
        
        # Create 3 subplots per function: true, predicted, uncertainty
        ax1 = fig.add_subplot(num_test_functions, 3, 3*i + 1, projection='3d')
        ax2 = fig.add_subplot(num_test_functions, 3, 3*i + 2, projection='3d')
        ax3 = fig.add_subplot(num_test_functions, 3, 3*i + 3, projection='3d')
        
        # Plot 1: Ground truth
        scatter1 = ax1.scatter(target_x_np[:, 0], target_x_np[:, 1], target_y_np, 
                               c=target_y_np, cmap='viridis', s=20, alpha=0.6)
        ax1.scatter(context_x_np[:, 0], context_x_np[:, 1], context_y_np,
                    c='red', s=100, marker='o', edgecolors='black', linewidths=1.5,
                    label='Context', zorder=10)
        ax1.set_xlabel('x₁', fontsize=10)
        ax1.set_ylabel('x₂', fontsize=10)
        ax1.set_zlabel('y', fontsize=10)
        ax1.set_title(f'Function {i+1}: Ground Truth', fontsize=11, fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=10)
        
        # Plot 2: Predicted mean
        scatter2 = ax2.scatter(target_x_np[:, 0], target_x_np[:, 1], pred_mean_np, 
                               c=pred_mean_np, cmap='viridis', s=20, alpha=0.6)
        ax2.scatter(context_x_np[:, 0], context_x_np[:, 1], context_y_np,
                    c='red', s=100, marker='o', edgecolors='black', linewidths=1.5,
                    label='Context', zorder=10)
        ax2.set_xlabel('x₁', fontsize=10)
        ax2.set_ylabel('x₂', fontsize=10)
        ax2.set_zlabel('y', fontsize=10)
        ax2.set_title(f'Function {i+1}: Prediction', fontsize=11, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=10)
        
        # Plot 3: Uncertainty (std)
        scatter3 = ax3.scatter(target_x_np[:, 0], target_x_np[:, 1], pred_std_np, 
                               c=pred_std_np, cmap='plasma', s=20, alpha=0.6)
        ax3.scatter(context_x_np[:, 0], context_x_np[:, 1], np.zeros_like(context_y_np),
                    c='blue', s=100, marker='o', edgecolors='black', linewidths=1.5,
                    label='Context (z=0)', zorder=10, alpha=0.7)
        ax3.set_xlabel('x₁', fontsize=10)
        ax3.set_ylabel('x₂', fontsize=10)
        ax3.set_zlabel('σ', fontsize=10)
        ax3.set_title(f'Function {i+1}: Uncertainty', fontsize=11, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def evaluate_and_plot(
    model: TransformerNeuralProcess,
    device: str,
    num_test_functions: int = 3,
    num_test_points: int = 200,
    num_context_points: int = 10,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_std: float = 1.0,
    noise_std: float = 0.1,
    kernel_length_scale_prior: PriorSpec = None,
    kernel_std_prior: PriorSpec = None,
    save_path: Optional[str] = 'tnp_predictions.png'
):
    """
    Evaluate the TNP model on sampled GP functions and plot predictions.
    
    Args:
        model: Trained TNP model
        device: Device to use for inference
        num_test_functions: Number of test functions to sample
        num_test_points: Total points per test function
        num_context_points: Number of context points to use
        x_range: Range for x values
        kernel_length_scale: GP kernel length scale
        kernel_std: GP kernel std dev
        noise_std: GP noise std dev
        kernel_length_scale_prior: Prior for length scale (low, high)
        kernel_std_prior: Prior for kernel std dev (low, high)
        save_path: Path to save the plot (None to skip saving)
    """
    print("\n" + "="*60)
    print("Testing the trained model...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        x_dim = model.x_dim
        y_dim = model.y_dim

        if y_dim != 1:
            raise ValueError("evaluate_and_plot currently supports y_dim=1 for plotting.")

        # Sample noise-free test functions from GP
        x_test, y_true = sample_gp_functions(
            num_samples=num_test_functions,
            num_points=num_test_points,
            x_range=x_range,
            kernel_length_scale=kernel_length_scale,
            kernel_std=kernel_std,
            noise_std=0.0,
            kernel_length_scale_prior=kernel_length_scale_prior,
            kernel_std_prior=kernel_std_prior,
            x_dim=x_dim,
            y_dim=y_dim
        )

        # Add observation noise for context/target values
        if noise_std > 0:
            y_obs = y_true + np.random.normal(
                scale=noise_std,
                size=y_true.shape
            )
        else:
            y_obs = y_true
        
        # Convert to tensors
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_obs = torch.tensor(y_obs, dtype=torch.float32, device=device)
        y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
        
        # Prepare data for each test function
        context_x_list = []
        context_y_list = []
        target_x_list = []
        target_y_list = []
        pred_mean_list = []
        pred_std_list = []
        
        for i in range(num_test_functions):
            context_x = x_test[i:i+1, :num_context_points, :]
            context_y = y_obs[i:i+1, :num_context_points, :]
            target_x = x_test[i:i+1, num_context_points:, :]
            target_y_true = y_true[i:i+1, num_context_points:, :]
            
            # Get predictions
            pred_mean, pred_std = model(context_x, context_y, target_x)
            
            # Move to CPU for plotting
            context_x_list.append(context_x[0].cpu().numpy())
            context_y_list.append(context_y[0].cpu().numpy())
            target_x_list.append(target_x[0].cpu().numpy())
            target_y_list.append(target_y_true[0].cpu().numpy())
            pred_mean_list.append(pred_mean[0].cpu().numpy())
            pred_std_list.append(pred_std[0].cpu().numpy())
        
        # Plot based on input dimension
        if x_dim == 1:
            _plot_1d(
                num_test_functions,
                context_x_list,
                context_y_list,
                target_x_list,
                target_y_list,
                pred_mean_list,
                pred_std_list,
                save_path
            )
        elif x_dim == 2:
            _plot_2d(
                num_test_functions,
                context_x_list,
                context_y_list,
                target_x_list,
                target_y_list,
                pred_mean_list,
                pred_std_list,
                save_path
            )
        else:
            print(f"Warning: plotting not implemented for x_dim={x_dim}. Skipping visualization.")
