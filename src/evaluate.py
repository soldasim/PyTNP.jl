import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from tnp import TransformerNeuralProcess
from gp_sampler import sample_gp_functions


def evaluate_and_plot(
    model: TransformerNeuralProcess,
    device: str,
    num_test_functions: int = 3,
    num_test_points: int = 200,
    num_context_points: int = 10,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    kernel_length_scale: float = 0.4,
    kernel_variance: float = 1.0,
    noise_variance: float = 0.01,
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
        kernel_variance: GP kernel variance
        noise_variance: GP noise variance
        save_path: Path to save the plot (None to skip saving)
    """
    print("\n" + "="*60)
    print("Testing the trained model...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Sample noise-free test functions from GP
        x_test, y_true = sample_gp_functions(
            num_samples=num_test_functions,
            num_points=num_test_points,
            x_range=x_range,
            kernel_length_scale=kernel_length_scale,
            kernel_variance=kernel_variance,
            noise_variance=0.0,
            x_dim=1,
            y_dim=1
        )

        # Add observation noise for context/target values
        if noise_variance > 0:
            y_obs = y_true + np.random.normal(
                scale=np.sqrt(noise_variance),
                size=y_true.shape
            )
        else:
            y_obs = y_true
        
        # Convert to tensors
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_obs = torch.tensor(y_obs, dtype=torch.float32, device=device)
        y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
        
        # Create plots
        fig, axes = plt.subplots(1, num_test_functions, figsize=(15, 4))
        if num_test_functions == 1:
            axes = [axes]
        
        # For each test function, use first num_context_points as context
        # and predict on the remaining points
        for i in range(num_test_functions):
            context_x = x_test[i:i+1, :num_context_points, :]
            context_y = y_obs[i:i+1, :num_context_points, :]
            target_x = x_test[i:i+1, num_context_points:, :]
            target_y_true = y_true[i:i+1, num_context_points:, :]
            
            # Get predictions
            pred_mean, pred_std = model(context_x, context_y, target_x)
            
            # Move to CPU for plotting
            context_x_np = context_x[0, :, 0].cpu().numpy()
            context_y_np = context_y[0, :, 0].cpu().numpy()
            target_x_np = target_x[0, :, 0].cpu().numpy()
            target_y_np = target_y_true[0, :, 0].cpu().numpy()
            pred_mean_np = pred_mean[0, :, 0].cpu().numpy()
            pred_std_np = pred_std[0, :, 0].cpu().numpy()
            
            # Sort by x for plotting
            sort_idx = np.argsort(target_x_np)
            target_x_sorted = target_x_np[sort_idx]
            target_y_sorted = target_y_np[sort_idx]
            pred_mean_sorted = pred_mean_np[sort_idx]
            pred_std_sorted = pred_std_np[sort_idx]
            
            # Plot
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
