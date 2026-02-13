"""
Main training and evaluation script for the Transformer Neural Process.

This script trains a TNP model on Gaussian Process samples and evaluates it
by making predictions on test functions.
"""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weights import load_model, save_model
from gp_sampler import make_gp_sampler
from tnp import initialize_tnp
from train import train_tnp
from evaluate import evaluate_and_plot


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "data/tnp_model_py.pt"
    TRAIN_MODEL = not os.path.exists(MODEL_PATH)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    print("\nInitializing TNP model...")
    model = initialize_tnp(
        x_dim=1, # TODO try x_dim=2
        y_dim=1,
        dim_model=128,
        num_heads=4,
        encoder_depth=2,
        device=device
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    if TRAIN_MODEL:
        print("\nTraining TNP model...")
        sample_batch = make_gp_sampler(
            batch_size=32,
            num_total_points_range=(2, 101),
            x_range=(-2.0, 2.0),
            kernel_length_scale_prior=(0.1, 2.0),
            kernel_std_prior=(0.1, 1.0),
            noise_std=1e-8,
            x_dim=model.x_dim,
            y_dim=model.y_dim
        )

        _, losses, _ = train_tnp(
            model=model,
            sample_batch=sample_batch,
            num_iterations=2000, # TODO increase to 10_000
            print_freq=100,
            device=device
        )
        
        print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
        
        # Save model weights + hyperparameters
        save_model(model, MODEL_PATH)
        print(f"Model weights saved to {MODEL_PATH}")
    else:
        # Load model weights + hyperparameters
        model = load_model(MODEL_PATH, device=device)
        print(f"Model weights loaded from {MODEL_PATH}")
    
    # Test the model
    evaluate_and_plot(
        model=model,
        device=device,
        num_test_functions=3,
        num_test_points=200,
        num_context_points=10,
        x_range=(-2.0, 2.0),
        kernel_length_scale_prior=(0.1, 2.0),
        kernel_std_prior=(0.1, 1.0),
        noise_std=1e-8,
        save_path='data/tnp_python_predictions.png'
    )
