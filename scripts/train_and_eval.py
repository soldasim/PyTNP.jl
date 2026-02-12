"""
Main training and evaluation script for the Transformer Neural Process.

This script trains a TNP model on Gaussian Process samples and evaluates it
by making predictions on test functions.
"""

import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnp import initialize_tnp
from train import train_tnp
from evaluate import evaluate_and_plot


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "data/tnp_model.pt"
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
        x_dim=1,
        y_dim=1,
        dim_model=128,
        num_heads=4,
        num_encoder_layers=2,
        device=device
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    if TRAIN_MODEL:
        print("\nTraining TNP model...")
        losses = train_tnp(
            model=model,
            num_iterations=5000,
            batch_size=16,
            num_context_range=(3, 20),
            num_total_points=50,
            learning_rate=1e-4,
            print_freq=500,
            device=device
        )
        
        print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
        
        # Save model weights
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model weights saved to {MODEL_PATH}")
    else:
        # Load model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        print(f"Model weights loaded from {MODEL_PATH}")
    
    # Test the model
    evaluate_and_plot(
        model=model,
        device=device,
        num_test_functions=3,
        num_test_points=200,
        num_context_points=10,
        x_range=(-2.0, 2.0),
        kernel_length_scale=0.4,
        kernel_variance=1.0,
        noise_variance=0.01,
        save_path='data/tnp_predictions.png'
    )
