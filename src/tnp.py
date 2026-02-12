import torch
import torch.nn as nn
from typing import Tuple, Optional


class TransformerNeuralProcess(nn.Module):
    """
    A simple Transformer Neural Process model.
    
    The model processes context points (xc, yc) and target points (xt) to predict
    mean and std for yt using a masked transformer encoder.
    """
    
    def __init__(
        self,
        x_dim: int = 1,
        y_dim: int = 1,
        dim_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim_model = dim_model
        
        # Embedder: MLP that maps (x, y) pairs to embeddings
        self.embedder = nn.Sequential(
            nn.Linear(x_dim + y_dim, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, dim_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Predictor: MLP that maps encodings to mean and std
        self.predictor = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, 2)  # Output: mean and std
        )
        
    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TNP.
        
        Args:
            context_x: Context input locations [batch_size, num_context, x_dim]
            context_y: Context output values [batch_size, num_context, y_dim]
            target_x: Target input locations [batch_size, num_target, x_dim]
            
        Returns:
            mean: Predicted mean [batch_size, num_target, y_dim]
            std: Predicted std [batch_size, num_target, y_dim]
        """
        batch_size = context_x.shape[0]
        num_context = context_x.shape[1]
        num_target = target_x.shape[1]
        
        # Concatenate context points (xc, yc)
        context_input = torch.cat([context_x, context_y], dim=-1)  # [B, nc, x_dim+y_dim]
        
        # Concatenate target points (xt, zeros)
        target_zeros = torch.zeros(batch_size, num_target, self.y_dim, device=target_x.device)
        target_input = torch.cat([target_x, target_zeros], dim=-1)  # [B, nt, x_dim+y_dim]
        
        # Concatenate context and target
        full_input = torch.cat([context_input, target_input], dim=1)  # [B, nc+nt, x_dim+y_dim]
        
        # Step 1: Embed all points
        embeddings = self.embedder(full_input)  # [B, nc+nt, dim_model]
        
        # Step 2: Create attention mask
        # Each position attends to all context points but ignores target points
        mask = torch.zeros(num_context + num_target, num_context + num_target, device=target_x.device)
        # Block attention to target points (columns nc:nc+nt)
        mask[:, num_context:] = float('-inf')
        
        # Encode with masked transformer
        encodings = self.encoder(embeddings, mask=mask)  # [B, nc+nt, dim_model]
        
        # Step 3: Extract target encodings and predict
        target_encodings = encodings[:, num_context:, :]  # [B, nt, dim_model]
        predictions = self.predictor(target_encodings)  # [B, nt, 2]
        
        # Split into mean and std
        mean = predictions[:, :, 0:1]  # [B, nt, 1]
        std = predictions[:, :, 1:2]  # [B, nt, 1]
        
        # Ensure std is positive
        std = torch.nn.functional.softplus(std) + 1e-6
        
        return mean, std


def initialize_tnp(
    x_dim: int = 1,
    y_dim: int = 1,
    dim_model: int = 128,
    num_heads: int = 4,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    device: Optional[str] = None
) -> TransformerNeuralProcess:
    """
    Initialize a Transformer Neural Process model.
    
    Args:
        x_dim: Dimension of input x
        y_dim: Dimension of output y
        dim_model: Hidden dimension of transformer
        num_heads: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        device: Device to place model on ('mps', 'cuda', or 'cpu')
        
    Returns:
        Initialized TNP model
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    model = TransformerNeuralProcess(
        x_dim=x_dim,
        y_dim=y_dim,
        dim_model=dim_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    model = model.to(device)
    return model
