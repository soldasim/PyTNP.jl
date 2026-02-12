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
        embedder_depth: int = 2,
        predictor_depth: int = 2,
        num_heads: int = 4,
        encoder_depth: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input validation
        if embedder_depth < 1:
            raise ValueError("embedder_depth must be >= 1")
        if predictor_depth < 1:
            raise ValueError("predictor_depth must be >= 1")

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dim_model = dim_model
        self.embedder_depth = embedder_depth
        self.predictor_depth = predictor_depth
        
        # Embedder: MLP that maps (x, y) pairs to embeddings
        embedder_layers = [nn.Linear(x_dim + y_dim, dim_model)]
        for _ in range(embedder_depth - 1):
            embedder_layers.append(nn.ReLU())
            embedder_layers.append(nn.Linear(dim_model, dim_model))
        self.embedder = nn.Sequential(*embedder_layers)
        
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
            num_layers=encoder_depth
        )
        
        # Predictor: MLP that maps encodings to mean and std
        predictor_layers = []
        for _ in range(predictor_depth - 1):
            predictor_layers.append(nn.Linear(dim_model, dim_model))
            predictor_layers.append(nn.ReLU())
        predictor_layers.append(nn.Linear(dim_model, 2))  # Output: mean and std
        self.predictor = nn.Sequential(*predictor_layers)
        
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
    embedder_depth: int = 2,
    predictor_depth: int = 2,
    num_heads: int = 4,
    encoder_depth: int = 2,
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
        embedder_depth: Number of layers in the embedder MLP
        predictor_depth: Number of layers in the predictor MLP
        num_heads: Number of attention heads
        encoder_depth: Number of transformer encoder layers
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
        embedder_depth=embedder_depth,
        predictor_depth=predictor_depth,
        num_heads=num_heads,
        encoder_depth=encoder_depth,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    model = model.to(device)
    model.eval()
    return model
