import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CustomAttention(nn.Module):
    """
    Multi-head attention with softmax1 (softmax with 1 added to denominator).
    Computes: softmax1(x) = exp(x) / (1 + sum(exp(x)))
    
    Uses scaled_dot_product_attention with custom softmax1.
    """
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers for Q, K, V projections and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key):
        """
        Forward pass for multi-head attention with softmax1.
        
        Args:
            query: [batch_size, n_q, embed_dim]
            key: [batch_size, n_k, embed_dim]
        
        Returns:
            a0: [batch_size, n_q]
            attn_weights: [batch_size, n_q, n_k]
        """
        batch_size = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [batch, tgt_len, embed_dim]
        K = self.k_proj(key)    # [batch, src_len, embed_dim]
        
        # Reshape for multi-head: [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores using scaled_dot_product_attention pattern
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, num_heads, tgt_len, src_len]
        
        # Apply softmax1 with numerical stability: exp(x) / (1 + sum(exp(x)))
        # Subtract max for numerical stability (log-sum-exp trick)
        scores_max = scores.amax(dim=-1, keepdim=True)  # [batch, num_heads, tgt_len, 1]
        exp_scores = torch.exp(scores - scores_max)  # [batch, num_heads, tgt_len, src_len]
        exp_prior = torch.exp(0. - scores_max)  # [batch, num_heads, tgt_len, 1]

        Z = exp_prior + exp_scores.sum(dim=-1, keepdim=True)  # [batch, num_heads, tgt_len, 1]
        attn_weights = exp_scores / Z
        a0 = exp_prior / Z
        
        # Average attention weights across heads
        attn_weights = attn_weights.mean(dim=1)  # [batch, tgt_len, src_len]
        a0 = a0.mean(dim=1).squeeze(-1)  # [batch, tgt_len]
        
        return a0, attn_weights


class StructuredTNP(nn.Module):
    """
    A Structured Transformer Neural Process model with additional modifications to the predictor.
    """
    
    def __init__(
        self,
        x_dim: int = 1,
        y_dim: int = 1,
        dim_model: int = 64,
        embedder_depth: int = 4,
        predictor_depth: int = 2,
        num_heads: int = 8,
        encoder_depth: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.0
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
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
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
        
        # Cross-attention: target encodings attend to context encodings (with softmax1)
        self.cross_attention = CustomAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Predictor: MLP that maps encodings to log_std
        DIM_IN = dim_model + 1 + 1 # encoding + rho + a0
        DIM_OUT = 1 # log_std only
        assert predictor_depth >= 2
        
        predictor_layers = []
        predictor_layers.append(nn.Linear(DIM_IN, dim_model))
        predictor_layers.append(nn.ReLU())
        for _ in range(predictor_depth - 2):
            predictor_layers.append(nn.Linear(dim_model, dim_model))
            predictor_layers.append(nn.ReLU())
        predictor_layers.append(nn.Linear(dim_model, DIM_OUT))
        self.predictor = nn.Sequential(*predictor_layers)
        
    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        return_params: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TNP with attention-weighted mean prediction.
        
        The mean is computed as a weighted average of context_y (using cross-attention).
        
        Args:
            context_x: Context input locations [batch_size, num_context, x_dim]
            context_y: Context output values [batch_size, num_context, y_dim]
            target_x: Target input locations [batch_size, num_target, x_dim]
            
        Returns:
            mean: Predicted mean = A*context_y [batch_size, num_target, y_dim]
            log_std: Predicted log(std) [batch_size, num_target, y_dim]
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
        
        # Step 3: Extract context and target encodings
        context_encodings = encodings[:, :num_context, :]  # [B, nc, dim_model]
        target_encodings = encodings[:, num_context:, :]  # [B, nt, dim_model]
        
        # Step 4: Apply cross-attention to get attention weights
        # Query: target encodings, Key/Value: context encodings
        a0, attn_weights = self.cross_attention(
            target_encodings, 
            context_encodings, 
        )  # attn_weights: [B, nt, nc]
        # rho: sum of squares of attention weights along context dim
        rho = (a0 ** 2) + (attn_weights ** 2).sum(dim=-1)  # [B, nt]
        
        # Step 5: Compute weighted average of context_y
        # attn_weights: [B, nt, nc], context_y: [B, nc, y_dim]
        mean = torch.bmm(attn_weights, context_y)  # [B, nt, y_dim]
        
        # Step 6: MLP for mean residuals & log_std
        # Prepare input for predictor: concat target_encodings, rho, a0
        predictor_input = torch.cat([
            target_encodings,   # target_encodings: [B, nt, dim_model]
            rho.unsqueeze(-1),  # rho: [B, nt]
            a0.unsqueeze(-1)    # a0: [B, nt]
        ], dim=-1)  # [B, nt, dim_model + y_dim + 1 + 1]

        # mean_residual, log_std = self.predictor(predictor_input).split(1, dim=-1)  # [B, nt, 1] each
        log_std = self.predictor(predictor_input)  # [B, nt, 1]
        
        if return_params:
            sum_weights = attn_weights.sum(dim=-1)  # [B, nt]
            return mean, log_std, attn_weights, sum_weights
        else:
            return mean, log_std


def initialize_structured_tnp(
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
) -> StructuredTNP:
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
    
    model = StructuredTNP(
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
