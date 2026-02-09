from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_circular_weights(
    attn_weights: torch.Tensor,
    N: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Gathers circularly shifted weights based on attention scores.

    Args:
        attn_weights (torch.Tensor): Attention weights (B, h, N).
        N (int): Query sequence length.
        seq_len (int): Key/Value sequence length (M).
        num_heads (int): Number of attention heads.
        device (torch.device): Device for tensor creation.

    Returns:
        torch.Tensor: Circularly shifted weights (roll_weights) in (B, h, N, M) format.
                      roll_weights[b, h, i, j] = attn_weights[b, h, (j-i)%N]
    """
    B: int = attn_weights.shape[0]

    # Ensure attn_weights is strictly (B, h, N)
    if not (
        attn_weights.dim() == 3
        and attn_weights.shape[1] == num_heads
        and attn_weights.shape[2] == N
    ):
        raise ValueError(
            f"Unexpected attn_weights shape: {attn_weights.shape}. Expected (B, h, N), where h is num_heads."
        )

    # Create circular shift indices: indices[i, j] = (j - i) % N
    # Modulo N because weights depend on the query index i
    col_indices = torch.arange(seq_len, device=device)  # 0 to M-1
    row_indices = torch.arange(N, device=device)  # 0 to N-1
    indices = (col_indices.unsqueeze(0) - row_indices.unsqueeze(1)) % N  # N x M

    # Expand indices for batch and head dimensions
    indices = indices.view(1, 1, N, seq_len).expand(
        B, num_heads, N, seq_len
    )  # B, h, N, M

    # Gather weights from attn_weights (B, h, N) based on shifted indices
    roll_weights = torch.gather(
        attn_weights.unsqueeze(3).expand(B, num_heads, N, seq_len),  # B, h, N, M
        dim=2,  # Gather along the N dimension (query dimension)
        index=indices,
    )  # B, h, N, M

    roll_weights /= N  # Normalize weights by N

    return roll_weights


class CircularConvolutionalAttention(nn.Module):
    """
    Implementation of Circular-Convolutional Attention (CAT)
    This implements the gather-based version mentioned in the paper.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # In CAT, we reduce parameter count by using a single projection W_A instead of W_Q and W_K
        self.W_A = nn.Linear(dim, num_heads, bias=qkv_bias)  # Maps to z ∈ R^N×h
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)  # Value projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Project input to attention scores and values
        z = self.W_A(x)  # B, N, h (where h is number of heads)
        z = z.permute(0, 2, 1)  # B, N, h -> B, h, N

        # Apply softmax along sequence dimension
        z_star = F.softmax(z, dim=-1)  # B, h, N
        z_star = self.attn_drop(z_star)  # Apply dropout

        # Gather circularly shifted weights
        roll_weights = _gather_circular_weights(
            attn_weights=z_star,
            N=N,
            seq_len=N,  # Self-attention, seq_len is N
            num_heads=self.num_heads,
            device=x.device,
        )  # B, h, N, N

        # Project input to values
        V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # B, N, h, C/h -> B, h, N, C/h

        # Apply weights to values (B, h, N, N) @ (B, h, N, C/h) -> B, h, N, C/h
        output = torch.matmul(roll_weights, V)

        # Reshape and project output
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)  # B, N, C
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class AveragedKeyCircularConvolutionalAttention(nn.Module):
    """
    Implementation of the Averaged-Key Alternative mentioned in the paper
    This variant preserves separate W_Q and W_K projections, enabling cross-attention support.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Keep separate query, key, value projections
        self.W_Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_K = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C)
            context: Optional context tensor for cross-attention
                     If None, performs self-attention
        """
        B, N, C = x.shape

        # Project queries from input
        Q = self.W_Q(x).reshape(B, N, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # B, h, N, C/h

        # Handle cross-attention if context is provided
        if context is not None:
            assert context.dim() == 3, "Context must have 3 dimensions (B, M, C)"
            B_ctx, M, C_ctx = context.shape
            assert B == B_ctx and C == C_ctx, "Batch size and feature dim must match"

            K = self.W_K(context).reshape(B, M, self.num_heads, self.head_dim)
            K = K.permute(0, 2, 1, 3)  # B, h, M, C/h

            V = self.W_V(context).reshape(B, M, self.num_heads, self.head_dim)
            V = V.permute(0, 2, 1, 3)  # B, h, M, C/h
        else:
            # Self-attention case
            K = self.W_K(x).reshape(B, N, self.num_heads, self.head_dim)
            K = K.permute(0, 2, 1, 3)  # B, h, N, C/h

            V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
            V = V.permute(0, 2, 1, 3)  # B, h, N, C/h

        # Average the keys - equation from Section 4.2
        K_avg = K.mean(dim=2, keepdim=True)  # B, h, 1, C/h

        # Compute attention scores with averaged key
        z_circ = torch.matmul(Q, K_avg.transpose(-1, -2)) * self.scale  # B, h, N, 1
        z_circ = z_circ.squeeze(-1)  # B, h, N

        # Apply softmax
        attn = F.softmax(z_circ, dim=-1)  # B, h, N
        attn = self.attn_drop(attn)  # B, h, N

        # Determine key/value sequence length
        seq_len = V.size(2)  # M for cross-attention, N for self-attention

        # Gather circularly shifted weights
        # attn is already (B, h, N)
        roll_weights = _gather_circular_weights(
            attn_weights=attn,
            N=N,
            seq_len=seq_len,  # Use M or N depending on context
            num_heads=self.num_heads,
            device=x.device,
        )  # B, h, N, M

        # Apply weights to values (B, h, N, M) @ (B, h, M, C/h) -> B, h, N, C/h
        output = torch.matmul(roll_weights, V)

        # Reshape and project output
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)  # B, N, C
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class CausalCircularConvolutionalAttention(nn.Module):
    """
    Implementation of CAT for causal language modeling
    This ensures future tokens don't leak into current positions,
    as mentioned in the paper's WikiText-103 experiments.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.W_A = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Project input to attention scores and values
        z = self.W_A(x)  # B, N, h

        # Apply softmax along sequence dimension and permute to (B, h, N)
        z_star = F.softmax(z, dim=1).permute(0, 2, 1)  # B, N, h -> B, h, N
        z_star = self.attn_drop(z_star)  # Apply dropout

        # Project input to values
        V = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # B, h, N, C/h

        # Gather circularly shifted weights
        # z_star is already (B, h, N)
        roll_weights = _gather_circular_weights(
            attn_weights=z_star,
            N=N,
            seq_len=N,  # Causal attention implies self-attention, seq_len is N
            num_heads=self.num_heads,
            device=x.device,
        )  # B, h, N, N

        # Apply causal mask
        mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        mask = mask.view(1, 1, N, N).expand_as(roll_weights)  # Expand to B, h, N, N
        masked_roll_weights = roll_weights.masked_fill(~mask, 0.0)  # Apply mask

        # Apply weights to values (B, h, N, N) @ (B, h, N, C/h) -> B, h, N, C/h
        output = torch.matmul(masked_roll_weights, V)

        # Reshape and project output
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)  # B, N, C
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


# Demonstration of usage
if __name__ == "__main__":
    # Create sample input
    batch_size, seq_len, dim = 2, 32, 256
    x = torch.randn(batch_size, seq_len, dim)

    # Test CAT
    cat = CircularConvolutionalAttention(dim=dim)
    out_cat = cat(x)
    print(f"CAT output shape: {out_cat.shape}")

    # Test Averaged-Key Alternative (self-attention)
    avg_key = AveragedKeyCircularConvolutionalAttention(dim=dim)
    out_avg = avg_key(x)
    print(f"Averaged-Key output shape: {out_avg.shape}")

    # Test Averaged-Key Alternative (cross-attention)
    context = torch.randn(batch_size, seq_len * 2, dim)  # Different sequence length
    out_cross = avg_key(x, context)
    print(f"Averaged-Key cross-attention output shape: {out_cross.shape}")

    # Test Causal CAT
    causal_cat = CausalCircularConvolutionalAttention(dim=dim)
    out_causal = causal_cat(x)
    print(f"Causal CAT output shape: {out_causal.shape}")
