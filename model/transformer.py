import torch
import torch.nn as nn


class TransformerForwardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_dim: int = 2048,
        use_positional_encoding: bool = True,
    ):
        """
        Transformer-based forward model for AEM surrogate simulation.

        Args:
            input_dim (int): Number of geometric features (e.g. 14)
            output_dim (int): Number of spectral output points (e.g. 2001)
            hidden_dim (int): Transformer model dimension (e.g. 512)
            num_layers (int): Number of Transformer encoder layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            ff_dim (int): Hidden size in feedforward network (defaults to 2048)
            use_positional_encoding (bool): Whether to add learnable positional encoding
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Optional positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.register_parameter("positional_encoding", None)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to spectrum
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            Tensor: Output predictions [batch_size, output_dim]
        """
        x = self.input_proj(x).unsqueeze(1)  # Shape: [B, 1, H]
        if self.use_positional_encoding:
            x = x + self.positional_encoding  # Broadcasting
        x = self.transformer(x).squeeze(1)  # Shape: [B, H]
        return self.output_proj(x)  # Shape: [B, output_dim]
