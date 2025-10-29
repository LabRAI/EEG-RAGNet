import torch
import torch.nn as nn

class SemanticQuery(nn.Module):
    """
    SemanticQuery module: projects EEG node features into the semantic embedding space
    aligned with the knowledge graph.
    
    This module is designed to learn a transformation from EEG spatial-temporal features
    (e.g., output from an encoder or STGNN layer) into a semantic space where each node
    embedding can be used as a query vector for retrieving related biomedical knowledge.
    """

    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, dropout: float = 0.1):
        """
        Initialize the semantic projection network.

        Args:
            input_dim (int): Input feature dimension (EEG node feature dimension)
            hidden_dim (int): Hidden layer dimension for intermediate representation
            proj_dim (int): Output projection dimension (semantic space dimension)
            dropout (float): Dropout rate for regularization
        """
        super(SemanticQuery, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.layer_norm = nn.LayerNorm(proj_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating semantic query embeddings.

        Args:
            X (torch.Tensor): Input EEG node features (batch_size, num_nodes, input_dim)

        Returns:
            torch.Tensor: Semantic query vectors (batch_size, num_nodes, proj_dim)
        """
        # Project each node feature vector into semantic space
        Q = self.projector(X)
        Q = self.layer_norm(Q)
        return Q


if __name__ == "__main__":
    # Example test run
    batch_size = 2
    num_nodes = 19   # EEG channels
    input_dim = 64   # Input feature dimension
    hidden_dim = 128
    proj_dim = 256

    X = torch.randn(batch_size, num_nodes, input_dim)

    model = SemanticQuery(input_dim, hidden_dim, proj_dim)
    Q = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Output (semantic query) shape: {Q.shape}")
