"""Dense Autoencoder: 20 -> 32 -> 16 -> 8 -> 16 -> 32 -> 20 (ReLU, sigmoid output)."""
import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 20, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """Per-flow mean squared reconstruction error (the anomaly score)."""
        return torch.mean((x - self.forward(x)) ** 2, dim=1)

    def per_feature_error(self, x):
        """Per-feature squared error (x_i - x_hat_i)^2: the attribution."""
        return (x - self.forward(x)) ** 2
