"""Variational Autoencoder (VAE) para detección de anomalías.

Entrada z = [x_hat, m_hat, contexto]. El VAE se entrena solo con motores
sanos; el error de reconstrucción alto indica anomalía.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden)
        self.fc_out = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruction_error(self, x):
        """Error de reconstrucción por muestra (media sobre features)."""
        self.eval()
        x_hat, _, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=-1)


def vae_loss(x, x_hat, mu, logvar, kld_weight=1.0):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld_weight * kld, recon, kld
