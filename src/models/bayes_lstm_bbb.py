"""Bayesian LSTM con *Bayes by Backprop* (BBB).

Implementación fiel a Blundell et al. (2015), "Weight Uncertainty in Neural
Networks", aplicada a una celda LSTM — el enfoque del paper de referencia
(Chen et al., 2026).

Idea
----
En lugar de pesos deterministas, cada peso es una variable aleatoria con
distribución variacional q(W) = N(mu, sigma^2), con sigma = softplus(rho).
En cada forward se *muestrea* W mediante el truco de reparametrización:
    W = mu + softplus(rho) * eps,   eps ~ N(0, 1)
El entrenamiento minimiza la energía libre variacional:
    L = E_q[ -log p(D|W) ]  +  KL[ q(W) || p(W) ]
donde el primer término es el error de predicción (MSE/NLL) y el segundo
regulariza hacia el prior p(W) = N(0, prior_sigma^2).

En inferencia se hacen varias pasadas (muestreando pesos) y se reporta
media (predicción) y varianza (incertidumbre epistémica).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class BayesianLinear(nn.Module):
    """Capa lineal con pesos variacionales (Bayes by Backprop)."""

    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Parámetros variacionales de los pesos y sesgos.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.kl = 0.0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)   # sigma inicial pequeña
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _kl(mu, sigma, prior_sigma):
        # KL[N(mu,sigma^2) || N(0,prior_sigma^2)] sumada sobre todos los pesos.
        return torch.sum(
            torch.log(torch.tensor(prior_sigma)) - torch.log(sigma)
            + (sigma ** 2 + mu ** 2) / (2 * prior_sigma ** 2) - 0.5
        )

    def forward(self, x):
        w_sigma = Fnn.softplus(self.weight_rho)
        b_sigma = Fnn.softplus(self.bias_rho)
        weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        # Acumula KL para esta pasada.
        self.kl = (self._kl(self.weight_mu, w_sigma, self.prior_sigma)
                   + self._kl(self.bias_mu, b_sigma, self.prior_sigma))
        return Fnn.linear(x, weight, bias)


class BayesianLSTMCell(nn.Module):
    """Celda LSTM cuyas transformaciones lineales son bayesianas."""

    def __init__(self, input_dim, hidden_dim, prior_sigma=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x2h = BayesianLinear(input_dim, 4 * hidden_dim, prior_sigma)
        self.h2h = BayesianLinear(hidden_dim, 4 * hidden_dim, prior_sigma)

    def forward(self, x, state):
        h, c = state
        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    @property
    def kl(self):
        return self.x2h.kl + self.h2h.kl


class BayesianLSTM_BBB(nn.Module):
    """LSTM bayesiano (una capa recurrente) + cabezal bayesiano de salida."""

    def __init__(self, input_dim, hidden_dim, output_dim, prior_sigma=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = BayesianLSTMCell(input_dim, hidden_dim, prior_sigma)
        self.head = BayesianLinear(hidden_dim, output_dim, prior_sigma)

    def forward(self, x):
        # x: (B, T, F)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        c = x.new_zeros(B, self.hidden_dim)
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
        y = self.head(h)
        return y

    def kl_loss(self):
        return self.cell.kl + self.head.kl

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_samples=30):
        """Media y varianza sobre n_samples muestreos de pesos."""
        self.eval()
        preds = [self.forward(x).cpu().numpy() for _ in range(n_samples)]
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0), preds.var(axis=0)


def elbo_loss(y_pred, y_true, kl, kl_weight):
    """Energía libre variacional = MSE (≈ -log p(D|W)) + kl_weight * KL."""
    nll = Fnn.mse_loss(y_pred, y_true, reduction="mean")
    return nll + kl_weight * kl, nll
