"""Modèle LSTM et perte asymétrique."""
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


def perte_asym(pred, cible, alpha: float):
    """MSE pondérée : l'erreur est multipliée par alpha quand on surestime (alpha=1 -> MSE classique)."""
    diff = pred - cible
    poids = torch.where(diff > 0, torch.full_like(diff, alpha), torch.ones_like(diff))
    return (poids * diff ** 2).mean()
