"""Chargement du modèle entraîné et prédiction de la RUL (utilisé par l'API et le dashboard)."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from .preprocessing import derniere_fenetre, lisser


class RULPredictor:
    def __init__(self, config: dict, scaler, model=None):
        self.config = config
        self.scaler = scaler
        self.model = model
        self.capteurs = config['capteurs']
        self.window = config['window']
        self.smooth = config['smooth']
        self.rul_max = config['rul_max']

    @classmethod
    def load(cls, dossier: str = 'artifacts') -> 'RULPredictor':
        import torch
        from .model import LSTMRegressor

        with open(os.path.join(dossier, 'config.json')) as f:
            config = json.load(f)
        scaler = joblib.load(os.path.join(dossier, 'scaler.joblib'))
        model = LSTMRegressor(len(config['capteurs']), config['hidden'], config['layers'], config['dropout'])
        model.load_state_dict(torch.load(os.path.join(dossier, 'model.pt'), map_location='cpu'))
        model.eval()
        return cls(config, scaler, model)

    def prepare(self, cycles: pd.DataFrame) -> np.ndarray:
        """Données brutes (1 ligne par cycle) -> tenseur (1, window, nb_capteurs), même traitement qu'à l'entraînement."""
        manquants = [c for c in self.capteurs if c not in cycles.columns]
        if manquants:
            raise ValueError(f"Capteurs manquants : {manquants}")
        df = cycles[self.capteurs].astype(float).copy()
        df['unit'] = 0
        df = lisser(df, self.capteurs, self.smooth)
        valeurs = self.scaler.transform(df[self.capteurs].values)
        fenetre = derniere_fenetre(valeurs, self.window)
        return fenetre[None, :, :].astype(np.float32)

    def _forward(self, x: np.ndarray) -> float:
        import torch
        with torch.no_grad():
            pred = self.model(torch.tensor(x)).item() * self.rul_max
        return pred

    def predict(self, cycles: pd.DataFrame) -> float:
        pred = self._forward(self.prepare(cycles))
        return float(np.clip(pred, 0, self.rul_max))


def statut_maintenance(rul: float) -> str:
    """Code couleur de maintenance : rouge (<= 30 cycles), orange (<= 60), sinon vert."""
    if rul <= 30:
        return 'rouge'
    if rul <= 60:
        return 'orange'
    return 'vert'
