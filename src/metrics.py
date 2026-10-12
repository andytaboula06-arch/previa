"""Métriques d'évaluation : RMSE, S-score NASA, F1."""
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error


def s_score(y_pred, y_true) -> float:
    """S-score NASA : pénalise beaucoup plus les surestimations (d > 0) que les sous-estimations."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluer(y_pred, y_true, seuil_panne: int = 30) -> dict:
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        's_score': s_score(y_pred, y_true),
        'f1': float(f1_score(y_true <= seuil_panne, y_pred <= seuil_panne)),
        'surestimations': int((y_pred > y_true).sum()),
    }
