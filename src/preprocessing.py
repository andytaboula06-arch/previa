"""Préparation des données C-MAPSS : chargement, RUL, lissage, fenêtrage."""
import numpy as np
import pandas as pd

COLS = ['unit', 'cycle'] + [f'set{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
TOUS_CAPTEURS = [f's{i}' for i in range(1, 22)]


def charger(chemin: str) -> pd.DataFrame:
    """Lit un fichier brut C-MAPSS (train_FD001.txt, test_FD001.txt...)."""
    return pd.read_csv(chemin, sep=r'\s+', header=None, names=COLS)


def ajouter_rul(df: pd.DataFrame, rul_max: int = 125) -> pd.DataFrame:
    """RUL = dernier cycle du moteur - cycle actuel, plafonnée à rul_max."""
    df = df.copy()
    df['RUL_clip'] = (df.groupby('unit')['cycle'].transform('max') - df['cycle']).clip(upper=rul_max)
    return df


def choisir_capteurs(train: pd.DataFrame) -> list:
    """Garde les capteurs non constants."""
    return [c for c in TOUS_CAPTEURS if train[c].std() > 1e-6]


def lisser(df: pd.DataFrame, cols: list, w: int = 5) -> pd.DataFrame:
    """Moyenne mobile (passé uniquement) calculée séparément pour chaque moteur."""
    df = df.copy()
    df[cols] = df.groupby('unit')[cols].transform(lambda x: x.rolling(w, min_periods=1).mean())
    return df


def derniere_fenetre(valeurs: np.ndarray, w: int) -> np.ndarray:
    """Les w derniers cycles ; si le moteur est plus court, on répète son 1er cycle au début."""
    if len(valeurs) < w:
        valeurs = np.vstack([np.repeat(valeurs[:1], w - len(valeurs), axis=0), valeurs])
    return valeurs[-w:]


def fenetres_train(df: pd.DataFrame, cols: list, w: int = 30):
    """Toutes les fenêtres glissantes ; cible = RUL au dernier cycle de la fenêtre."""
    X, y = [], []
    for _, g in df.groupby('unit'):
        v, r = g[cols].values, g['RUL_clip'].values
        for t in range(w - 1, len(g)):
            X.append(v[t - w + 1:t + 1])
            y.append(r[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fenetres_test(df: pd.DataFrame, cols: list, w: int = 30) -> np.ndarray:
    """Une fenêtre par moteur de test (ses w derniers cycles)."""
    return np.array([derniere_fenetre(g[cols].values, w) for _, g in df.groupby('unit')], dtype=np.float32)
