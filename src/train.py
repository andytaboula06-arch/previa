"""Entraînement complet : python -m src.train --data-dir data --out artifacts

Sauvegarde dans --out : model.pt, scaler.joblib, config.json (tout ce dont l'API a besoin).
Les essais sont tracés dans MLflow (base locale mlflow.db).
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluer
from .model import LSTMRegressor, perte_asym
from .preprocessing import (ajouter_rul, charger, choisir_capteurs, fenetres_test,
                            fenetres_train, lisser)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--subset', default='FD001')
    p.add_argument('--out', default='artifacts')
    p.add_argument('--alpha', type=float, default=3.0)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--smooth', type=int, default=5)
    p.add_argument('--rul-max', type=int, default=125)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-mlflow', action='store_true')
    return p.parse_args()


def preparer_donnees(a):
    d = a.data_dir
    train = ajouter_rul(charger(f'{d}/train_{a.subset}.txt'), a.rul_max)
    test = charger(f'{d}/test_{a.subset}.txt')
    y_test = pd.read_csv(f'{d}/RUL_{a.subset}.txt', sep=r'\s+', header=None)[0].values.astype(np.float32)

    capteurs = choisir_capteurs(train)
    train_s, test_s = lisser(train, capteurs, a.smooth), lisser(test, capteurs, a.smooth)

    scaler = MinMaxScaler().fit(train_s[capteurs].values)   # appris sur le train uniquement
    train_s[capteurs] = scaler.transform(train_s[capteurs].values)
    test_s[capteurs] = scaler.transform(test_s[capteurs].values)

    # validation = les 15 % derniers moteurs du train (séparation par moteur)
    units = sorted(train_s['unit'].unique())
    n_val = max(1, int(0.15 * len(units)))
    val_units = units[-n_val:]
    tr = train_s[~train_s['unit'].isin(val_units)]
    va = train_s[train_s['unit'].isin(val_units)]

    X_tr, y_tr = fenetres_train(tr, capteurs, a.window)
    X_va, y_va = fenetres_train(va, capteurs, a.window)
    X_te = fenetres_test(test_s, capteurs, a.window)
    return capteurs, scaler, (X_tr, y_tr), (X_va, y_va), (X_te, y_test)


def main():
    a = parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    capteurs, scaler, (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = preparer_donnees(a)
    print(f"train {X_tr.shape} | val {X_va.shape} | test {X_te.shape} | {len(capteurs)} capteurs")

    mlflow = None
    if not a.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment(f'cmapss_{a.subset.lower()}_rul')
        mlflow.start_run()
        mlflow.log_params(vars(a))

    def loader(X, y, shuffle):
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y / a.rul_max)),
                          batch_size=a.batch, shuffle=shuffle)

    tr_loader, va_loader = loader(X_tr, y_tr, True), loader(X_va, y_va, False)
    model = LSTMRegressor(len(capteurs), a.hidden, a.layers, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    meilleur, etat, attente = float('inf'), None, 0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = perte_asym(model(xb), yb, a.alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        model.eval(); vtot = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                vtot += perte_asym(model(xb), yb, a.alpha).item() * len(xb)
        tl, vl = tot / len(X_tr), vtot / len(X_va)
        if mlflow:
            mlflow.log_metric('train_loss', tl, step=ep); mlflow.log_metric('val_loss', vl, step=ep)
        if ep % 5 == 0 or ep == 1:
            print(f'époque {ep:2d} | train {tl:.5f} | val {vl:.5f}')
        if vl < meilleur - 1e-6:
            meilleur, attente = vl, 0
            etat = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            attente += 1
            if attente >= a.patience:
                print(f"early stopping à l'époque {ep}")
                break

    model.load_state_dict(etat); model.eval()
    with torch.no_grad():
        pred = np.clip(model(torch.tensor(X_te)).numpy() * a.rul_max, 0, a.rul_max)
    res = evaluer(pred, y_te)
    print('Résultats test :', {k: round(v, 2) for k, v in res.items()})

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), f'{a.out}/model.pt')
    joblib.dump(scaler, f'{a.out}/scaler.joblib')
    config = dict(capteurs=capteurs, window=a.window, smooth=a.smooth, rul_max=a.rul_max,
                  hidden=a.hidden, layers=a.layers, dropout=a.dropout, subset=a.subset)
    with open(f'{a.out}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Artefacts sauvegardés dans {a.out}/')

    if mlflow:
        mlflow.log_metrics(res)
        mlflow.log_artifacts(a.out)
        mlflow.end_run()


if __name__ == '__main__':
    main()
