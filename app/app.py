"""Dashboard Streamlit : prédiction de la RUL d'un moteur turbofan (NASA C-MAPSS FD001).

Lancer :  streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import RULPredictor, statut_maintenance
from src.preprocessing import charger

st.set_page_config(page_title='Maintenance prédictive – RUL', page_icon='✈️', layout='wide')


@st.cache_resource
def charger_modele():
    return RULPredictor.load(str(ROOT / 'artifacts'))


@st.cache_data
def charger_donnees():
    test = charger(str(ROOT / 'data' / 'test_FD001.txt'))
    rul_finale = pd.read_csv(ROOT / 'data' / 'RUL_FD001.txt', sep=r'\s+', header=None)[0].values
    return test, rul_finale


@st.cache_data(show_spinner='Calcul de la RUL à chaque cycle…')
def trajectoire(unit: int):
    """RUL prédite à chaque cycle, comme si on avait reçu les données cycle après cycle."""
    modele, (test, _) = charger_modele(), charger_donnees()
    g = test[test['unit'] == unit].sort_values('cycle')
    X = g[modele.capteurs]
    return [modele.predict(X.iloc[:k]) for k in range(1, len(g) + 1)]


try:
    modele = charger_modele()
    test, rul_finale = charger_donnees()
except Exception as e:
    st.error(f"Impossible de charger le modèle ou les données : {e}")
    st.info("Entraîne le modèle (`python -m src.train`) et place `test_FD001.txt` et `RUL_FD001.txt` dans `data/`.")
    st.stop()

COULEURS = {'vert': ('✅', 'success', "Moteur en bon état"),
            'orange': ('⚠️', 'warning', 'Planifier une maintenance prochainement'),
            'rouge': ('🚨', 'error', 'Maintenance urgente')}

st.title('✈️ Maintenance prédictive – durée de vie restante (RUL)')
st.caption("Un LSTM estime combien de cycles de vol il reste à un moteur turbofan avant la panne "
           "(données simulées NASA C-MAPSS, jeu FD001).")

# ---------- Barre latérale ----------
st.sidebar.header('Paramètres')
unites = sorted(test['unit'].unique())
unit = st.sidebar.selectbox('Moteur (jeu de test)', unites)
moteur = test[test['unit'] == unit].sort_values('cycle').reset_index(drop=True)
n = len(moteur)

cycle_courant = st.sidebar.slider('Cycle courant (simule l\'avancement des vols)',
                                  min_value=min(10, n), max_value=n, value=n)
st.sidebar.caption(f'Ce moteur a {n} cycles enregistrés dans le jeu de test.')
capteurs_choisis = st.sidebar.multiselect(
    'Capteurs à afficher', modele.capteurs,
    default=[c for c in ['s2', 's4', 's11', 's15'] if c in modele.capteurs])

# ---------- Prédiction ----------
historique = moteur.iloc[:cycle_courant]
rul_pred = modele.predict(historique[modele.capteurs])
rul_vraie = int(rul_finale[unit - 1]) + (n - cycle_courant)   # RUL réelle au cycle courant
icone, niveau, message = COULEURS[statut_maintenance(rul_pred)]

c1, c2, c3 = st.columns(3)
c1.metric('RUL prédite', f'{rul_pred:.0f} cycles')
c2.metric('RUL réelle', f'{rul_vraie} cycles',
          delta=f'{rul_pred - rul_vraie:+.0f} (erreur du modèle)', delta_color='off')
c3.metric('Cycle courant', f'{cycle_courant} / {n}')
getattr(st, niveau)(f'{icone} {message}')
if rul_pred > rul_vraie:
    st.caption('⚠️ Le modèle surestime la RUL ici : c\'est le cas le plus dangereux, pénalisé par le S-score.')

# ---------- Graphiques ----------
onglet1, onglet2 = st.tabs(['📈 Capteurs', '🎯 Évolution de la RUL'])

with onglet1:
    if not capteurs_choisis:
        st.info('Choisis au moins un capteur dans la barre latérale.')
    else:
        fig, axes = plt.subplots(len(capteurs_choisis), 1, figsize=(10, 2.2 * len(capteurs_choisis)), sharex=True)
        axes = [axes] if len(capteurs_choisis) == 1 else axes
        for ax, c in zip(axes, capteurs_choisis):
            ax.plot(moteur['cycle'], moteur[c], color='tab:blue', lw=1)
            ax.axvline(cycle_courant, color='red', ls='--', lw=1)
            ax.set_ylabel(c)
        axes[-1].set_xlabel('Cycle')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with onglet2:
    traj = trajectoire(int(unit))
    cycles = moteur['cycle'].values
    vraie = [min(int(rul_finale[unit - 1]) + (n - k), modele.rul_max) for k in range(1, n + 1)]
    debut = modele.window   # avant 30 cycles, la fenêtre est incomplète
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles[debut - 1:], vraie[debut - 1:], color='black', label=f'RUL réelle (plafonnée à {modele.rul_max})')
    ax.plot(cycles[debut - 1:], traj[debut - 1:], color='tab:orange', label='RUL prédite')
    ax.axvline(cycle_courant, color='red', ls='--', lw=1, label='Cycle courant')
    ax.axhline(30, color='tab:red', ls=':', lw=1)
    ax.set_xlabel('Cycle'); ax.set_ylabel('RUL (cycles)'); ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Plus la courbe orange suit la noire, mieux c\'est. Elle devient fiable surtout en fin de vie, '
               'quand la dégradation est visible. Ligne pointillée rouge : seuil critique (30 cycles).')

with st.expander('ℹ️ Comment ça marche ?'):
    st.markdown(f"""
- Les relevés des capteurs sont **lissés**, **normalisés**, puis les **{modele.window} derniers cycles** sont donnés à un réseau **LSTM**.
- La RUL est **plafonnée à {modele.rul_max}** : un moteur neuf ne montre aucune dégradation visible.
- Le modèle est entraîné avec une **perte asymétrique** qui pénalise davantage la surestimation de la RUL
  (plus dangereuse que la sous-estimation), comme le **S-score** de la NASA.
- Code couleur : 🟢 > 60 cycles · 🟠 31 à 60 cycles · 🔴 ≤ 30 cycles.
""")
