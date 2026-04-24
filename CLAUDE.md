# CLAUDE.md — Projet P9 OpenClassrooms

## Contexte

Projet de Data Science (OpenClassrooms, Nathan, Mars 2026).
**Objectif** : Preuve de concept — prédire l'issue d'un match de League of Legends pro en temps réel, à différents checkpoints de la partie.

## Dataset

- **Source** : Oracle's Elixir 2025 (`2025_LoL_esports_match_data_from_OraclesElixir.csv`)
- **Scope** : matchs professionnels (LEC, LCS, LCK, LPL, Worlds)
- **Format** : une ligne par équipe par match (filtré sur participantid 100/200)
- **Taille** : 9 236 matchs complets après nettoyage → `data_cleaned.csv`
- **Tâche** : classification binaire (victoire / défaite)

## Architecture du projet

```
p9/
├── dashboard.py          # App Streamlit de prédiction interactive
├── plan_previsionnel.md  # Document de cadrage du projet
│
├── data/
│   ├── 2025_LoL_esports_match_data_from_OraclesElixir.csv
│   └── data_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb      # Exploration, nettoyage, feature engineering par checkpoint
│   └── 02_models.ipynb   # Entraînement XGBoost + TabNet + FT-Transformer, évaluation, sérialisation
│
└── models/
    ├── features_{10,15,20,25}min.pkl
    ├── scaler_{10,15,20,25}min.pkl
    ├── xgb_{10,15,20,25}min.pkl
    ├── tabnet_{10,15,20,25}min.zip
    ├── tabnet_importances_{10,15,20,25}min.pkl
    └── ftt_{10,15,20,25}min.pt
```

## Checkpoints temporels et features

| Checkpoint | Nb features | XGBoost Acc | TabNet Acc | FT-Transf. Acc | XGBoost AUC | TabNet AUC | FT-Transf. AUC |
|------------|-------------|-------------|------------|----------------|-------------|------------|----------------|
| @10 min    | 8           | 67.96%      | 69.61%     | 69.61%         | 0.754       | 0.766      | 0.766          |
| @15 min    | 17          | 75.05%      | 76.29%     | 76.24%         | 0.832       | 0.834      | 0.839          |
| @20 min    | 23          | 78.33%      | 79.27%     | 79.46%         | 0.870       | 0.872      | 0.876          |
| @25 min    | 30          | 87.94%      | 88.59%     | 88.61%         | 0.954       | 0.954      | 0.957          |

**Checkpoint de production : @15 min** (meilleur compromis timing / performance).

### Features @10 min (7)
`side`, `firstblood`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`

### Features @15 min (16)
= @10 min + `firstdragon`, `firstherald`, `firsttower`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`

## Modèles

### XGBoost (baseline)
- 300 estimators, max_depth=6, learning_rate=0.05
- Pas de normalisation nécessaire
- Sérialisé avec `pickle`

### TabNet
- Librairie : `pytorch-tabnet`
- Params : n_d=32, n_a=32, n_steps=5, gamma=1.5
- Nécessite un `StandardScaler` sur les features
- Interprétabilité native via masques d'attention (pas besoin de SHAP)
- Sérialisé avec `.save_model()` → fichier `.zip`

### FT-Transformer (modèle principal, < 5 ans)
- Librairie : `rtdl` v0.0.13
- Architecture : `FTTransformer.make_default(n_num_features=N, cat_cardinalities=None, d_out=1)`
- Papier : Gorishniy et al., NeurIPS 2021 — https://arxiv.org/abs/2106.11959
- Optimizer : AdamW via `model.make_default_optimizer()`, Loss : `BCEWithLogitsLoss`
- Early stopping : patience=20, max_epochs=200, batch_size=256
- Sérialisé avec `torch.save(model, ...)` → fichier `.pt`

## Dashboard Streamlit (`dashboard.py`)

Lancer avec : `streamlit run dashboard.py`

**Deux modes :**
1. **Charger un match existant** : sélection par ligue/date/équipes, stats auto-remplies
2. **Saisie manuelle** : sliders pour les stats numériques, toggles pour les objectifs binaires

**Output** : probabilités de victoire XGBoost + TabNet + FT-Transformer + bar chart feature importance TabNet

## Stack technique

```
pandas, numpy          # Data processing
matplotlib, seaborn    # Visualisation EDA
scikit-learn           # Préprocessing, métriques, split
xgboost                # Baseline
pytorch-tabnet         # TabNet
rtdl                   # FT-Transformer
torch                  # Backend DL
streamlit              # Dashboard
pickle                 # Sérialisation modèles
```

## Références clés

1. Arik & Pfister (2021) — TabNet — https://arxiv.org/abs/1908.07442
2. Junior & Campelo (2023) — LoL match prediction (LightGBM) — https://arxiv.org/abs/2309.02449
3. Chen & Guestrin (2016) — XGBoost — https://arxiv.org/abs/1603.02754

## Points importants

- **Pas de data leakage** : chaque checkpoint n'utilise que les stats disponibles à cet instant
- **FT-Transformer meilleur AUC** à partir de @15 min, TabNet et FT-Transformer dominent XGBoost à chaque checkpoint
- Le projet est **complet** : EDA, modèles, dashboard tous fonctionnels
- Les modèles sont sérialisés dans `models/`, les données dans `data/`
- Le dashboard tourne depuis la racine `p9/` (`streamlit run dashboard.py`)
