# LoL Pro Match Predictor

Preuve de concept — prédiction de l'issue d'un match professionnel de League of Legends en temps réel, à différents checkpoints de la partie.

Projet Data Science P9, OpenClassrooms — Nathan Gracia, 2026.

---

## Présentation

Le modèle prédit la victoire ou la défaite d'une équipe à partir des statistiques disponibles à 10, 15, 20 et 25 minutes de jeu. Trois algorithmes sont comparés :

- **XGBoost** — baseline classique (arbres boostés)
- **TabNet** — baseline deep learning (Arik & Pfister, 2021)
- **FT-Transformer** — modèle principal (Gorishniy et al., NeurIPS 2021), optimisé via Optuna

| Checkpoint | FT-Transformer AUC | XGBoost AUC |
|------------|-------------------|-------------|
| @10 min    | 0.766             | 0.754       |
| @15 min    | 0.839             | 0.832       |
| @20 min    | 0.876             | 0.870       |
| @25 min    | 0.957             | 0.954       |

Dataset : Oracle's Elixir 2025 — 9 236 matchs professionnels (LEC, LCS, LCK, LPL, Worlds).

---

## Structure du projet

```
p9/
├── dashboard.py               # Application Streamlit
├── run_optuna_ftt.py          # Optimisation Optuna du FT-Transformer
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   └── data_cleaned.csv       # Dataset nettoyé (9 236 matchs)
│
├── models/
│   ├── ftt/                   # FT-Transformer (4 checkpoints)
│   ├── tabnet/                # TabNet (4 checkpoints)
│   ├── xgboost/               # XGBoost (4 checkpoints)
│   └── preprocessing/         # Scalers et listes de features
│
└── notebooks/
    ├── 01_eda.ipynb            # Exploration et feature engineering
    └── 02_models.ipynb         # Entraînement et évaluation
```

---

## Lancer le dashboard

### En local

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

### Via Docker

```bash
docker compose up -d
```

Le dashboard est accessible sur `http://localhost:8501`.

> Le fichier `data/2025_LoL_esports_match_data_from_OraclesElixir.csv` (76 Mo) n'est pas inclus dans le repo. Il doit être placé dans `data/` avant de lancer l'application.

---

## Dashboard

Deux modes de prédiction :

- **Charger un match existant** : sélection par ligue, date et équipes — les stats sont remplies automatiquement
- **Saisie manuelle** : sliders pour les stats numériques, toggles pour les objectifs binaires

Chaque mode affiche les probabilités de victoire des trois modèles aux 4 checkpoints, ainsi que l'interprétabilité TabNet (feature importance globale et locale).

---

## Références

- Gorishniy Y. et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Arik S., Pfister T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021. [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)
- Chen T., Guestrin C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Junior U., Campelo F. (2023). *Predicting the Outcome of League of Legends Games*. [arXiv:2309.02449](https://arxiv.org/abs/2309.02449)
