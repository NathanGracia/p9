# Plan Prévisionnel - Preuve de Concept

## Auteur
Nathan
Formation Data Science - OpenClassrooms
Date : Mars 2026

---

## 1. Algorithme envisagé

**TabNet** (Arik & Pfister, 2021 - Google Brain)

TabNet est un réseau de neurones conçu spécifiquement pour les données tabulaires. Il repose sur un mécanisme d'**attention séquentielle** : à chaque étape de décision, le modèle sélectionne les features les plus pertinentes plutôt que de toutes les utiliser simultanément. Cela lui confère deux avantages majeurs :

- **Performance** : compétitif face aux méthodes ensemblistes (XGBoost, LightGBM) sur des données tabulaires
- **Interprétabilité** : les masques d'attention permettent de visualiser quelles features ont contribué à chaque prédiction

**Arguments justifiant le choix :**

1. **Récence** : publié en 2021, TabNet date de moins de 5 ans et représente une avancée significative dans l'application des réseaux de neurones aux données tabulaires
2. **Pertinence** : le dataset Oracle's Elixir est purement tabulaire (stats par match), TabNet est donc directement applicable
3. **Différenciation** : la littérature existante sur la prédiction de matchs LoL (Junior & Campelo, 2023) utilise LightGBM, XGBoost et des réseaux classiques — aucune étude connue n'applique TabNet sur ce problème
4. **Interprétabilité** : l'aspect "quelles stats influencent le plus la victoire" est particulièrement intéressant dans le contexte esport

**Baseline retenue :** XGBoost — algorithme de référence sur les données tabulaires, utilisé dans la majorité des études comparables

---

## 2. Dataset

**Oracle's Elixir — Matchs professionnels LoL 2024**

- **Source** : [oracleselixir.com/tools/downloads](https://oracleselixir.com/tools/downloads)
- **Contenu** : données de matchs professionnels (LEC, LCS, LCK, LPL, Worlds 2024)
- **Format** : CSV, mis à jour quotidiennement
- **Tâche** : classification binaire — prédire l'issue d'un match (victoire/défaite) à partir des statistiques d'équipe

**Features disponibles (exemples) :**
- Statistiques early game (kills, gold diff à 10/15 min, dragons, tours)
- Statistiques globales de match (KDA, vision score, dégâts)
- Données de draft (champions joués, positions)

**Pourquoi ce dataset :**
- Données récentes (2024), méta actuel
- Contexte pro = données plus structurées et homogènes que le ranked
- Aucun papier existant n'applique TabNet sur ce dataset spécifiquement

---

## 3. Références bibliographiques

1. **Arik, S. O., & Pfister, T. (2021).** TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence.*
   → https://arxiv.org/abs/1908.07442
   *(Papier fondateur de TabNet, publié sur Arxiv)*

2. **Junior, J. B. S., & Campelo, C. E. C. (2023).** League of Legends: Real-Time Result Prediction.
   → https://arxiv.org/abs/2309.02449
   *(Référence principale sur la prédiction de matchs LoL, utilise LightGBM comme meilleur modèle — sert de comparaison pour situer nos résultats)*

3. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *KDD 2016.*
   → https://arxiv.org/abs/1603.02754
   *(Papier de référence pour la baseline XGBoost)*

---

## 4. Démarche de test (Preuve de Concept)

### Étape 1 — Collecte et préparation des données
- Téléchargement du fichier CSV 2024 depuis Oracle's Elixir
- Filtrage sur les lignes de type "team" (une ligne par équipe par match)
- Sélection des features pertinentes (stats early game + stats globales)
- Encodage des variables catégorielles (champions, équipes)
- Split train/test (80/20, stratifié)

### Étape 2 — Baseline XGBoost
- Entraînement d'un modèle XGBoost avec hyperparamètres par défaut
- Optimisation par GridSearch/RandomSearch
- Évaluation : accuracy, F1-score, AUC-ROC

### Étape 3 — Modèle TabNet
- Implémentation via la librairie `pytorch-tabnet`
- Entraînement sur les mêmes features et split que la baseline
- Optimisation des hyperparamètres (n_steps, gamma, learning rate)
- Évaluation : mêmes métriques que la baseline

### Étape 4 — Comparaison et analyse
- Tableau comparatif des métriques (accuracy, F1, AUC-ROC)
- Visualisation des masques d'attention TabNet (feature importance)
- Analyse des features les plus déterminantes pour la victoire en pro play

### Étape 5 — Dashboard Streamlit
- Interface permettant d'entrer des stats d'équipe
- Prédiction de l'issue du match en temps réel
- Affichage de la contribution de chaque feature (interprétabilité TabNet)

### Note sur la réutilisation de code
Le code d'implémentation TabNet s'appuiera sur la documentation officielle de `pytorch-tabnet` et des tutoriels publics. Le dataset Oracle's Elixir est différent de tout dataset utilisé dans ces tutoriels (qui utilisent généralement Titanic, Adult Income ou similaires). Cette utilisation sera explicitement mentionnée dans la note méthodologique.
