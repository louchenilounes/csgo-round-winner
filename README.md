# CS:GO Round Winner – Prédiction du gagnant d'un round

[![CI](https://github.com/louchenilounes/csgo-round-winner/actions/workflows/ci.yml/badge.svg)](https://github.com/louchenilounes/csgo-round-winner/actions/workflows/ci.yml)

Prédire la probabilité de victoire d'une équipe (Terroristes ou Anti-terroristes) à partir de l'état d'un round de CS:GO : joueurs vivants, santé, armure, argent, armes, bombe posée, temps restant.

> Projet en cours de développement.

## Avancement

| Étape | Contenu | Statut |
| --- | --- | --- |
| 0 | Organisation du projet | Terminé |
| 1 | Chargement propre des données | Terminé |
| 2 | Analyse exploratoire | Terminé |
| 3 | Reconstruction des rounds et correction de la fuite de données | Terminé |
| 4 | Feature engineering | Terminé |
| 5 | Baselines | Terminé |
| 6 | Modèles avancés | Terminé |
| 7 | Évaluation complète | Terminé |
| 8 | Interprétabilité (SHAP) | Terminé |
| 9 | MLOps léger (MLflow) | Terminé |
| 10 | Démo (FastAPI + Streamlit + Docker) | Terminé |
| 11 | Qualité du code (tests, CI) | Terminé |
| 12 | Présentation finale | À faire |

## Structure du projet

```
csgo-round-winner/
├── app/
│   ├── api.py            # API FastAPI (/health, /predict)
│   └── streamlit_app.py  # interface Streamlit
├── data/
│   ├── raw/              # données brutes (non versionnées)
│   └── processed/        # données nettoyées (non versionnées)
├── models/               # modèles entraînés (non versionnés)
├── notebooks/            # notebooks d'analyse
│   ├── 00_notebook_initial.ipynb   # première version du projet
│   ├── 01_chargement_donnees.ipynb # chargement et qualité des données
│   ├── 02_eda.ipynb                # analyse exploratoire
│   ├── 03_rounds_et_fuite.ipynb    # rounds, matchs et fuite de données
│   ├── 04_feature_engineering.ipynb # création des variables
│   ├── 05_baselines.ipynb          # comparaison à des références simples
│   ├── 06_modeles_avances.ipynb    # comparaison des modèles, optimisation Optuna
│   ├── 07_evaluation_finale.ipynb  # évaluation sur le test, calibration
│   └── 08_interpretation_shap.ipynb # interprétation du modèle (SHAP)
├── reports/              # graphiques et tableaux de résultats
├── src/                  # code Python réutilisable
│   ├── baselines.py      # règles et modèles de référence
│   ├── config.py         # chemins et graine aléatoire
│   ├── data.py           # téléchargement, lecture ARFF, types, contrôles
│   ├── eda.py            # calculs de l'analyse exploratoire
│   ├── evaluation.py     # validation croisée groupée et métriques
│   ├── experiments.py    # relance des expériences et enregistrement dans MLflow
│   ├── explain.py        # valeurs SHAP (TreeSHAP de LightGBM)
│   ├── features.py       # création des variables et pipelines scikit-learn
│   ├── final_eval.py     # bootstrap par match, calibration
│   ├── rounds.py         # reconstruction des rounds/matchs, split train/test
│   ├── serving.py        # situation de jeu -> entrée du modèle, prédiction expliquée
│   ├── train.py          # entraînement et sauvegarde du modèle final
│   ├── tracking.py       # suivi MLflow (runs, Model Registry)
│   ├── tuning.py         # optimisation des hyperparamètres (Optuna)
│   ├── leakage.py        # mesure de la fuite de données
│   └── viz.py            # style et couleurs des graphiques
├── scripts/              # utilitaires (modèle synthétique pour la CI)
├── tests/                # tests automatiques (pytest)
├── .github/workflows/    # intégration continue (GitHub Actions)
├── pyproject.toml        # configuration de ruff
├── Dockerfile
├── docker-compose.yml
├── requirements.txt      # environnement complet (analyse, entraînement, tests)
├── requirements-app.txt  # dépendances minimales de l'image Docker
└── README.md
```

## Installation

```bash
git clone https://github.com/<ton-pseudo>/csgo-round-winner.git
cd csgo-round-winner
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Dataset **CS:GO Round Snapshots** (OpenML) : 122 410 photos d'état de round, prises environ toutes les 20 secondes pendant des matchs de haut niveau, 96 variables + le gagnant du round.

Pour télécharger les données et afficher le rapport de qualité :

```bash
python -m src.data
```

Le fichier est téléchargé une seule fois dans `data/raw/`.

Pour reconstruire les rounds et les matchs, puis créer le découpage train/test (par match) :

```bash
python -m src.rounds
```

## Suivi des expériences (MLflow)

Relancer les expériences principales et enregistrer le modèle final dans le Model Registry :

```bash
python -m src.experiments
```

Consulter les runs dans l'interface web de MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

puis ouvrir http://127.0.0.1:5000.

## Démo

Le modèle doit d'abord être entraîné (`python -m src.train`, ou `python -m src.experiments`).

**Interface Streamlit** : on règle une situation de jeu (carte, chrono, bombe, joueurs vivants, équipement de chaque équipe) et l'application affiche la probabilité de victoire avec les facteurs qui l'expliquent (valeurs SHAP).

```bash
streamlit run app/streamlit_app.py
```

**API FastAPI** (documentation interactive sur http://127.0.0.1:8000/docs) :

```bash
uvicorn app.api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"map": "de_inferno", "time_left": 25, "bomb_planted": true,
          "ct": {"players_alive": 2}, "t": {"players_alive": 3}}'
```

Les champs non renseignés prennent des valeurs cohérentes avec le nombre de joueurs vivants ; les situations impossibles (santé supérieure à 100 par joueur, bombe posée avec plus de 40 s…) sont refusées avec un message explicite (code 422).

**Docker** (API sur le port 8000, interface sur le port 8501) :

```bash
docker compose up --build
```

## Qualité du code

```bash
python -m pytest                       # 61 tests
ruff check src app tests scripts       # analyse statique
ruff format --check src app tests scripts
```

À chaque push, GitHub Actions lance l'analyse statique et les tests, puis construit l'image Docker, la démarre et interroge l'API. Le modèle réel n'étant pas versionné, la CI utilise un petit modèle entraîné sur des données synthétiques (`scripts/make_ci_model.py`) : elle vérifie le fonctionnement de la chaîne, pas la performance.
