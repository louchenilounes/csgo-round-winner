# CS:GO Round Winner – Prédiction du gagnant d'un round

Prédire la probabilité de victoire d'une équipe (Terroristes ou Anti-terroristes) à partir de l'état d'un round de CS:GO : joueurs vivants, santé, armure, argent, armes, bombe posée, temps restant.

> Projet en cours d'amélioration. Ce README sera complété à chaque étape.

## Avancement

| Étape | Contenu | Statut |
| --- | --- | --- |
| 0 | Organisation du projet | Terminé |
| 1 | Chargement propre des données | Terminé |
| 2 | Analyse exploratoire | À faire |
| 3 | Reconstruction des rounds et correction de la fuite de données | À faire |
| 4 | Feature engineering | À faire |
| 5 | Baselines | À faire |
| 6 | Modèles avancés | À faire |
| 7 | Évaluation complète | À faire |
| 8 | Interprétabilité (SHAP) | À faire |
| 9 | MLOps léger (MLflow) | À faire |
| 10 | Démo (FastAPI + Streamlit + Docker) | À faire |
| 11 | Qualité du code (tests, CI) | À faire |
| 12 | Présentation finale | À faire |

## Structure du projet

```
csgo-round-winner/
├── app/                  # API et démo (étape 10)
├── data/
│   ├── raw/              # données brutes (non versionnées)
│   └── processed/        # données nettoyées (non versionnées)
├── models/               # modèles entraînés (non versionnés)
├── notebooks/            # notebooks d'analyse
│   ├── 00_notebook_initial.ipynb   # première version du projet
│   └── 01_chargement_donnees.ipynb # étape 1 : chargement et qualité
├── src/                  # code Python réutilisable
│   ├── config.py         # chemins et graine aléatoire
│   └── data.py           # téléchargement, lecture ARFF, types, contrôles
├── tests/                # tests automatiques
├── requirements.txt
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

## Vérifier l'installation

```bash
python -m src.config
python -m pytest
```
