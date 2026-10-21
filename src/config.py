"""Configuration centrale du projet.

Tous les chemins et réglages communs sont définis ici, une seule fois.
Les autres fichiers du projet importent ces valeurs au lieu de les réécrire.
"""

from pathlib import Path

# Graine aléatoire : rend les résultats reproductibles
# (même découpage train/test et mêmes modèles à chaque exécution).
SEED = 42

# Dossier racine du projet (le dossier qui contient src/).
ROOT_DIR = Path(__file__).resolve().parents[1]

# Dossiers de données et de modèles.
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Fichier de données brutes (créé par src/data.py).
RAW_DATA_PATH = RAW_DATA_DIR / "csgo_round_snapshots.csv"

# Nom de la colonne cible : 1 si les Terroristes gagnent le round, 0 sinon.
TARGET = "t_win"


if __name__ == "__main__":
    # Petit affichage pour vérifier que la configuration fonctionne.
    print("Configuration du projet")
    print(f"  SEED            : {SEED}")
    print(f"  Dossier racine  : {ROOT_DIR}")
    print(f"  Données brutes  : {RAW_DATA_DIR} (existe : {RAW_DATA_DIR.exists()})")
    print(f"  Données propres : {PROCESSED_DATA_DIR} (existe : {PROCESSED_DATA_DIR.exists()})")
    print(f"  Modèles         : {MODELS_DIR} (existe : {MODELS_DIR.exists()})")
