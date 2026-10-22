"""Entraînement du modèle final sur tout le train et sauvegarde du pipeline complet.

Le fichier sauvegardé contient tout le pipeline (création des variables, encodage
de la carte, modèle) : il prend en entrée les colonnes brutes d'un snapshot et
renvoie une probabilité de victoire des T.

Utilisation :
    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, TARGET
from src.features import build_pipeline
from src.rounds import load_processed
from src.serving import MODEL_PATH, load_model  # chemin et chargement définis dans serving.py
from src.tuning import BEST_PARAMS_PATH, load_best_params, make_lgbm

MODEL_INFO_PATH = MODELS_DIR / "lightgbm_pipeline.json"


def train_final_model(train: pd.DataFrame, params: dict) -> Pipeline:
    pipeline = build_pipeline(make_lgbm(params))
    pipeline.fit(train, train[TARGET])
    return pipeline


def save_model(pipeline: Pipeline, params: dict, n_rows: int,
               path: Path = MODEL_PATH, info_path: Path = MODEL_INFO_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    info = {"model": "LightGBM", "params": params, "train_rows": n_rows,
            "features": list(pipeline.named_steps["prep"].get_feature_names_out())}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    if not BEST_PARAMS_PATH.exists():
        raise SystemExit(f"{BEST_PARAMS_PATH.name} introuvable : lancer d'abord le notebook 06.")
    df = load_processed()
    train = df[df["split"] == "train"].reset_index(drop=True)
    params = load_best_params()
    model = train_final_model(train, params)
    path = save_model(model, params, len(train))
    print(f"Modèle entraîné sur {len(train):,} lignes et sauvegardé : {path}")
