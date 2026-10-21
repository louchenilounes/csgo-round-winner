"""Suivi des expériences avec MLflow.

Stockage local :
  - mlflow.db : base SQLite (paramètres, métriques, Model Registry) ;
  - mlruns/   : fichiers des runs (modèles, graphiques).
Les deux sont exclus de Git.

Interface web :
    mlflow ui --backend-store-uri sqlite:///mlflow.db
puis ouvrir http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src.config import ROOT_DIR

EXPERIMENT_NAME = "csgo-round-winner"
REGISTERED_MODEL_NAME = "csgo-round-winner-lightgbm"
METRICS = ["accuracy", "roc_auc", "log_loss", "brier"]


def setup(root: Path = ROOT_DIR, experiment: str = EXPERIMENT_NAME) -> str:
    """Configure MLflow (base SQLite + dossier d'artefacts) et renvoie l'id de l'expérience."""
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        exp_id = mlflow.create_experiment(experiment, artifact_location=(root / "mlruns").as_uri())
    else:
        exp_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def log_cv_run(name: str, params: dict, folds: pd.DataFrame, tags: dict | None = None) -> str:
    """Enregistre un run de validation croisée : paramètres, moyenne et écart-type, puis chaque pli."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"type": "validation_croisee", **(tags or {})})
        mlflow.log_params(params)
        for m in METRICS:
            mlflow.log_metric(f"cv_{m}", float(folds[m].mean()))
            mlflow.log_metric(f"cv_{m}_std", float(folds[m].std()))
            for fold, value in zip(folds["fold"], folds[m]):
                mlflow.log_metric(f"fold_{m}", float(value), step=int(fold))
        mlflow.log_metric("cv_seconds", float(folds["seconds"].sum()))
        return run.info.run_id


def log_final_model(pipeline, params: dict, test_metrics: dict, X_example: pd.DataFrame,
                    artifacts: list[Path] | None = None, register: bool = True) -> str:
    """Enregistre le modèle final (pipeline complet), ses métriques de test, et l'ajoute au registre."""
    with mlflow.start_run(run_name="Modèle final") as run:
        mlflow.set_tags({"type": "modele_final"})
        mlflow.log_params(params)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="figures")
        proba = pipeline.predict_proba(X_example)[:, 1]
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_example, proba),
            input_example=X_example.head(3),
            code_paths=[str(ROOT_DIR / "src")],
            # le pipeline contient des classes du projet (FeatureBuilder) : format cloudpickle,
            # compatible avec toutes les versions de MLflow (modèle produit localement, donc de confiance)
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME if register else None,
        )
        return run.info.run_id


def runs_table() -> pd.DataFrame:
    """Tous les runs de l'expérience, avec leurs métriques principales."""
    runs = mlflow.search_runs(order_by=["start_time ASC"])
    metrics = [f"metrics.{prefix}_{m}" for prefix in ("cv", "test") for m in METRICS]
    keep = [c for c in ["tags.mlflow.runName", "tags.type"] + metrics if c in runs.columns]
    names = {"tags.mlflow.runName": "runName", "tags.type": "type"}
    return runs[keep].rename(columns=lambda c: names.get(c, c.removeprefix("metrics.")))
