"""Optimisation des hyperparamètres de LightGBM avec Optuna.

Chaque essai (trial) propose un jeu d'hyperparamètres, évalué par validation croisée
groupée par match sur le train. Le critère optimisé est la log loss : elle mesure la
qualité des probabilités, qui sont la vraie sortie du modèle, et elle est plus stable
que l'accuracy d'un essai à l'autre.
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import ROOT_DIR, SEED
from src.evaluation import grouped_cv
from src.features import build_pipeline

BEST_PARAMS_PATH = ROOT_DIR / "reports" / "best_params_lightgbm.json"

# Valeurs par défaut raisonnables (point de départ de la recherche).
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def make_lgbm(params: dict | None = None, seed: int = SEED) -> LGBMClassifier:
    """LightGBM avec des réglages fixes (graine, silence) + les hyperparamètres donnés."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return LGBMClassifier(**p, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)


def suggest_params(trial: optuna.Trial) -> dict:
    """Espace de recherche."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def tune_lightgbm(
    train: pd.DataFrame, n_trials: int = 30, n_folds: int = 5, seed: int = SEED, show_progress: bool = True
) -> optuna.Study:
    """Lance la recherche et renvoie l'étude Optuna (tous les essais y sont conservés)."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        folds = grouped_cv(build_pipeline(make_lgbm(params, seed)), train, n_folds=n_folds)
        trial.set_user_attr("accuracy", float(folds["accuracy"].mean()))
        trial.set_user_attr("roc_auc", float(folds["roc_auc"].mean()))
        trial.set_user_attr("brier", float(folds["brier"].mean()))
        return float(folds["log_loss"].mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), study_name="lightgbm_log_loss"
    )
    study.enqueue_trial(DEFAULT_PARAMS)  # le 1er essai = réglages par défaut, pour comparaison
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """Un essai par ligne : hyperparamètres et métriques, trié par log loss."""
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append({"trial": t.number, "log_loss": t.value, **t.user_attrs, **t.params})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def save_best_params(params: dict, path: Path = BEST_PARAMS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return path


def load_best_params(path: Path = BEST_PARAMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
