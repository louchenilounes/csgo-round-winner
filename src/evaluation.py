"""Évaluation des modèles par validation croisée groupée.

Pour comparer des modèles ou des jeux de variables, on n'utilise JAMAIS le test.
On fait une validation croisée à l'intérieur du train, avec GroupKFold :
chaque match est entièrement dans un seul pli (fold). Comme pour le découpage
train/test, cela évite que des photos d'un même match soient à la fois
dans les données d'entraînement et de validation.

Métriques :
  - accuracy : part de bonnes prédictions (seuil 50 %) ;
  - ROC-AUC  : capacité à classer les situations de la plus à la moins favorable aux T ;
  - log loss : qualité des probabilités (plus c'est bas, mieux c'est) ;
  - Brier    : erreur quadratique moyenne des probabilités (plus c'est bas, mieux c'est).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.config import TARGET

N_FOLDS = 5


def compute_metrics(y_true, proba) -> dict:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y_true, (proba >= 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def grouped_cv(
    pipeline, df: pd.DataFrame, group_col: str = "match_id", n_folds: int = N_FOLDS
) -> pd.DataFrame:
    """Validation croisée GroupKFold. Renvoie une ligne de métriques par pli."""
    y = df[TARGET].to_numpy()
    groups = df[group_col].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(df, y, groups)):
        start = time.time()
        model = clone(pipeline)
        model.fit(df.iloc[tr], y[tr])
        proba = model.predict_proba(df.iloc[va])[:, 1]
        rows.append({"fold": fold, **compute_metrics(y[va], proba), "seconds": time.time() - start})
    return pd.DataFrame(rows)


def summarize_cv(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tableau moyenne ± écart-type des métriques, une ligne par configuration."""
    rows = []
    for name, folds in results.items():
        row = {"configuration": name}
        for m in ["accuracy", "roc_auc", "log_loss", "brier"]:
            row[m] = folds[m].mean()
            row[f"{m}_std"] = folds[m].std()
        row["seconds"] = folds["seconds"].sum()
        rows.append(row)
    return pd.DataFrame(rows)
