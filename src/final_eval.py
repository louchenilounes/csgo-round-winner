"""Outils pour l'évaluation finale sur le test.

  - intervalles de confiance par bootstrap en ré-échantillonnant des MATCHS entiers
    (les snapshots d'un même match ne sont pas indépendants) ;
  - courbe de calibration et erreur de calibration (ECE) : quand le modèle annonce
    70 %, les T gagnent-ils vraiment environ 70 % du temps ?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEED
from src.evaluation import compute_metrics


def bootstrap_by_group(y, proba, groups, n_boot: int = 1000, seed: int = SEED,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Intervalle de confiance de chaque métrique, en tirant des groupes avec remise."""
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        samples.append(compute_metrics(y[idx], proba[idx]))
    samples = pd.DataFrame(samples)
    point = compute_metrics(y, proba)
    return pd.DataFrame({
        "valeur": pd.Series(point),
        "borne_basse": samples.quantile(alpha / 2),
        "borne_haute": samples.quantile(1 - alpha / 2),
    })


def calibration_table(y, proba, n_bins: int = 10) -> pd.DataFrame:
    """Par tranche de probabilité prédite : probabilité moyenne, fréquence observée, effectif."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)})
    edges = np.linspace(0, 1, n_bins + 1)
    d["bin"] = pd.cut(d["p"], edges, include_lowest=True)
    out = d.groupby("bin", observed=True).agg(proba_moyenne=("p", "mean"),
                                              frequence_observee=("y", "mean"),
                                              n=("y", "size"))
    return out.reset_index()


def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    """ECE : écart moyen (pondéré par l'effectif) entre probabilité annoncée et fréquence observée."""
    t = calibration_table(y, proba, n_bins)
    weights = t["n"] / t["n"].sum()
    return float((weights * (t["proba_moyenne"] - t["frequence_observee"]).abs()).sum())


def metrics_by(df: pd.DataFrame, y_col: str, proba_col: str, by) -> pd.DataFrame:
    """Métriques calculées séparément pour chaque valeur de `by` (carte, phase du round...)."""
    rows = []
    for key, part in df.groupby(by, observed=True):
        if part[y_col].nunique() < 2:
            continue
        rows.append({"groupe": key, "n": len(part),
                     **compute_metrics(part[y_col].to_numpy(), part[proba_col].to_numpy())})
    return pd.DataFrame(rows)
