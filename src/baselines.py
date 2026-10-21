"""Modèles de référence (baselines).

Un score n'a de sens que comparé à des approches simples. Trois niveaux :
  - classe majoritaire : probabilité constante, égale au taux de victoire T du train ;
  - règles métier : « l'équipe qui a le plus de joueurs vivants gagne » ;
  - tables de fréquences : probabilité = taux de victoire T observé dans le train
    pour la même situation (même écart de joueurs, bombe posée ou non).

Tous les estimateurs suivent l'API scikit-learn (fit / predict_proba) et peuvent
donc être évalués avec la même validation croisée que les vrais modèles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FeatureBuilder


def _alive_diff(X: pd.DataFrame) -> pd.Series:
    return (X["t_players_alive"] - X["ct_players_alive"]).astype(int)


def _bomb(X: pd.DataFrame) -> pd.Series:
    return X["bomb_planted"].astype(bool).astype(int)


class AliveRule(BaseEstimator, ClassifierMixin):
    """Règle fixe : l'équipe avec le plus de joueurs vivants gagne.

    En cas d'égalité, la probabilité vaut 0,5. Si `bomb_tiebreak=True`, une égalité
    est tranchée par la bombe : bombe posée -> T, sinon -> CT (sans bombe, le temps
    joue pour les CT). Aucun apprentissage : c'est une règle de jeu.
    """

    def __init__(self, bomb_tiebreak: bool = False):
        self.bomb_tiebreak = bomb_tiebreak

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        diff = _alive_diff(X).to_numpy()
        p = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
        if self.bomb_tiebreak:
            p = np.where(diff == 0, _bomb(X).to_numpy().astype(float), p)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FrequencyTable(BaseEstimator, ClassifierMixin):
    """Probabilité = taux de victoire T observé dans le train pour la même situation.

    Paramètres
    ----------
    use_bomb   : si True, la situation est (écart de joueurs, bombe posée) ; sinon l'écart seul.
    min_count  : en dessous de ce nombre de lignes, la case n'est pas fiable et on utilise
                 le taux global du train.
    """

    def __init__(self, use_bomb: bool = False, min_count: int = 30):
        self.use_bomb = use_bomb
        self.min_count = min_count

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        diff = _alive_diff(X).astype(str)
        return diff + "_" + _bomb(X).astype(str) if self.use_bomb else diff

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        stats = y.groupby(self._keys(X)).agg(["mean", "size"])
        self.global_rate_ = float(y.mean())
        self.table_ = stats.loc[stats["size"] >= self.min_count, "mean"].to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._keys(X).map(self.table_).fillna(self.global_rate_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Garde une liste de colonnes (après FeatureBuilder)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


SIMPLE_LOGREG_COLUMNS = ["players_alive_diff", "bomb_planted", "time_left_no_bomb", "time_left_bomb"]


def simple_logistic_regression(columns: list[str] = SIMPLE_LOGREG_COLUMNS) -> Pipeline:
    """Régression logistique sur quelques variables seulement (joueurs, bombe, temps)."""
    return Pipeline([
        ("features", FeatureBuilder()),
        ("select", ColumnSelector(columns)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
