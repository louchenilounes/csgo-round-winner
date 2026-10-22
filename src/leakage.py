"""Mesure de la fuite de données.

Le même modèle (Random Forest, comme dans la première version du projet)
est entraîné avec trois façons de découper les données :

  - « Aléatoire (ancien) » : les lignes sont mélangées (protocole de la première version) ;
  - « Par round »          : toutes les photos d'un round sont du même côté ;
  - « Par match »          : toutes les photos d'un match sont du même côté (le plus strict).

Si le score baisse beaucoup quand on passe au découpage par groupe,
c'est la preuve que l'ancien score profitait de la fuite.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import SEED, TARGET

# Les 20 variables de la première version (corrélation > 0,15 avec t_win).
OLD_FEATURES = [
    "ct_armor",
    "ct_helmets",
    "t_helmets",
    "ct_defuse_kits",
    "t_armor",
    "ct_grenade_flashbang",
    "ct_players_alive",
    "ct_grenade_smokegrenade",
    "ct_weapon_awp",
    "t_weapon_ak47",
    "ct_health",
    "bomb_planted",
    "ct_weapon_m4a4",
    "ct_grenade_hegrenade",
    "ct_grenade_incendiarygrenade",
    "ct_weapon_ak47",
    "t_grenade_flashbang",
    "ct_weapon_sg553",
    "t_weapon_sg553",
    "ct_weapon_usps",
]

NOT_FEATURES = {TARGET, "round_winner", "match_id", "round_id", "snapshot_idx", "split", "label_fixed"}


def all_features(df: pd.DataFrame) -> list[str]:
    """Toutes les variables utilisables (sauf la cible, les identifiants et le texte)."""
    return [c for c in df.columns if c not in NOT_FEATURES and c != "map"]


def make_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Matrice X numérique (booléens -> 0/1)."""
    return df[features].astype(float)


def split_indices(df: pd.DataFrame, how: str, test_size: float = 0.2, seed: int = SEED):
    """Indices train/test selon le type de découpage : 'random', 'round' ou 'match'."""
    idx = np.arange(len(df))
    if how == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed)
    group = {"round": "round_id", "match": "match_id"}[how]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(idx, groups=df[group]))


def evaluate_split(
    df: pd.DataFrame, features: list[str], how: str, n_estimators: int = 100, seed: int = SEED
) -> dict:
    """Entraîne un Random Forest et renvoie ses scores sur le train et le test."""
    tr, te = split_indices(df, how, seed=seed)
    X, y = make_X(df, features), df[TARGET].to_numpy()
    start = time.time()
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    model.fit(X.iloc[tr], y[tr])
    proba = model.predict_proba(X.iloc[te])[:, 1]
    return {
        "train_accuracy": accuracy_score(y[tr], model.predict(X.iloc[tr])),
        "test_accuracy": accuracy_score(y[te], (proba >= 0.5).astype(int)),
        "test_roc_auc": roc_auc_score(y[te], proba),
        "n_test": len(te),
        "seconds": time.time() - start,
    }


def share_of_test_rounds_seen_in_train(df: pd.DataFrame, how: str, seed: int = SEED) -> float:
    """Part des lignes de test dont le round apparaît aussi dans le train (= la fuite)."""
    tr, te = split_indices(df, how, seed=seed)
    seen = set(df["round_id"].iloc[tr])
    return float(df["round_id"].iloc[te].isin(seen).mean())


def compare_splits(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    splits=("random", "round", "match"),
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Tableau de comparaison : une ligne par (jeu de variables, découpage)."""
    names = {"random": "Aléatoire (ancien)", "round": "Par round", "match": "Par match"}
    rows = []
    for fs_name, feats in feature_sets.items():
        for how in splits:
            res = evaluate_split(df, feats, how, n_estimators=n_estimators)
            rows.append(
                {
                    "variables": fs_name,
                    "découpage": names[how],
                    "rounds du test vus en train": share_of_test_rounds_seen_in_train(df, how),
                    **res,
                }
            )
            print(
                f"  {fs_name:<22} | {names[how]:<18} | accuracy test {res['test_accuracy']:.3f} "
                f"| AUC {res['test_roc_auc']:.3f} | {res['seconds']:.0f} s"
            )
    return pd.DataFrame(rows)
