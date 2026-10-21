"""Étape 4 : création des variables (feature engineering) et pipeline scikit-learn.

Idées principales, tirées de l'analyse exploratoire (étape 2) :
  - les ÉCARTS entre équipes (T - CT) sont plus informatifs que les valeurs brutes ;
  - les ~70 colonnes d'armes sont presque toujours à 0 : on les regroupe par catégorie ;
  - l'effet du temps restant DÉPEND de la bombe : on sépare le temps « avant bombe »
    et le temps « après bombe » ;
  - la carte est une variable texte : on l'encode en one-hot.

Toutes les transformations sont dans un Pipeline scikit-learn. Ainsi, pendant
la validation croisée, rien n'est appris sur les données de validation.
Le FeatureBuilder ne fait que des calculs ligne par ligne (il n'apprend rien),
donc il ne peut pas créer de fuite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SIDES = ("ct", "t")

# Catégories d'armes de CS:GO.
WEAPON_GROUPS = {
    "rifle": ["ak47", "m4a4", "m4a1s", "aug", "sg553", "famas", "galilar"],
    "awp": ["awp"],
    "other_sniper": ["ssg08", "g3sg1", "scar20"],  # scout et snipers automatiques
    "smg": ["mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"],
    "heavy": ["nova", "xm1014", "mag7", "sawedoff", "m249", "negev"],
    "pistol": ["glock", "usps", "p2000", "p250", "fiveseven", "tec9", "cz75auto",
               "deagle", "r8revolver", "elite"],
}
# L'AWP est gardé seul : c'est l'arme la plus chère et la plus décisive du jeu.

GRENADES = ["flashbang", "smokegrenade", "hegrenade", "incendiarygrenade", "molotovgrenade", "decoygrenade"]

# Variables « de base » gardées telles quelles.
BASE_COLUMNS = ["time_left", "ct_score", "t_score", "ct_health", "t_health", "ct_armor", "t_armor",
                "ct_money", "t_money", "ct_helmets", "t_helmets", "ct_defuse_kits",
                "ct_players_alive", "t_players_alive"]


def weapon_columns(side: str, group: str) -> list[str]:
    return [f"{side}_weapon_{w}" for w in WEAPON_GROUPS[group]]


def grenade_columns(side: str) -> list[str]:
    return [f"{side}_grenade_{g}" for g in GRENADES]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Crée les nouvelles variables à partir des colonnes brutes.

    Paramètres
    ----------
    keep_raw_weapons : si True, garde aussi les colonnes d'armes et de grenades d'origine.
    """

    def __init__(self, keep_raw_weapons: bool = False):
        self.keep_raw_weapons = keep_raw_weapons

    def fit(self, X, y=None):
        return self  # aucun apprentissage : calculs ligne par ligne

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        # 1. Variables de base
        for col in BASE_COLUMNS:
            out[col] = X[col].astype(float)
        bomb = X["bomb_planted"].astype(bool).astype(float)
        out["bomb_planted"] = bomb

        # 2. Équipement par catégorie (nombre d'armes de chaque type par équipe)
        for side in SIDES:
            for group in WEAPON_GROUPS:
                out[f"{side}_{group}"] = X[weapon_columns(side, group)].sum(axis=1).astype(float)
            out[f"{side}_grenades"] = X[grenade_columns(side)].sum(axis=1).astype(float)
            # armes principales = tout sauf les pistolets
            out[f"{side}_primary"] = sum(out[f"{side}_{g}"] for g in WEAPON_GROUPS if g != "pistol")

        # 3. Écarts T - CT (positif = avantage T)
        for name in ["players_alive", "health", "armor", "money", "helmets", "grenades",
                     "primary", "rifle", "awp"]:
            out[f"{name}_diff"] = out[f"t_{name}"] - out[f"ct_{name}"]

        # 4. Moyennes par joueur vivant (0 si plus personne)
        for side in SIDES:
            alive = out[f"{side}_players_alive"]
            out[f"{side}_health_per_player"] = np.where(alive > 0, out[f"{side}_health"] / alive.replace(0, 1), 0.0)
            out[f"{side}_armor_per_player"] = np.where(alive > 0, out[f"{side}_armor"] / alive.replace(0, 1), 0.0)

        # 5. Temps et bombe : l'effet du temps s'inverse quand la bombe est posée
        out["time_left_no_bomb"] = out["time_left"] * (1 - bomb)
        out["time_left_bomb"] = out["time_left"] * bomb

        # 6. Contexte du match
        out["round_number"] = out["ct_score"] + out["t_score"]
        out["score_diff"] = out["t_score"] - out["ct_score"]

        # 7. Carte (encodée plus loin par le ColumnTransformer)
        out["map"] = X["map"].astype(str)

        if self.keep_raw_weapons:
            raw = [c for c in X.columns if "_weapon_" in c or "_grenade_" in c]
            out = pd.concat([out, X[raw].astype(float)], axis=1)

        return out


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Encode la carte en one-hot ; met les variables numériques à l'échelle si demandé.

    La mise à l'échelle n'est utile que pour les modèles linéaires ou à distances
    (régression logistique, KNN), pas pour les arbres.
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("map", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["map"])],
        remainder=numeric,
        verbose_feature_names_out=False,
    )


def build_pipeline(model, scale: bool = False, keep_raw_weapons: bool = False) -> Pipeline:
    """Pipeline complet : création des variables -> encodage -> modèle."""
    return Pipeline([
        ("features", FeatureBuilder(keep_raw_weapons=keep_raw_weapons)),
        ("prep", build_preprocessor(scale=scale)),
        ("model", model),
    ])


class RawColumns(BaseEstimator, TransformerMixin):
    """Sélectionne les colonnes brutes du dataset (sans création de variables), pour comparaison."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = BASE_COLUMNS + [c for c in X.columns if "_weapon_" in c or "_grenade_" in c]
        out = X[cols].astype(float)
        out["bomb_planted"] = X["bomb_planted"].astype(bool).astype(float)
        out["map"] = X["map"].astype(str)
        return out


def build_raw_pipeline(model, scale: bool = False) -> Pipeline:
    """Même pipeline, mais avec les variables brutes uniquement (référence)."""
    return Pipeline([
        ("features", RawColumns()),
        ("prep", build_preprocessor(scale=scale)),
        ("model", model),
    ])
