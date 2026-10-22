"""Mise à disposition du modèle : de la description d'une situation de jeu à une prédiction.

Le modèle attend les 96 colonnes brutes du dataset, dont 68 colonnes d'armes et 12 de
grenades. Pour une démo, on décrit plutôt l'équipement par catégorie (fusils, AWP, SMG...).
Comme `FeatureBuilder` ne garde que les totaux par catégorie, il suffit de placer chaque
total sur une arme représentative de sa catégorie : la prédiction est identique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.config import MODELS_DIR
from src.explain import explain_row
from src.features import GRENADES, WEAPON_GROUPS

# Le module ne dépend que de scikit-learn et LightGBM (pas du code d'entraînement),
# pour garder l'image Docker légère.
MODEL_PATH = MODELS_DIR / "lightgbm_pipeline.joblib"


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)

MAPS = ["de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train", "de_vertigo", "de_cache"]

# Arme représentative de chaque catégorie, par camp.
REPRESENTATIVE = {
    "ct": {"rifle": "m4a4", "awp": "awp", "other_sniper": "ssg08", "smg": "mp9", "heavy": "nova", "pistol": "usps"},
    "t": {"rifle": "ak47", "awp": "awp", "other_sniper": "ssg08", "smg": "mac10", "heavy": "nova", "pistol": "glock"},
}


class TeamState(BaseModel):
    """État d'une équipe. Santé et armure sont des totaux d'équipe (100 max par joueur).

    Les champs laissés vides prennent une valeur cohérente avec le nombre de joueurs
    vivants : santé et armure pleines, un casque, un fusil et un pistolet par joueur.
    """

    players_alive: int = Field(5, ge=0, le=5)
    health: float | None = Field(None, ge=0, le=500)
    armor: float | None = Field(None, ge=0, le=500)
    money: float = Field(4000, ge=0, le=80000)
    helmets: int | None = Field(None, ge=0, le=5)
    rifles: int | None = Field(None, ge=0, le=5)
    awp: int = Field(0, ge=0, le=5)
    other_snipers: int = Field(0, ge=0, le=5)
    smgs: int = Field(0, ge=0, le=5)
    heavy: int = Field(0, ge=0, le=5)
    pistols: int | None = Field(None, ge=0, le=5)
    grenades: int = Field(8, ge=0, le=20)

    @model_validator(mode="after")
    def fill_and_check(self):
        n = self.players_alive
        if self.health is None:
            self.health = 100.0 * n
        if self.armor is None:
            self.armor = 100.0 * n
        if self.helmets is None:
            self.helmets = n
        if self.pistols is None:
            self.pistols = n
        if self.rifles is None:
            self.rifles = max(0, n - self.awp - self.other_snipers - self.smgs - self.heavy)
        if self.health > 100 * n:
            raise ValueError("la santé d'équipe ne peut pas dépasser 100 par joueur vivant")
        if self.armor > 100 * n:
            raise ValueError("l'armure d'équipe ne peut pas dépasser 100 par joueur vivant")
        if self.rifles + self.awp + self.other_snipers + self.smgs + self.heavy > n:
            raise ValueError("plus d'armes principales que de joueurs vivants")
        return self


class RoundState(BaseModel):
    """Situation de jeu à un instant du round."""

    map: Literal["de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass",
                 "de_train", "de_vertigo", "de_cache"] = "de_dust2"
    time_left: float = Field(115, ge=0, le=175, description="secondes restantes")
    bomb_planted: bool = False
    ct_score: int = Field(0, ge=0, le=30)
    t_score: int = Field(0, ge=0, le=30)
    ct_defuse_kits: int = Field(0, ge=0, le=5)
    ct: TeamState = TeamState()
    t: TeamState = TeamState()

    @model_validator(mode="after")
    def bomb_timer(self):
        if self.bomb_planted and self.time_left > 40:
            raise ValueError("bombe posée : le chrono ne peut pas dépasser 40 s")
        return self


def to_model_input(state: RoundState) -> pd.DataFrame:
    """Une ligne au format des données brutes (96 colonnes), prête pour le pipeline."""
    row = {
        "time_left": state.time_left, "ct_score": state.ct_score, "t_score": state.t_score,
        "map": state.map, "bomb_planted": state.bomb_planted, "ct_defuse_kits": state.ct_defuse_kits,
    }
    for side, team in (("ct", state.ct), ("t", state.t)):
        row.update({
            f"{side}_health": team.health, f"{side}_armor": team.armor, f"{side}_money": team.money,
            f"{side}_helmets": team.helmets, f"{side}_players_alive": team.players_alive,
        })
        for group, weapons in WEAPON_GROUPS.items():
            for w in weapons:
                row[f"{side}_weapon_{w}"] = 0.0
        counts = {"rifle": team.rifles, "awp": team.awp, "other_sniper": team.other_snipers,
                  "smg": team.smgs, "heavy": team.heavy, "pistol": team.pistols}
        for group, n in counts.items():
            row[f"{side}_weapon_{REPRESENTATIVE[side][group]}"] = float(n)
        for g in GRENADES:
            row[f"{side}_grenade_{g}"] = 0.0
        row[f"{side}_grenade_flashbang"] = float(team.grenades)
    df = pd.DataFrame([row])
    df["map"] = pd.Categorical(df["map"], categories=MAPS)
    df["bomb_planted"] = df["bomb_planted"].astype("boolean")
    return df


def predict(model, state: RoundState, top: int = 5) -> dict:
    """Probabilités de victoire et principaux facteurs (contributions SHAP)."""
    X = to_model_input(state)
    proba_t = float(model.predict_proba(X)[0, 1])
    table = explain_row(model, X, top=top)
    factors = [{"variable": str(v), "contribution": round(float(c), 4)}
               for v, c in zip(table["variable"], table["contribution"]) if "autres" not in str(v)]
    return {
        "proba_t_win": round(proba_t, 4),
        "proba_ct_win": round(1 - proba_t, 4),
        "favori": "T" if proba_t >= 0.5 else "CT",
        "facteurs": factors,
    }
