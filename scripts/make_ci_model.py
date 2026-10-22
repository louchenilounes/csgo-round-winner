"""Crée un petit modèle sur des données synthétiques, au bon format.

Utilisé uniquement par l'intégration continue, pour vérifier que l'image Docker
se construit et que l'API répond. Le vrai modèle (entraîné sur les données OpenML)
n'est pas versionné ; il se crée avec `python -m src.train`.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import GRENADES, WEAPON_GROUPS, build_pipeline  # noqa: E402
from src.serving import MAPS, MODEL_PATH  # noqa: E402


def synthetic_data(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "time_left": rng.uniform(0, 175, n),
            "ct_score": rng.integers(0, 15, n).astype(float),
            "t_score": rng.integers(0, 15, n).astype(float),
            "map": pd.Categorical(rng.choice(MAPS, n), categories=MAPS),
            "bomb_planted": pd.array(rng.random(n) < 0.2, dtype="boolean"),
            "ct_defuse_kits": rng.integers(0, 6, n).astype(float),
        }
    )
    for side in ("ct", "t"):
        alive = rng.integers(0, 6, n)
        df[f"{side}_players_alive"] = alive.astype(float)
        df[f"{side}_health"] = alive * rng.uniform(20, 100, n)
        df[f"{side}_armor"] = alive * rng.uniform(0, 100, n)
        df[f"{side}_money"] = rng.uniform(0, 30000, n)
        df[f"{side}_helmets"] = rng.integers(0, 6, n).astype(float)
        for weapons in WEAPON_GROUPS.values():
            for w in weapons:
                df[f"{side}_weapon_{w}"] = rng.integers(0, 2, n).astype(float)
        for g in GRENADES:
            df[f"{side}_grenade_{g}"] = rng.integers(0, 3, n).astype(float)
    logit = 0.8 * (df["t_players_alive"] - df["ct_players_alive"]) + 1.2 * df["bomb_planted"].astype(float)
    df["t_win"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


if __name__ == "__main__":
    data = synthetic_data()
    model = build_pipeline(LGBMClassifier(n_estimators=50, random_state=0, verbose=-1))
    model.fit(data, data["t_win"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle de CI sauvegardé : {MODEL_PATH}")
