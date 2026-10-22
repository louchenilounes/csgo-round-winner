"""Tests de la création des variables (src/features.py) et de l'évaluation (src/evaluation.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation import compute_metrics, grouped_cv, summarize_cv
from src.features import GRENADES, WEAPON_GROUPS, FeatureBuilder, build_pipeline, build_raw_pipeline

ALL_WEAPONS = [
    "ak47",
    "aug",
    "awp",
    "bizon",
    "cz75auto",
    "deagle",
    "elite",
    "famas",
    "fiveseven",
    "g3sg1",
    "galilar",
    "glock",
    "m249",
    "m4a1s",
    "m4a4",
    "mac10",
    "mag7",
    "mp5sd",
    "mp7",
    "mp9",
    "negev",
    "nova",
    "p2000",
    "p250",
    "p90",
    "r8revolver",
    "sawedoff",
    "scar20",
    "sg553",
    "ssg08",
    "tec9",
    "ump45",
    "usps",
    "xm1014",
]


def make_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "time_left": rng.uniform(0, 175, n),
            "ct_score": rng.integers(0, 15, n).astype(float),
            "t_score": rng.integers(0, 15, n).astype(float),
            "map": pd.Categorical(rng.choice(["de_dust2", "de_nuke", "de_inferno"], n)),
            "bomb_planted": pd.array(rng.random(n) < 0.3, dtype="boolean"),
            "ct_health": rng.uniform(0, 500, n),
            "t_health": rng.uniform(0, 500, n),
            "ct_armor": rng.uniform(0, 500, n),
            "t_armor": rng.uniform(0, 500, n),
            "ct_money": rng.uniform(0, 30000, n),
            "t_money": rng.uniform(0, 30000, n),
            "ct_helmets": rng.integers(0, 6, n).astype(float),
            "t_helmets": rng.integers(0, 6, n).astype(float),
            "ct_defuse_kits": rng.integers(0, 6, n).astype(float),
            "ct_players_alive": rng.integers(0, 6, n).astype(float),
            "t_players_alive": rng.integers(0, 6, n).astype(float),
        }
    )
    for side in ("ct", "t"):
        for w in ALL_WEAPONS:
            df[f"{side}_weapon_{w}"] = rng.integers(0, 2, n).astype(float)
        for g in GRENADES:
            df[f"{side}_grenade_{g}"] = rng.integers(0, 3, n).astype(float)
    df["t_win"] = (df["t_players_alive"] > df["ct_players_alive"]).astype(int)
    df.loc[:1, "t_win"] = [0, 1]  # les deux classes présentes
    df["match_id"] = np.arange(n) // 4
    return df


def test_every_weapon_in_exactly_one_group():
    grouped = [w for ws in WEAPON_GROUPS.values() for w in ws]
    assert sorted(grouped) == sorted(ALL_WEAPONS)
    assert len(grouped) == len(set(grouped))


def test_feature_builder_values():
    df = make_df()
    out = FeatureBuilder().transform(df)
    row = df.iloc[0]
    assert out.loc[0, "players_alive_diff"] == row["t_players_alive"] - row["ct_players_alive"]
    assert out.loc[0, "t_rifle"] == sum(row[f"t_weapon_{w}"] for w in WEAPON_GROUPS["rifle"])
    assert out.loc[0, "round_number"] == row["ct_score"] + row["t_score"]
    # temps séparé selon la bombe : une seule des deux colonnes est non nulle
    assert ((out["time_left_bomb"] == 0) | (out["time_left_no_bomb"] == 0)).all()


def test_no_nan_or_inf_when_no_player_alive():
    df = make_df()
    df.loc[:, "t_players_alive"] = 0.0
    out = FeatureBuilder().transform(df).drop(columns="map")
    assert np.isfinite(out.to_numpy()).all()
    assert (out["t_health_per_player"] == 0).all()


def test_feature_builder_does_not_use_target():
    df = make_df()
    a = FeatureBuilder().transform(df)
    b = FeatureBuilder().transform(df.drop(columns="t_win"))
    pd.testing.assert_frame_equal(a, b)


def test_keep_raw_weapons_adds_columns():
    df = make_df()
    small = FeatureBuilder().transform(df)
    big = FeatureBuilder(keep_raw_weapons=True).transform(df)
    assert big.shape[1] == small.shape[1] + 2 * (len(ALL_WEAPONS) + len(GRENADES))


@pytest.mark.parametrize("builder", [build_pipeline, build_raw_pipeline])
def test_pipelines_fit_and_predict_probabilities(builder):
    df = make_df()
    pipe = builder(LogisticRegression(max_iter=500), scale=True)
    pipe.fit(df, df["t_win"])
    proba = pipe.predict_proba(df)[:, 1]
    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_grouped_cv_and_summary():
    df = make_df(n=60)
    folds = grouped_cv(build_pipeline(LogisticRegression(max_iter=500), scale=True), df, n_folds=3)
    assert len(folds) == 3
    summary = summarize_cv({"logreg": folds})
    assert 0 <= summary.loc[0, "accuracy"] <= 1


def test_compute_metrics_perfect_prediction():
    m = compute_metrics(np.array([0, 1, 1, 0]), np.array([0.0, 1.0, 1.0, 0.0]))
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
    assert m["log_loss"] < 1e-4 and m["brier"] < 1e-8
