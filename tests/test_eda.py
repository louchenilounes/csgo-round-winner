"""Tests des fonctions d'analyse exploratoire (src/eda.py) sur un petit tableau fait main."""

import pandas as pd
import pytest

from src.eda import add_differences, alive_matrix, time_profile, top_correlations, win_rate_by


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "map": pd.Categorical(["de_dust2", "de_dust2", "de_nuke", "de_nuke", "de_nuke", "de_dust2"]),
            "bomb_planted": pd.array([False, True, False, True, False, False], dtype="boolean"),
            "time_left": [170.0, 35.0, 120.0, 20.0, 60.0, 100.0],
            "ct_players_alive": [5.0, 2.0, 5.0, 1.0, 4.0, 5.0],
            "t_players_alive": [5.0, 4.0, 3.0, 3.0, 6.0, 5.0],  # 6 = anomalie
            "ct_health": [500.0, 150.0, 500.0, 80.0, 400.0, 500.0],
            "t_health": [500.0, 350.0, 250.0, 300.0, 600.0, 500.0],
            "ct_armor": [0.0, 100.0, 400.0, 50.0, 300.0, 200.0],
            "t_armor": [0.0, 300.0, 200.0, 250.0, 500.0, 100.0],
            "ct_money": [4000.0, 1000.0, 9000.0, 500.0, 3000.0, 2000.0],
            "t_money": [4000.0, 3000.0, 2000.0, 1500.0, 8000.0, 1000.0],
            "t_win": [0, 1, 0, 1, 1, 0],
        }
    )


def test_win_rate_by_map(small_df):
    out = win_rate_by(small_df, "map").set_index("map")
    assert out.loc["de_dust2", "t_win_rate"] == pytest.approx(1 / 3)
    assert out.loc["de_nuke", "t_win_rate"] == pytest.approx(2 / 3)
    assert out["n"].sum() == len(small_df)


def test_alive_matrix_excludes_anomalies(small_df):
    rates, counts = alive_matrix(small_df)
    assert 6 not in rates.columns  # la ligne à 6 joueurs T est exclue
    assert counts.sum().sum() == 5
    assert rates.loc[5, 5] == pytest.approx(0.0)
    assert rates.loc[2, 4] == pytest.approx(1.0)


def test_add_differences_sign(small_df):
    d = add_differences(small_df)
    assert d.loc[1, "alive_diff"] == 2  # 4 T contre 2 CT -> +2 pour les T
    assert d.loc[2, "money_diff"] == -7000  # 2000 - 9000


def test_time_profile_splits_bomb(small_df):
    out = time_profile(small_df, step=50)
    assert set(out["bomb_planted"]) == {False, True}
    assert out["n"].sum() == len(small_df)


def test_top_correlations_excludes_target(small_df):
    corr = top_correlations(small_df, n=5)
    assert "t_win" not in corr.index
    assert len(corr) == 5
