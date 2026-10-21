"""Tests des modèles de référence (src/baselines.py)."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import AliveRule, FrequencyTable, simple_logistic_regression
from tests.test_features import make_df


@pytest.fixture
def situations():
    return pd.DataFrame({
        "t_players_alive": [5, 3, 4, 4, 2, 2],
        "ct_players_alive": [4, 5, 4, 4, 2, 3],
        "bomb_planted": pd.array([False, False, False, True, True, False], dtype="boolean"),
    })


def test_alive_rule(situations):
    proba = AliveRule().fit(situations, [0] * 6).predict_proba(situations)[:, 1]
    assert proba.tolist() == [1.0, 0.0, 0.5, 0.5, 0.5, 0.0]


def test_alive_rule_bomb_tiebreak(situations):
    proba = AliveRule(bomb_tiebreak=True).fit(situations, [0] * 6).predict_proba(situations)[:, 1]
    # égalités : sans bombe -> CT (0), avec bombe -> T (1)
    assert proba.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_frequency_table_learns_rates():
    X = pd.DataFrame({"t_players_alive": [5] * 4 + [4] * 4, "ct_players_alive": [5] * 8,
                      "bomb_planted": pd.array([False] * 8, dtype="boolean")})
    y = [1, 1, 1, 0, 0, 0, 0, 1]  # écart 0 : 75 % ; écart -1 : 25 %
    table = FrequencyTable(min_count=1).fit(X, y)
    proba = table.predict_proba(X)[:, 1]
    assert proba[0] == pytest.approx(0.75)
    assert proba[-1] == pytest.approx(0.25)


def test_frequency_table_falls_back_to_global_rate():
    X = pd.DataFrame({"t_players_alive": [5, 5, 5, 1], "ct_players_alive": [5, 5, 5, 5],
                      "bomb_planted": pd.array([False] * 4, dtype="boolean")})
    y = [1, 0, 1, 0]
    table = FrequencyTable(min_count=2).fit(X, y)
    unseen = pd.DataFrame({"t_players_alive": [3], "ct_players_alive": [1],
                           "bomb_planted": pd.array([True], dtype="boolean")})
    assert table.predict_proba(unseen)[0, 1] == pytest.approx(0.5)  # taux global
    # l'écart -4 n'a qu'une ligne (< min_count) : taux global aussi
    assert table.predict_proba(X.iloc[[3]])[0, 1] == pytest.approx(0.5)


def test_frequency_table_with_bomb_separates_situations():
    X = pd.DataFrame({"t_players_alive": [5] * 4, "ct_players_alive": [5] * 4,
                      "bomb_planted": pd.array([True, True, False, False], dtype="boolean")})
    table = FrequencyTable(use_bomb=True, min_count=1).fit(X, [1, 1, 0, 0])
    assert table.predict_proba(X)[:, 1].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_simple_logistic_regression_runs():
    df = make_df()
    pipe = simple_logistic_regression().fit(df, df["t_win"])
    proba = pipe.predict_proba(df)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert pipe.named_steps["model"].coef_.shape == (1, 4)
