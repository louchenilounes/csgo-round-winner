"""Tests des outils d'évaluation finale (src/final_eval.py)."""

import numpy as np
import pandas as pd
import pytest

from src.final_eval import bootstrap_by_group, calibration_table, expected_calibration_error, metrics_by


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)  # parfaitement calibré par construction
    groups = np.arange(2000) // 20
    return y, p, groups


def test_bootstrap_interval_contains_point_estimate(predictions):
    y, p, g = predictions
    ci = bootstrap_by_group(y, p, g, n_boot=200)
    assert set(ci.index) == {"accuracy", "roc_auc", "log_loss", "brier"}
    assert (ci["borne_basse"] <= ci["valeur"]).all()
    assert (ci["valeur"] <= ci["borne_haute"]).all()


def test_calibration_of_calibrated_predictions(predictions):
    y, p, _ = predictions
    table = calibration_table(y, p, n_bins=5)
    assert table["n"].sum() == len(y)
    assert expected_calibration_error(y, p, n_bins=5) < 0.05


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 500)
    p = np.where(y == 1, 0.99, 0.95)  # annonce ~97 % alors que la fréquence est 50 %
    assert expected_calibration_error(y, p) > 0.4


def test_metrics_by_group():
    df = pd.DataFrame(
        {"y": [0, 1, 0, 1, 1, 1], "p": [0.2, 0.8, 0.3, 0.6, 0.9, 0.7], "map": ["a", "a", "b", "b", "c", "c"]}
    )
    out = metrics_by(df, "y", "p", "map")
    assert set(out["groupe"]) == {"a", "b"}  # la carte « c » n'a qu'une classe : ignorée
    assert out.loc[out["groupe"] == "a", "accuracy"].iloc[0] == 1.0
