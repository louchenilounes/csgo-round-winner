"""Tests de l'interprétation SHAP (src/explain.py)."""

import numpy as np
import pytest

from src.explain import BASE, explain_row, global_importance, shap_values
from src.train import train_final_model
from tests.test_features import make_df


@pytest.fixture(scope="module")
def fitted():
    df = make_df(n=200, seed=1)
    model = train_final_model(df, {"n_estimators": 30, "min_child_samples": 5})
    return model, df


def test_contributions_add_up_to_prediction(fitted):
    model, df = fitted
    shap_df = shap_values(model, df)
    logit = shap_df.sum(axis=1).to_numpy()
    proba = model.predict_proba(df)[:, 1]
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), proba, rtol=1e-6)


def test_map_columns_are_grouped(fitted):
    model, df = fitted
    cols = shap_values(model, df).columns
    assert "map" in cols
    assert not any(c.startswith("map_") for c in cols)
    assert BASE in cols


def test_global_importance_sorted(fitted):
    model, df = fitted
    imp = global_importance(shap_values(model, df))
    assert BASE not in imp.index
    assert (imp.diff().dropna() <= 0).all()


def test_explain_row(fitted):
    model, df = fitted
    table = explain_row(model, df.iloc[[0]], top=5)
    assert len(table) == 6  # 5 variables + « autres variables »
    proba = model.predict_proba(df.iloc[[0]])[0, 1]
    assert table.attrs["proba_finale"] == pytest.approx(proba, rel=1e-6)
    assert 0 < table.attrs["proba_base"] < 1
