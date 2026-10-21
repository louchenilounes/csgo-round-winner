"""Tests de l'optimisation (src/tuning.py) et de l'entraînement final (src/train.py)."""

import numpy as np

from src.train import load_model, save_model, train_final_model
from src.tuning import (DEFAULT_PARAMS, load_best_params, make_lgbm, save_best_params,
                        trials_table, tune_lightgbm)
from tests.test_features import make_df


def test_make_lgbm_uses_defaults_and_overrides():
    model = make_lgbm({"num_leaves": 7})
    assert model.get_params()["num_leaves"] == 7
    assert model.get_params()["learning_rate"] == DEFAULT_PARAMS["learning_rate"]


def test_tuning_runs_and_first_trial_is_default():
    df = make_df(n=80)
    study = tune_lightgbm(df, n_trials=3, n_folds=2, show_progress=False)
    table = trials_table(study)
    assert len(table) == 3
    assert {"accuracy", "roc_auc", "brier", "log_loss"} <= set(table.columns)
    first = study.trials[0].params
    assert first["num_leaves"] == DEFAULT_PARAMS["num_leaves"]


def test_best_params_roundtrip(tmp_path):
    path = save_best_params({"num_leaves": 12, "learning_rate": 0.03}, tmp_path / "p.json")
    assert load_best_params(path) == {"num_leaves": 12, "learning_rate": 0.03}


def test_train_save_and_reload(tmp_path):
    df = make_df(n=60)
    params = {"n_estimators": 20, "min_child_samples": 5}
    model = train_final_model(df, params)
    path = save_model(model, params, len(df), tmp_path / "m.joblib", tmp_path / "m.json")
    reloaded = load_model(path)
    np.testing.assert_allclose(model.predict_proba(df), reloaded.predict_proba(df))
