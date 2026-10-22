"""Tests du suivi MLflow (src/tracking.py), dans un dossier temporaire."""

import mlflow
import numpy as np
import pytest

from src.evaluation import grouped_cv
from src.features import build_pipeline
from src.tracking import REGISTERED_MODEL_NAME, log_cv_run, log_final_model, runs_table, setup
from src.train import train_final_model
from src.tuning import make_lgbm
from tests.test_features import make_df


@pytest.fixture
def tracking(tmp_path):
    setup(root=tmp_path, experiment="test")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_log_cv_run(tracking):
    df = make_df(n=60)
    folds = grouped_cv(build_pipeline(make_lgbm({"n_estimators": 10, "min_child_samples": 5})), df, n_folds=3)
    log_cv_run("essai", {"modele": "LightGBM"}, folds)
    table = runs_table()
    assert table["runName"].tolist() == ["essai"]
    assert table["cv_log_loss"].iloc[0] == pytest.approx(folds["log_loss"].mean())


def test_log_and_reload_final_model(tracking):
    df = make_df(n=60)
    params = {"n_estimators": 10, "min_child_samples": 5}
    model = train_final_model(df, params)
    X = df.drop(columns=["t_win", "match_id"])
    log_final_model(model, params, {"accuracy": 0.7, "log_loss": 0.5}, X)
    reloaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/1")
    np.testing.assert_allclose(reloaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
    table = runs_table()
    assert table.loc[table["runName"] == "Modèle final", "test_accuracy"].iloc[0] == pytest.approx(0.7)
