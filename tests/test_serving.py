"""Tests de la mise à disposition du modèle (src/serving.py) et de l'API (app/api.py)."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.features import FeatureBuilder
from src.serving import RoundState, TeamState, predict, to_model_input
from src.train import save_model, train_final_model
from tests.test_features import make_df


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    df = make_df(n=200, seed=2)
    model = train_final_model(df, {"n_estimators": 30, "min_child_samples": 5})
    folder = tmp_path_factory.mktemp("model")
    return save_model(model, {}, len(df), folder / "m.joblib", folder / "m.json")


def test_model_input_has_all_raw_columns():
    X = to_model_input(RoundState())
    raw = make_df(n=5).drop(columns=["t_win", "match_id"])
    assert set(raw.columns) <= set(X.columns)
    assert len(X) == 1


def test_categories_are_preserved():
    state = RoundState(t=TeamState(rifles=3, awp=1, smgs=1, pistols=2, grenades=6))
    feats = FeatureBuilder().transform(to_model_input(state)).iloc[0]
    assert feats["t_rifle"] == 3 and feats["t_awp"] == 1 and feats["t_smg"] == 1
    assert feats["t_pistol"] == 2 and feats["t_grenades"] == 6
    assert feats["t_primary"] == 5


def test_defaults_follow_players_alive():
    team = TeamState(players_alive=2, awp=1)
    assert team.health == 200 and team.armor == 200 and team.helmets == 2
    assert team.rifles == 1 and team.pistols == 2


@pytest.mark.parametrize("kwargs", [
    {"t": {"players_alive": 2, "health": 300}},   # santé > 100 par joueur
    {"bomb_planted": True, "time_left": 90},      # bombe posée avec plus de 40 s
    {"ct": {"players_alive": 1, "rifles": 3, "health": 100, "armor": 100}},  # 3 fusils pour 1 joueur
])
def test_invalid_states_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        RoundState(**kwargs)


def test_predict_output(model_path):
    from src.train import load_model
    out = predict(load_model(model_path), RoundState())
    assert out["proba_t_win"] + out["proba_ct_win"] == pytest.approx(1)
    assert out["favori"] in {"T", "CT"}
    assert len(out["facteurs"]) == 5


def test_api(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from app.api import app
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        r = client.post("/predict", json={"map": "de_nuke", "time_left": 30, "bomb_planted": True})
        assert r.status_code == 200
        assert 0 <= r.json()["proba_t_win"] <= 1
        bad = client.post("/predict", json={"map": "de_unknown"})
        assert bad.status_code == 422
