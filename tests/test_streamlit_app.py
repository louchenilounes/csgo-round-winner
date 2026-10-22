"""Tests de l'interface Streamlit (exécutée sans navigateur avec AppTest).

Ces tests nécessitent le modèle entraîné (models/lightgbm_pipeline.joblib) ;
ils sont ignorés s'il est absent.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.serving import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="modèle non entraîné")
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def probability_t(at: AppTest) -> float:
    return float(at.metric[0].value.rstrip("%")) / 100


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert 0 <= probability_t(at) <= 1


def test_scenarios_behave_like_the_game():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].select("Bombe posée, 2 T contre 3 CT").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state["bomb"] is True
    with_bomb = probability_t(at)
    at.sidebar.toggle(key="bomb").set_value(False).run()
    assert at.session_state["time_left"] == 30  # le chrono ne doit pas être réinitialisé
    assert probability_t(at) < with_bomb  # sans la bombe, les T ont moins de chances

    at.sidebar.selectbox[0].select("Round d'éco des T").run()
    at.sidebar.button[0].click().run()
    assert probability_t(at) < 0.5  # T sans armes principales contre CT équipés


def test_bomb_clamps_timer():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.slider(key="time_left").set_value(150).run()
    at.sidebar.toggle(key="bomb").set_value(True).run()
    assert not at.exception
    assert at.session_state["time_left"] <= 40


def test_fewer_players_clamps_health():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.slider(key="t_players_alive").set_value(2).run()
    assert not at.exception
    assert at.number_input(key="t_health").value <= 200
