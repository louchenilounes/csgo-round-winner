"""Tests du module src/data.py.

Ils n'utilisent pas Internet : on fabrique un petit fichier ARFF de test
dans un dossier temporaire (tmp_path est fourni automatiquement par pytest).
"""

import pandas as pd
import pytest

from src.data import check_data, clean_types, read_arff

MINI_ARFF = """% Petit fichier de test
@RELATION csgo_test

@ATTRIBUTE time_left NUMERIC
@ATTRIBUTE ct_score NUMERIC
@ATTRIBUTE t_score NUMERIC
@ATTRIBUTE map {de_dust2,de_inferno,de_mirage}
@ATTRIBUTE bomb_planted {False,True}
@ATTRIBUTE ct_health NUMERIC
@ATTRIBUTE 't_health' NUMERIC
@ATTRIBUTE round_winner {CT,T}

@DATA
175.0,0.0,0.0,de_dust2,False,500.0,500.0,CT
156.0,0.0,0.0,de_dust2,False,500.0,500.0,CT
96.0,0.0,0.0,de_dust2,True,391.0,?,T
175.0,0.0,0.0,de_dust2,False,500.0,500.0,CT
% commentaire au milieu
40.0,1.0,0.0,de_inferno,True,200.0,300.0,T
"""


@pytest.fixture
def mini_arff(tmp_path):
    path = tmp_path / "mini.arff"
    path.write_text(MINI_ARFF, encoding="utf-8")
    return path


def test_read_arff_columns_and_rows(mini_arff):
    df = read_arff(mini_arff)
    assert list(df.columns) == [
        "time_left",
        "ct_score",
        "t_score",
        "map",
        "bomb_planted",
        "ct_health",
        "t_health",
        "round_winner",
    ]
    assert len(df) == 5  # les commentaires et lignes vides sont ignorés


def test_question_mark_becomes_missing(mini_arff):
    df = read_arff(mini_arff)
    assert df["t_health"].isna().sum() == 1


def test_clean_types_and_target(mini_arff):
    df = clean_types(read_arff(mini_arff))
    assert str(df["map"].dtype) == "category"
    assert str(df["bomb_planted"].dtype) == "boolean"
    assert df["bomb_planted"].tolist() == [False, False, True, False, True]
    assert pd.api.types.is_float_dtype(df["time_left"])
    # t_win = 1 seulement quand les T gagnent
    assert df["t_win"].tolist() == [0, 0, 1, 0, 1]


def test_clean_types_rejects_unknown_winner(mini_arff):
    df = read_arff(mini_arff)
    df.loc[0, "round_winner"] = "DRAW"
    with pytest.raises(ValueError):
        clean_types(df)


def test_check_data_report(mini_arff):
    report = check_data(clean_types(read_arff(mini_arff)))
    assert report["n_rows"] == 5
    assert report["n_missing_values"] == 1
    assert report["columns_with_missing"] == ["t_health"]
    assert report["n_duplicated_rows"] == 1  # la ligne 4 est identique à la ligne 1
    assert report["share_t_win"] == pytest.approx(0.4)
    assert report["n_maps"] == 2
