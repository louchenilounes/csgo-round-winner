"""Tests de la reconstruction des rounds (src/rounds.py) et de la mesure de fuite (src/leakage.py).

On fabrique une petite suite de photos dont on connaît la bonne réponse.
Elle reproduit les particularités du vrai dataset :
  - la 1re photo d'un round porte le gagnant du round PRÉCÉDENT (erreur à corriger) ;
  - la bombe posée remet le chrono à 40 s (même round) ;
  - une ligne impossible avec 6 joueurs.

match 0 (de_dust2)
  round 0 : score 0-0, étiqueté CT
  round 1 : score 0-1, 1re photo étiquetée CT (gagnant précédent), les suivantes T -> vrai gagnant T
  round 2 : score 0-2, 1re photo étiquetée T (gagnant précédent), la suivante CT -> vrai gagnant CT
match 1 (de_dust2, le score repart à 0 -> nouveau match)
  round 3 : score 0-0, gagnant T
match 2 (de_nuke -> nouvelle carte)
  round 4 : score 0-0, gagnant CT
"""

import pandas as pd
import pytest

from src.leakage import share_of_test_rounds_seen_in_train
from src.rounds import (add_round_ids, check_rounds, drop_impossible_rows, fix_labels,
                        make_split, score_based_winner)

ROWS = [
    # map,       ct, t, time,  bomb,  winner, ct_alive, t_alive
    ("de_dust2", 0, 0, 175.0, False, "CT", 5, 5),
    ("de_dust2", 0, 0, 155.0, False, "CT", 5, 5),
    ("de_dust2", 0, 0, 135.0, False, "CT", 5, 4),
    ("de_dust2", 0, 1, 174.9, False, "CT", 5, 5),  # 1re photo : gagnant du round précédent
    ("de_dust2", 0, 1, 50.0, False, "T", 4, 5),
    ("de_dust2", 0, 1, 20.0, False, "T", 3, 5),
    ("de_dust2", 0, 1, 40.0, True, "T", 3, 5),    # bombe posée : chrono remonte, même round
    ("de_dust2", 0, 2, 174.9, False, "T", 5, 5),  # 1re photo : gagnant précédent (T)
    ("de_dust2", 0, 2, 120.0, False, "CT", 5, 6), # 6 joueurs : ligne impossible
    ("de_dust2", 0, 2, 100.0, False, "CT", 5, 5),
    ("de_dust2", 0, 0, 175.0, False, "T", 5, 5),
    ("de_dust2", 0, 0, 100.0, False, "T", 5, 5),
    ("de_nuke", 0, 0, 175.0, False, "CT", 5, 5),
]


@pytest.fixture
def seq():
    df = pd.DataFrame(ROWS, columns=["map", "ct_score", "t_score", "time_left", "bomb_planted",
                                     "round_winner", "ct_players_alive", "t_players_alive"])
    df["map"] = df["map"].astype("category")
    df["round_winner"] = df["round_winner"].astype("category")
    df["bomb_planted"] = df["bomb_planted"].astype("boolean")
    df["t_win"] = (df["round_winner"] == "T").astype(int)
    return df


@pytest.fixture
def rounds(seq):
    return fix_labels(add_round_ids(drop_impossible_rows(seq)))


def test_drop_impossible_rows(seq):
    clean = drop_impossible_rows(seq)
    assert len(clean) == len(seq) - 1
    assert clean["t_players_alive"].max() == 5


def test_round_and_match_ids(rounds):
    assert rounds["round_id"].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
    assert rounds["match_id"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
    assert rounds["snapshot_idx"].tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 0]


def test_bomb_plant_does_not_start_new_round(rounds):
    # lignes 5 et 6 : le chrono passe de 20 à 40 car la bombe est posée -> même round
    assert rounds.loc[5, "round_id"] == rounds.loc[6, "round_id"]


def test_fix_labels_uses_last_snapshot(rounds):
    # round 1 : 1re photo étiquetée CT, vrai gagnant T
    r1 = rounds[rounds["round_id"] == 1]
    assert r1["round_winner"].astype(str).tolist() == ["T"] * 4
    assert r1["t_win"].tolist() == [1] * 4
    # round 2 : 1re photo étiquetée T, vrai gagnant CT
    assert rounds.loc[rounds["round_id"] == 2, "t_win"].tolist() == [0, 0]
    assert rounds["label_fixed"].sum() == 2


def test_score_based_winner(rounds):
    sw = score_based_winner(rounds)
    assert sw.loc[1] == "T"      # t_score passe de 1 à 2
    assert sw.loc[0] == "T"      # t_score passe de 0 à 1
    assert pd.isna(sw.loc[2])    # dernier round du match : inconnu


def test_check_rounds(rounds):
    r = check_rounds(rounds)
    assert r["n_rounds"] == 5
    assert r["n_matches"] == 3
    assert r["rounds_with_several_winners"] == 0
    assert r["n_labels_fixed"] == 2
    assert r["share_score_step_1"] == pytest.approx(1.0)


def test_split_keeps_matches_together(rounds):
    rounds["split"] = make_split(rounds, test_size=0.34)
    assert (rounds.groupby("match_id")["split"].nunique() == 1).all()
    assert set(rounds["split"]) == {"train", "test"}


def test_group_split_has_no_leak(rounds):
    assert share_of_test_rounds_seen_in_train(rounds, "round") == 0.0
    assert share_of_test_rounds_seen_in_train(rounds, "match") == 0.0
