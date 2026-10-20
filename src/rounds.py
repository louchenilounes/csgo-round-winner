"""Étape 3 : reconstruction des rounds et des matchs, correction des étiquettes,
puis découpage train/test sans fuite.

Le problème
-----------
Chaque ligne du dataset est une photo d'un round, prise environ toutes les 20 secondes.
Un même round donne donc plusieurs lignes, toutes avec le même gagnant.
Avec un découpage aléatoire des lignes, des photos d'un même round se retrouvent
à la fois dans le train et dans le test : le modèle « reconnaît » le round et
son score est trop optimiste. C'est une fuite de données (data leakage).

La solution
-----------
1. Retrouver à quel round (round_id) et à quel match (match_id) appartient chaque ligne.
2. Découper par groupe : toutes les photos d'un match vont ensemble dans le train OU dans le test.

Comment on retrouve les matchs et les rounds ?
----------------------------------------------
Les lignes sont rangées dans l'ordre chronologique.
  - NOUVEAU MATCH : la carte change, ou le score total (ct_score + t_score) diminue
    (un nouveau match repart de 0-0).
  - NOUVEAU ROUND : nouveau match, ou le score total change.
    Le score augmente de 1 à chaque fin de round, donc un round = un même score dans un même match.

Une erreur d'étiquette dans le dataset
--------------------------------------
En analysant les vraies données, on a découvert que la PREMIÈRE photo de chaque round
(prise pendant la phase d'achat, chrono vers 175 s, score déjà mis à jour) porte
souvent le gagnant du round PRÉCÉDENT. Preuve : le gagnant de la dernière photo du round
correspond au gagnant déduit du score suivant dans environ 99 % des cas, contre
environ 64 % pour la première photo.
On corrige donc l'étiquette : le gagnant d'un round = le gagnant de sa dernière photo.

Utilisation :
    python -m src.rounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import PROCESSED_DATA_DIR, SEED, TARGET
from src.data import load_data

PROCESSED_PATH = PROCESSED_DATA_DIR / "csgo_rounds.csv"
MAX_PLAYERS = 5
TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# 1. Nettoyage minimal
# --------------------------------------------------------------------------

def drop_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes impossibles dans le jeu (plus de 5 joueurs vivants dans une équipe)."""
    bad = (df["ct_players_alive"] > MAX_PLAYERS) | (df["t_players_alive"] > MAX_PLAYERS)
    if bad.any():
        print(f"[clean] {bad.sum()} ligne(s) supprimée(s) : plus de {MAX_PLAYERS} joueurs dans une équipe")
    return df.loc[~bad].reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Reconstruction des matchs et des rounds
# --------------------------------------------------------------------------

def add_round_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute match_id, round_id et snapshot_idx (numéro de la photo dans le round).

    Les lignes doivent être dans l'ordre d'origine du fichier (ordre chronologique).
    """
    d = df.copy()
    score_sum = d["ct_score"] + d["t_score"]
    prev_score = score_sum.shift()
    map_changed = d["map"].astype(str) != d["map"].astype(str).shift()

    new_match = map_changed | (score_sum < prev_score)
    new_match.iloc[0] = True
    new_round = new_match | (score_sum != prev_score)

    d["match_id"] = new_match.cumsum().astype(int) - 1
    d["round_id"] = new_round.cumsum().astype(int) - 1
    d["snapshot_idx"] = d.groupby("round_id").cumcount()
    return d


# --------------------------------------------------------------------------
# 3. Correction des étiquettes
# --------------------------------------------------------------------------

def fix_labels(d: pd.DataFrame) -> pd.DataFrame:
    """Gagnant du round = gagnant de sa dernière photo. Recalcule t_win.

    Ajoute la colonne label_fixed (True si l'étiquette de la ligne a été corrigée).
    """
    d = d.copy()
    true_winner = d.groupby("round_id")["round_winner"].transform("last").astype(str)
    d["label_fixed"] = d["round_winner"].astype(str) != true_winner
    d["round_winner"] = pd.Categorical(true_winner, categories=["CT", "T"])
    d[TARGET] = (true_winner == "T").astype(int)
    return d


def score_based_winner(d: pd.DataFrame) -> pd.Series:
    """Gagnant de chaque round déduit du score du round suivant (même match).

    CT si seul ct_score augmente de 1, T si seul t_score augmente de 1, sinon inconnu
    (dernier round du match, changement de côté à la mi-temps...).
    """
    r = d.groupby("round_id").agg(match_id=("match_id", "first"),
                                  ct=("ct_score", "first"), t=("t_score", "first"))
    dct = r.groupby("match_id")["ct"].shift(-1) - r["ct"]
    dt = r.groupby("match_id")["t"].shift(-1) - r["t"]
    winner = np.where((dct == 1) & (dt == 0), "CT", np.where((dct == 0) & (dt == 1), "T", None))
    return pd.Series(winner, index=r.index, name="score_winner")


# --------------------------------------------------------------------------
# 4. Vérifications
# --------------------------------------------------------------------------

def check_rounds(d: pd.DataFrame) -> dict:
    """Contrôles de cohérence de la reconstruction et des étiquettes."""
    rounds = d.groupby("round_id").agg(
        match_id=("match_id", "first"),
        n_snapshots=("round_id", "size"),
        n_winners=("round_winner", "nunique"),
        winner=("round_winner", "last"),
        score_sum=("ct_score", "first"),
    )
    rounds["score_sum"] += d.groupby("round_id")["t_score"].first()
    step = rounds.groupby("match_id")["score_sum"].diff().dropna()
    per_match = d.groupby("match_id")["round_id"].nunique()

    sw = score_based_winner(d)
    known = sw.notna()
    agreement = (rounds.loc[known, "winner"].astype(str) == sw[known]).mean() if known.any() else float("nan")

    return {
        "n_rows": len(d),
        "n_rounds": int(d["round_id"].nunique()),
        "n_matches": int(d["match_id"].nunique()),
        "n_labels_fixed": int(d["label_fixed"].sum()) if "label_fixed" in d else 0,
        "rounds_with_several_winners": int((rounds["n_winners"] > 1).sum()),
        "snapshots_per_round_median": float(rounds["n_snapshots"].median()),
        "snapshots_per_round_max": int(rounds["n_snapshots"].max()),
        "rounds_over_20_snapshots": int((rounds["n_snapshots"] > 20).sum()),
        "rounds_per_match_median": float(per_match.median()),
        "share_score_step_1": float((step == 1).mean()) if len(step) else float("nan"),
        "share_score_step_2plus": float((step >= 2).mean()) if len(step) else float("nan"),
        "share_winner_known_from_score": float(known.mean()),
        "label_agreement_with_score": float(agreement),
    }


def print_round_report(r: dict) -> None:
    print("\n========== Reconstruction des rounds ==========")
    print(f"Lignes                            : {r['n_rows']:,}")
    print(f"Matchs reconstruits               : {r['n_matches']:,}")
    print(f"Rounds reconstruits               : {r['n_rounds']:,}")
    print(f"Rounds par match (médiane)        : {r['rounds_per_match_median']:.0f}")
    print(f"Photos par round (médiane / max)  : {r['snapshots_per_round_median']:.0f} / {r['snapshots_per_round_max']}")
    print(f"Rounds de plus de 20 photos       : {r['rounds_over_20_snapshots']} (pauses : chrono bloqué)")
    print("Écart de score entre 2 rounds consécutifs d'un même match :")
    print(f"   +1 (normal)                    : {r['share_score_step_1']:.1%}")
    print(f"   +2 ou plus (round sans photo)  : {r['share_score_step_2plus']:.1%}")
    print("--- Étiquettes ---")
    print(f"Lignes dont le gagnant a été corrigé : {r['n_labels_fixed']:,} ({r['n_labels_fixed'] / r['n_rows']:.1%})")
    print(f"Rounds avec 2 gagnants (doit = 0)    : {r['rounds_with_several_winners']}")
    print(f"Gagnant vérifiable par le score      : {r['share_winner_known_from_score']:.1%} des rounds")
    print(f"Accord étiquette / score suivant     : {r['label_agreement_with_score']:.1%}")
    print("===============================================\n")


# --------------------------------------------------------------------------
# 5. Découpage train / test par match
# --------------------------------------------------------------------------

def make_split(d: pd.DataFrame, group_col: str = "match_id",
               test_size: float = TEST_SIZE, seed: int = SEED) -> pd.Series:
    """Renvoie une colonne 'train' / 'test' : un groupe entier va d'un seul côté."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(d, d[TARGET], groups=d[group_col]))
    split = pd.Series("train", index=d.index)
    split.iloc[test_idx] = "test"
    return split


# --------------------------------------------------------------------------
# 6. Pipeline complet
# --------------------------------------------------------------------------

def build_processed_data(save: bool = True) -> pd.DataFrame:
    """Données brutes -> lignes impossibles retirées -> rounds/matchs -> étiquettes corrigées -> split -> CSV."""
    d = fix_labels(add_round_ids(drop_impossible_rows(load_data())))
    d["split"] = make_split(d)
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(PROCESSED_PATH, index=False)
        print(f"[save] {PROCESSED_PATH.name} sauvegardé ({len(d):,} lignes)")
    return d


def load_processed() -> pd.DataFrame:
    """Charge les données de l'étape 3 (les crée si besoin)."""
    if not PROCESSED_PATH.exists():
        return build_processed_data()
    d = pd.read_csv(PROCESSED_PATH)
    d["map"] = d["map"].astype("category")
    d["round_winner"] = d["round_winner"].astype("category")
    for col in ["bomb_planted", "label_fixed"]:
        d[col] = d[col].astype(str).str.lower().map({"true": True, "false": False}).astype("boolean")
    return d


if __name__ == "__main__":
    data = build_processed_data()
    print_round_report(check_rounds(data))
    counts = data["split"].value_counts()
    print("Découpage par match :")
    for name in ["train", "test"]:
        part = data[data["split"] == name]
        print(f"   {name:<5}: {counts[name]:>7,} lignes | {part['round_id'].nunique():>5,} rounds | "
              f"{part['match_id'].nunique():>4,} matchs | victoires T {part[TARGET].mean():.1%}")
