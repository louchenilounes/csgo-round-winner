"""Fonctions de calcul pour l'analyse exploratoire.

Les calculs sont ici (testables), les graphiques sont dans le notebook.
Toutes les fonctions renvoient le taux de victoire des Terroristes (t_win moyen)
et le nombre de lignes, pour toujours savoir sur combien de données repose un chiffre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET


def win_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Taux de victoire T et nombre de lignes pour chaque valeur de `col`."""
    out = df.groupby(col, observed=True)[TARGET].agg(t_win_rate="mean", n="size").reset_index()
    return out


def alive_matrix(df: pd.DataFrame, max_players: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Taux de victoire T selon (joueurs CT vivants, joueurs T vivants).

    Renvoie deux tableaux : les taux (lignes = CT vivants, colonnes = T vivants)
    et les effectifs. Les lignes avec plus de `max_players` joueurs (anomalie) sont exclues.
    """
    ok = (df["ct_players_alive"] <= max_players) & (df["t_players_alive"] <= max_players)
    d = df.loc[ok]
    rates = d.pivot_table(
        index="ct_players_alive", columns="t_players_alive", values=TARGET, aggfunc="mean", observed=True
    )
    counts = d.pivot_table(
        index="ct_players_alive", columns="t_players_alive", values=TARGET, aggfunc="size", observed=True
    )
    rates.index = rates.index.astype(int)
    rates.columns = rates.columns.astype(int)
    counts.index = counts.index.astype(int)
    counts.columns = counts.columns.astype(int)
    return rates, counts


def add_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les écarts T - CT (positif = avantage T) pour les principales ressources."""
    d = df.copy()
    d["alive_diff"] = d["t_players_alive"] - d["ct_players_alive"]
    d["health_diff"] = d["t_health"] - d["ct_health"]
    d["armor_diff"] = d["t_armor"] - d["ct_armor"]
    d["money_diff"] = d["t_money"] - d["ct_money"]
    return d


def win_rate_by_bins(df: pd.DataFrame, col: str, bins, labels=None) -> pd.DataFrame:
    """Découpe `col` en intervalles et calcule le taux de victoire T dans chacun."""
    d = df.assign(bin=pd.cut(df[col], bins=bins, labels=labels, include_lowest=True))
    return win_rate_by(d, "bin")


def time_profile(df: pd.DataFrame, step: float = 10.0) -> pd.DataFrame:
    """Taux de victoire T par tranche de temps restant, séparément bombe posée / non posée.

    Rappel du jeu : un round dure 1 min 55 s ; quand la bombe est posée,
    le chrono repart à 40 s. Les deux situations ne sont donc pas comparables
    à temps restant égal, d'où la séparation.
    """
    edges = np.arange(0, df["time_left"].max() + step, step)
    d = df.assign(time_bin=pd.cut(df["time_left"], bins=edges, include_lowest=True))
    out = (
        d.groupby(["bomb_planted", "time_bin"], observed=True)[TARGET]
        .agg(t_win_rate="mean", n="size")
        .reset_index()
    )
    out["time_mid"] = out["time_bin"].apply(lambda iv: iv.mid).astype(float)
    return out


def top_correlations(df: pd.DataFrame, n: int = 15) -> pd.Series:
    """Les `n` variables numériques les plus corrélées (en valeur absolue) avec la cible.

    Attention : la corrélation ne mesure que les liens linéaires, une variable à la fois.
    On l'utilise ici pour décrire les données, pas pour choisir les variables du modèle.
    """
    num = df.select_dtypes(include=["number", "bool", "boolean"]).astype(float)
    corr = num.corr()[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)
