"""Interprétation du modèle avec les valeurs SHAP.

LightGBM calcule les valeurs SHAP exactes de ses arbres (algorithme TreeSHAP) avec
l'option `pred_contrib=True`. Pour une ligne, on obtient une contribution par variable,
plus une valeur de base, en log-odds (logit) :

    logit(probabilité) = valeur de base + somme des contributions

Une contribution positive pousse vers une victoire T, une contribution négative vers
une victoire CT. Les colonnes one-hot de la carte sont regroupées en une seule
variable `map`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BASE = "valeur_de_base"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transformed_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Variables telles que le modèle les voit (après création des variables et encodage)."""
    return pipeline[:-1].transform(X)


def shap_values(pipeline: Pipeline, X: pd.DataFrame, group_map: bool = True) -> pd.DataFrame:
    """Contributions SHAP (log-odds), une colonne par variable + la valeur de base."""
    Xt = transformed_features(pipeline, X)
    contrib = pipeline[-1].predict(Xt, pred_contrib=True)
    out = pd.DataFrame(contrib, columns=list(Xt.columns) + [BASE], index=X.index)
    if group_map:
        map_cols = [c for c in out.columns if c.startswith("map_")]
        out.insert(0, "map", out[map_cols].sum(axis=1))
        out = out.drop(columns=map_cols)
    return out


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Importance globale = moyenne de |contribution|, triée par ordre décroissant."""
    return shap_df.drop(columns=BASE).abs().mean().sort_values(ascending=False)


def explain_row(pipeline: Pipeline, row: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Explication d'un snapshot : les `top` plus grosses contributions, le reste regroupé.

    Renvoie un tableau (variable, valeur, contribution) avec, en attributs,
    la probabilité de base et la probabilité finale.
    """
    shap_row = shap_values(pipeline, row).iloc[0]
    values = transformed_features(pipeline, row).iloc[0]
    base = shap_row[BASE]
    contrib = shap_row.drop(BASE)
    order = contrib.abs().sort_values(ascending=False).index
    main, rest = order[:top], order[top:]

    def shown_value(name):
        if name == "map":
            return str(row["map"].iloc[0])
        return values[name]

    table = pd.DataFrame(
        {"variable": main, "valeur": [shown_value(n) for n in main], "contribution": contrib[main].to_numpy()}
    )
    if len(rest):
        table.loc[len(table)] = [f"{len(rest)} autres variables", "", contrib[rest].sum()]
    table.attrs["proba_base"] = float(_sigmoid(base))
    table.attrs["proba_finale"] = float(_sigmoid(base + contrib.sum()))
    return table
