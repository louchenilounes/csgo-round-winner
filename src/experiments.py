"""Relance les expériences principales du projet et les enregistre dans MLflow.

1. Validation croisée par match (train) : meilleure référence, régression logistique,
   Random Forest, LightGBM par défaut et LightGBM optimisé.
2. Modèle final : LightGBM optimisé entraîné sur tout le train, évalué sur le test,
   enregistré avec le pipeline complet dans le Model Registry de MLflow.

Utilisation :
    python -m src.experiments
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.baselines import FrequencyTable
from src.config import ROOT_DIR, SEED, TARGET
from src.evaluation import compute_metrics, grouped_cv
from src.features import build_pipeline
from src.rounds import load_processed
from src.tracking import log_cv_run, log_final_model, runs_table, setup
from src.train import save_model, train_final_model
from src.tuning import DEFAULT_PARAMS, load_best_params, make_lgbm

warnings.filterwarnings("ignore", category=UserWarning)


def candidates(best_params: dict) -> dict:
    """Nom -> (pipeline, paramètres à enregistrer)."""
    return {
        "Table écart de joueurs x bombe": (
            FrequencyTable(use_bomb=True),
            {"modele": "FrequencyTable", "use_bomb": True},
        ),
        "Régression logistique": (
            build_pipeline(LogisticRegression(max_iter=2000), scale=True),
            {"modele": "LogisticRegression", "variables": "49 créées"},
        ),
        "Random Forest": (
            build_pipeline(
                RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
            ),
            {"modele": "RandomForest", "n_estimators": 300, "min_samples_leaf": 20},
        ),
        "LightGBM": (build_pipeline(make_lgbm(DEFAULT_PARAMS)), {"modele": "LightGBM", **DEFAULT_PARAMS}),
        "LightGBM optimisé": (
            build_pipeline(make_lgbm(best_params)),
            {"modele": "LightGBM", "optimise": "Optuna", **best_params},
        ),
    }


def main() -> None:
    setup()
    df = load_processed()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    best_params = load_best_params()

    for name, (pipeline, params) in candidates(best_params).items():
        folds = grouped_cv(pipeline, train)
        log_cv_run(name, params, folds, tags={"split": "GroupKFold par match, 5 plis"})
        print(
            f"{name:<32}: log loss {folds['log_loss'].mean():.4f} | accuracy {folds['accuracy'].mean():.2%}"
        )

    model = train_final_model(train, best_params)
    save_model(model, best_params, len(train))
    test_metrics = compute_metrics(test[TARGET].to_numpy(), model.predict_proba(test)[:, 1])
    figures = [
        ROOT_DIR / "reports" / "figures" / f
        for f in ["15_calibration_test.png", "18_shap_importance.png", "21_shap_explication_locale.png"]
    ]
    X_example = test.drop(
        columns=[TARGET, "round_winner", "match_id", "round_id", "snapshot_idx", "label_fixed", "split"]
    )
    log_final_model(model, best_params, test_metrics, X_example, artifacts=figures)
    print(
        f"Modèle final : test accuracy {test_metrics['accuracy']:.2%} | "
        f"log loss {test_metrics['log_loss']:.4f}"
    )
    print("\n" + runs_table().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
