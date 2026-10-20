"""Tests de base : la configuration se charge et les dossiers existent."""

from src import config


def test_seed_is_fixed():
    assert config.SEED == 42


def test_project_folders_exist():
    for folder in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.MODELS_DIR]:
        assert folder.exists(), f"Dossier manquant : {folder}"
