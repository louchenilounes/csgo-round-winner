"""Chargement des données CS:GO Round Snapshots.

Fonctions :
  1. télécharger le fichier ARFF depuis OpenML (une seule fois) ;
  2. le lire correctement (noms de colonnes, valeurs manquantes « ? ») ;
  3. donner les bons types aux colonnes et créer la cible t_win ;
  4. sauvegarder un CSV propre dans data/raw/ ;
  5. vérifier la qualité des données (manquants, doublons, équilibre de la cible).

Utilisation depuis le terminal (à la racine du projet) :
    python -m src.data
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_DATA_PATH, TARGET

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

# Adresse du fichier sur OpenML.
OPENML_URL = "https://www.openml.org/data/download/22102255/dataset"

# Fichier brut téléchargé (format ARFF, un format texte utilisé par OpenML).
RAW_ARFF_PATH = RAW_DATA_DIR / "csgo_round_snapshots.arff"

# Valeurs attendues pour ce dataset (servent aux vérifications).
EXPECTED_N_ROWS = 122_410
EXPECTED_N_COLUMNS = 97  # 96 variables + round_winner (avant création de t_win)

# Colonnes qui ne sont pas des nombres.
CATEGORICAL_COLUMNS = ["map", "round_winner"]
BOOLEAN_COLUMNS = ["bomb_planted"]

# Quelques colonnes indispensables : si l'une manque, le fichier n'est pas le bon.
KEY_COLUMNS = [
    "time_left", "ct_score", "t_score", "map", "bomb_planted",
    "ct_health", "t_health", "ct_armor", "t_armor", "ct_money", "t_money",
    "ct_helmets", "t_helmets", "ct_defuse_kits",
    "ct_players_alive", "t_players_alive", "round_winner",
]


# --------------------------------------------------------------------------
# 1. Téléchargement
# --------------------------------------------------------------------------

def download_raw_data(url: str = OPENML_URL, path: Path = RAW_ARFF_PATH,
                      force: bool = False) -> Path:
    """Télécharge le fichier ARFF depuis OpenML et le sauvegarde dans data/raw/.

    Si le fichier existe déjà, on ne le retélécharge pas (sauf si force=True).
    """
    path = Path(path)
    if path.exists() and not force:
        print(f"[download] Fichier déjà présent : {path.name} (pas de nouveau téléchargement)")
        return path

    import requests  # importé ici : seulement nécessaire pour télécharger

    print(f"[download] Téléchargement depuis {url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()  # erreur claire si le site répond mal (404, 500...)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 Mo à la fois
                f.write(chunk)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[download] Terminé : {path.name} ({size_mb:.1f} Mo)")
    return path


# --------------------------------------------------------------------------
# 2. Lecture du fichier ARFF
# --------------------------------------------------------------------------

def _parse_attribute_name(line: str) -> str:
    """Extrait le nom de colonne d'une ligne « @ATTRIBUTE nom type ».

    Gère aussi les noms entre guillemets, par exemple : @ATTRIBUTE 'time_left' NUMERIC
    """
    rest = line.strip()[len("@attribute"):].strip()
    if rest and rest[0] in "'\"":
        quote = rest[0]
        end = rest.index(quote, 1)
        return rest[1:end]
    return rest.split()[0]


def read_arff(path: Path) -> pd.DataFrame:
    """Lit un fichier ARFF et renvoie un DataFrame pandas.

    Un fichier ARFF a deux parties :
      - un en-tête avec une ligne « @ATTRIBUTE » par colonne ;
      - les données, après la ligne « @DATA », au format CSV.
    Les valeurs manquantes sont écrites « ? ».
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    columns: list[str] = []
    data_start = None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            columns.append(_parse_attribute_name(line))
        elif low.startswith("@data"):
            data_start = i + 1
            break

    if data_start is None or not columns:
        raise ValueError(f"{path.name} ne ressemble pas à un fichier ARFF (pas de @ATTRIBUTE ou @DATA).")

    # On garde seulement les vraies lignes de données (ni vides, ni commentaires « % »).
    data_lines = [l for l in lines[data_start:] if l.strip() and not l.lstrip().startswith("%")]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=columns,
        na_values="?",
        quotechar="'",
        skipinitialspace=True,
    )
    return df


# --------------------------------------------------------------------------
# 3. Types et cible
# --------------------------------------------------------------------------

def _to_bool(series: pd.Series) -> pd.Series:
    """Convertit 'True'/'False' (texte), True/False ou 1/0 en vrais booléens."""
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    return series.astype(str).str.strip().str.lower().map(mapping).astype("boolean")


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Donne le bon type à chaque colonne et crée la cible t_win.

    - map et round_winner  -> catégories (texte)
    - bomb_planted         -> booléen
    - toutes les autres    -> nombres (float)
    - t_win                -> 1 si les Terroristes gagnent le round, 0 sinon
    """
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().astype("category")

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = _to_bool(df[col])

    numeric_cols = [c for c in df.columns if c not in CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Cible explicite (plus sûr que cat.codes, qui dépend de l'ordre alphabétique des catégories).
    winners = set(df["round_winner"].astype(str).unique())
    if not winners <= {"CT", "T"}:
        raise ValueError(f"Valeurs inattendues dans round_winner : {winners}")
    df[TARGET] = (df["round_winner"] == "T").astype(int)

    return df


# --------------------------------------------------------------------------
# 4. Chargement complet (point d'entrée principal)
# --------------------------------------------------------------------------

def build_raw_csv(arff_path: Path = RAW_ARFF_PATH, csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """ARFF -> DataFrame typé -> CSV propre. Renvoie le DataFrame."""
    df = clean_types(read_arff(arff_path))
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[build] CSV propre sauvegardé : {Path(csv_path).name} ({len(df):,} lignes)")
    return df


def load_data(csv_path: Path = RAW_DATA_PATH, download: bool = True) -> pd.DataFrame:
    """Charge les données prêtes à l'emploi.

    - Si le CSV propre existe : on le lit directement (rapide).
    - Sinon : on télécharge l'ARFF (si besoin), on le convertit, on sauvegarde le CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        if not RAW_ARFF_PATH.exists():
            if not download:
                raise FileNotFoundError(f"Ni {csv_path.name} ni {RAW_ARFF_PATH.name} n'existent dans data/raw/.")
            download_raw_data()
        return build_raw_csv(RAW_ARFF_PATH, csv_path)

    df = pd.read_csv(csv_path)
    # Le CSV ne garde pas les types : on les remet.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    for col in BOOLEAN_COLUMNS:
        df[col] = _to_bool(df[col])
    return df


# --------------------------------------------------------------------------
# 5. Vérifications de qualité
# --------------------------------------------------------------------------

def check_data(df: pd.DataFrame) -> dict:
    """Calcule un petit rapport de qualité et le renvoie sous forme de dictionnaire."""
    n_rows, n_cols = df.shape
    target_share = df[TARGET].value_counts(normalize=True).sort_index()

    report = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "missing_columns": [c for c in KEY_COLUMNS if c not in df.columns],
        "n_missing_values": int(df.isna().sum().sum()),
        "columns_with_missing": df.columns[df.isna().any()].tolist(),
        "n_duplicated_rows": int(df.duplicated().sum()),
        "share_t_win": float(target_share.get(1, 0.0)),
        "share_ct_win": float(target_share.get(0, 0.0)),
        "n_maps": int(df["map"].nunique()),
        "maps": sorted(df["map"].astype(str).unique().tolist()),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
    }
    return report


def print_report(report: dict) -> None:
    """Affiche le rapport de check_data de façon lisible."""
    ok = lambda cond: "OK " if cond else "!! "
    print("\n========== Rapport de qualité des données ==========")
    print(f"{ok(report['n_rows'] == EXPECTED_N_ROWS)}Lignes            : {report['n_rows']:,} (attendu {EXPECTED_N_ROWS:,})")
    print(f"{ok(report['n_columns'] == EXPECTED_N_COLUMNS + 1)}Colonnes          : {report['n_columns']} (attendu {EXPECTED_N_COLUMNS + 1}, t_win compris)")
    print(f"{ok(not report['missing_columns'])}Colonnes clés     : {'toutes présentes' if not report['missing_columns'] else report['missing_columns']}")
    print(f"{ok(report['n_missing_values'] == 0)}Valeurs manquantes: {report['n_missing_values']}"
          + (f" dans {report['columns_with_missing']}" if report['columns_with_missing'] else ""))
    print(f"   Lignes en double  : {report['n_duplicated_rows']:,} "
          f"({report['n_duplicated_rows'] / report['n_rows']:.1%}) -> conservées (voir la reconstruction des rounds)")
    print(f"   Victoires T       : {report['share_t_win']:.1%}")
    print(f"   Victoires CT      : {report['share_ct_win']:.1%}")
    print(f"   Cartes            : {report['n_maps']} -> {', '.join(report['maps'])}")
    print(f"   Types de colonnes : {report['dtypes']}")
    print("====================================================\n")


if __name__ == "__main__":
    data = load_data()
    print_report(check_data(data))
