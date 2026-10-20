"""Style commun des graphiques du projet.

Une seule source pour les couleurs et la mise en forme, pour que tous
les graphiques du projet se ressemblent.

Convention de couleurs (gardée dans tout le projet) :
  - Terroristes (T)        -> rouge
  - Anti-terroristes (CT)  -> bleu
  - Carte de chaleur       -> bleu (CT gagne) / gris (50-50) / rouge (T gagne)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import ROOT_DIR

FIGURES_DIR = ROOT_DIR / "reports" / "figures"

COLOR_T = "#e34948"
COLOR_CT = "#2a78d6"
COLOR_NEUTRAL = "#f0efec"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"

# Palette divergente : 0 = CT gagne toujours (bleu), 0.5 = 50/50 (gris), 1 = T gagne toujours (rouge)
CMAP_WIN = LinearSegmentedColormap.from_list("ct_t", [COLOR_CT, COLOR_NEUTRAL, COLOR_T])


def set_style() -> None:
    """Applique un style sobre : grille légère, pas de bordures inutiles."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": COLOR_TEXT_MUTED,
        "axes.edgecolor": COLOR_GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "xtick.color": COLOR_TEXT_MUTED,
        "ytick.color": COLOR_TEXT_MUTED,
        "legend.frameon": False,
    })


def save_fig(fig, name: str) -> Path:
    """Sauvegarde la figure dans reports/figures/<name>.png (utilisée ensuite dans le README)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path
