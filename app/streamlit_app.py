"""Interface de démonstration (Streamlit).

Lancement :
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from src.serving import MAPS, MODEL_PATH, RoundState, TeamState, load_model, predict  # noqa: E402
from src.viz import COLOR_CT, COLOR_T, set_style  # noqa: E402

st.set_page_config(page_title="CS:GO Round Winner", layout="wide")
set_style()

LABELS = {
    "primary_diff": "Écart d'armes principales", "armor_diff": "Écart d'armure",
    "players_alive_diff": "Écart de joueurs vivants", "health_diff": "Écart de santé",
    "bomb_planted": "Bombe posée", "money_diff": "Écart d'argent", "rifle_diff": "Écart de fusils",
    "awp_diff": "Écart d'AWP", "time_left": "Temps restant", "time_left_no_bomb": "Temps restant (sans bombe)",
    "time_left_bomb": "Temps restant (bombe posée)", "map": "Carte", "helmets_diff": "Écart de casques",
    "grenades_diff": "Écart de grenades", "score_diff": "Écart de score", "round_number": "Numéro du round",
}

SCENARIOS = {
    "Début de round, achat complet des deux côtés": dict(
        time_left=115, bomb=False, ct=dict(players_alive=5, health=500, armor=500, money=3000, helmets=5,
                                           rifles=4, awp=1, smgs=0, pistols=5, grenades=12),
        t=dict(players_alive=5, health=500, armor=500, money=3500, helmets=5, rifles=4, awp=1, smgs=0,
               pistols=5, grenades=12)),
    "Bombe posée, 2 T contre 3 CT": dict(
        time_left=30, bomb=True, ct=dict(players_alive=3, health=240, armor=250, money=1500, helmets=3,
                                         rifles=3, awp=0, smgs=0, pistols=3, grenades=4),
        t=dict(players_alive=2, health=150, armor=180, money=800, helmets=2, rifles=2, awp=0, smgs=0,
               pistols=2, grenades=2)),
    "Round d'éco des T": dict(
        time_left=115, bomb=False, ct=dict(players_alive=5, health=500, armor=500, money=6000, helmets=5,
                                           rifles=5, awp=0, smgs=0, pistols=5, grenades=12),
        t=dict(players_alive=5, health=500, armor=200, money=600, helmets=0, rifles=0, awp=0, smgs=0,
               pistols=5, grenades=3)),
}


SIDE_LABELS = {
    "players_alive": "Joueurs vivants", "health": "Santé", "armor": "Armure", "money": "Argent",
    "helmets": "Casques", "primary": "Armes principales", "rifle": "Fusils", "awp": "AWP",
    "other_sniper": "Autres snipers", "smg": "SMG", "heavy": "Armes lourdes", "pistol": "Pistolets",
    "grenades": "Grenades", "health_per_player": "Santé par joueur", "armor_per_player": "Armure par joueur",
    "defuse_kits": "Kits de désamorçage", "score": "Score",
}


def label(name: str) -> str:
    """Nom lisible d'une variable du modèle."""
    if name in LABELS:
        return LABELS[name]
    for side in ("ct", "t"):
        if name.startswith(f"{side}_") and name[len(side) + 1:] in SIDE_LABELS:
            return f"{SIDE_LABELS[name[len(side) + 1:]]} ({side.upper()})"
    return name


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def apply_scenario(name: str) -> None:
    sc = SCENARIOS[name]
    st.session_state["time_left"] = sc["time_left"]
    st.session_state["bomb"] = sc["bomb"]
    for side in ("ct", "t"):
        for key, value in sc[side].items():
            st.session_state[f"{side}_{key}"] = value


DEFAULT_TEAM = dict(players_alive=5, health=500, armor=500, money=4000, helmets=5,
                    rifles=4, awp=1, smgs=0, pistols=5, grenades=10)


def init_state() -> None:
    """Valeurs par défaut, définies une seule fois (les widgets les lisent via leur clé)."""
    defaults = {"bomb": False, "time_left": 115, "map": "de_dust2", "ct_score": 0, "t_score": 0, "kits": 2}
    for side in ("ct", "t"):
        defaults.update({f"{side}_{k}": v for k, v in DEFAULT_TEAM.items()})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clamp(key: str, upper: float) -> None:
    """Évite qu'une valeur dépasse un maximum qui vient de diminuer (moins de joueurs, bombe posée)."""
    if st.session_state[key] > upper:
        st.session_state[key] = upper


def team_inputs(side: str, title: str) -> dict:
    st.subheader(title)
    alive = st.slider("Joueurs vivants", 0, 5, key=f"{side}_players_alive")
    for key in ("health", "armor"):
        clamp(f"{side}_{key}", 100 * alive)
    c1, c2 = st.columns(2)
    health = c1.number_input("Santé totale", 0, 100 * alive, step=10, key=f"{side}_health")
    armor = c2.number_input("Armure totale", 0, 100 * alive, step=10, key=f"{side}_armor")
    money = c1.number_input("Argent ($)", 0, 80000, step=250, key=f"{side}_money")
    helmets = c2.number_input("Casques", 0, 5, key=f"{side}_helmets")
    rifles = c1.number_input("Fusils", 0, 5, key=f"{side}_rifles")
    awp = c2.number_input("AWP", 0, 5, key=f"{side}_awp")
    smgs = c1.number_input("SMG", 0, 5, key=f"{side}_smgs")
    pistols = c2.number_input("Pistolets", 0, 5, key=f"{side}_pistols")
    grenades = st.slider("Grenades", 0, 20, key=f"{side}_grenades")
    return dict(players_alive=alive, health=health, armor=armor, money=money, helmets=helmets,
                rifles=rifles, awp=awp, smgs=smgs, pistols=pistols, grenades=grenades)


st.title("Probabilité de victoire d'un round de CS:GO")
st.caption("Modèle LightGBM entraîné sur 97 000 snapshots de matchs professionnels "
           "(ROC-AUC 0,806 sur des matchs jamais vus). Réglez la situation, la probabilité se met à jour.")

if not MODEL_PATH.exists():
    st.error("Modèle introuvable : lancer d'abord `python -m src.train`.")
    st.stop()

init_state()

with st.sidebar:
    st.header("Situation")
    scenario = st.selectbox("Charger un exemple", ["—"] + list(SCENARIOS))
    if scenario != "—" and st.button("Appliquer l'exemple"):
        apply_scenario(scenario)
        st.rerun()
    game_map = st.selectbox("Carte", MAPS[:-1], key="map")
    bomb = st.toggle("Bombe posée", key="bomb")
    if bomb:
        clamp("time_left", 40)  # bombe posée : le chrono repart à 40 s au maximum
    # maximum fixe : changer le maximum recréerait le curseur et perdrait sa valeur
    time_left = st.slider("Temps restant (s)", 0, 175, key="time_left",
                          help="Avec la bombe posée, 40 s au maximum.")
    c1, c2 = st.columns(2)
    ct_score = c1.number_input("Score CT", 0, 30, key="ct_score")
    t_score = c2.number_input("Score T", 0, 30, key="t_score")
    kits = st.slider("Kits de désamorçage (CT)", 0, 5, key="kits")

col_ct, col_t = st.columns(2)
with col_ct:
    ct = team_inputs("ct", "Anti-terroristes (CT)")
with col_t:
    t = team_inputs("t", "Terroristes (T)")

try:
    state = RoundState(map=game_map, time_left=time_left, bomb_planted=bomb, ct_score=ct_score,
                       t_score=t_score, ct_defuse_kits=kits, ct=TeamState(**ct), t=TeamState(**t))
except ValidationError as e:
    st.error("Situation impossible : " + "; ".join(err["msg"].removeprefix("Value error, ")
                                                   for err in e.errors()))
    st.stop()

result = predict(get_model(), state, top=6)
p_t = result["proba_t_win"]

st.divider()
m1, m2, m3 = st.columns([1, 1, 2])
m1.metric("Victoire T", f"{p_t:.0%}")
m2.metric("Victoire CT", f"{1 - p_t:.0%}")
with m3:
    fig, ax = plt.subplots(figsize=(6, 0.6))
    ax.barh([0], [p_t], color=COLOR_T, height=0.6)
    ax.barh([0], [1 - p_t], left=[p_t], color=COLOR_CT, height=0.6)
    ax.axvline(0.5, color="white", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.axis("off")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.subheader("Ce qui influence la prédiction")
factors = result["facteurs"][::-1]
fig, ax = plt.subplots(figsize=(8, 3))
names = [label(f["variable"]) for f in factors]
values = [f["contribution"] for f in factors]
ax.barh(names, values, color=[COLOR_T if v > 0 else COLOR_CT for v in values], height=0.6)
ax.axvline(0, color="black", linewidth=1)
lim = max(abs(v) for v in values) * 1.2 or 1
ax.set_xlim(-lim, lim)
ax.set_xlabel("Contribution (SHAP, log-odds) : vers les CT ← → vers les T")
ax.grid(axis="y", visible=False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Rouge : pousse vers une victoire des T. Bleu : pousse vers une victoire des CT.")
