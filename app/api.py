"""API de prédiction (FastAPI).

Lancement :
    uvicorn app.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving import MODEL_PATH, RoundState, load_model, predict  # noqa: E402

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))
    if not path.exists():
        raise RuntimeError(f"Modèle introuvable : {path}. Lancer d'abord `python -m src.train`.")
    state["model"] = load_model(path)
    state["model_path"] = str(path)
    yield
    state.clear()


app = FastAPI(
    title="CS:GO Round Winner",
    description="Probabilité de victoire d'un round de CS:GO à partir de l'état de la partie.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict_round(round_state: RoundState) -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="modèle non chargé")
    return predict(state["model"], round_state)
