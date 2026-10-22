# Image de démonstration : API FastAPI (port 8000) et interface Streamlit (port 8501).
# Le modèle doit avoir été entraîné avant la construction : python -m src.train
FROM python:3.13-slim

# LightGBM a besoin de la bibliothèque OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY models/lightgbm_pipeline.joblib models/lightgbm_pipeline.joblib

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
