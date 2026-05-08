"""
models.py
=========
# v2 final — Abril 2026
Define los modelos de machine learning y funciones auxiliares para
entrenamiento y evaluación.

Modelos disponibles:
  - RandomForest : n_estimators=200, max_depth=5  (sklearn, baseline ensemble)
  - LightGBM     : n_estimators=500, num_leaves=31 (histogram-based GB, Jansen 2020 cap.12)

Configuración centralizada en config.py (RF_CONFIG, LGBM_CONFIG).
LGBM_AVAILABLE=False si lightgbm no está instalado — el proyecto sigue
funcionando solo con RandomForest.
"""

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from config import RF_CONFIG, LGBM_CONFIG, RANDOM_SEED

# Diccionario unificado de modelos disponibles
MODELS: dict = {"RandomForest": RandomForestRegressor(**RF_CONFIG)}
if LGBM_AVAILABLE:
    MODELS["LightGBM"] = LGBMRegressor(**LGBM_CONFIG)


def build_pipeline(model) -> Pipeline:
    """
    Construye un pipeline estándar con una copia fresca del estimador:
      1. Escalado estándar (StandardScaler)
      2. Modelo (clonado para evitar compartir estado entre iteraciones)

    NaN en features se cubren con ffill en 02_feature_engineering.py.
    El clon garantiza que cada llamada devuelva un Pipeline independiente,
    crítico en el walk-forward donde build_pipeline se llama en cada semana OOS.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  clone(model)),
    ])
