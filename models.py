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
funcionando solo con RandomForest
"""

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from config import RF_CONFIG, LGBM_CONFIG, RANDOM_SEED, DATA_DIR
from utils import ml_train_date_kept

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


def feature_importance_report(
    panel: pd.DataFrame,
    model_name: str = "RandomForest",
    train_start: str = "2008-01-01",
    train_end: str = "2019-12-31",
    feature_cols: list = None,
) -> pd.Series:
    """
    Entrena el modelo especificado con todos los datos del período IN-SAMPLE
    y genera un reporte de importancia de features.
    
    Parámetros
    ----------
    panel : pd.DataFrame
        DataFrame con las features y target
    model_name : str
        Nombre del modelo ("RandomForest" o "LightGBM")
    train_start : str
        Fecha de inicio del período de entrenamiento
    train_end : str
        Fecha de fin del período de entrenamiento
    feature_cols : list, optional
        Columnas de features. Si es None, se extraen automáticamente
        
    Retorna
    -------
    pd.Series
        Series con importancias ordenadas descendentemente
    """
    if feature_cols is None:
        exclude = {"date", "etf", "target", "return"}
        feature_cols = [c for c in panel.columns if c not in exclude]
    
    train = panel[
        (panel["date"] >= train_start) & (panel["date"] <= train_end)
    ].dropna(subset=["target"])
    train = train[ml_train_date_kept(train["date"])]

    pipe = build_pipeline(MODELS[model_name])
    pipe.fit(train[feature_cols].values, train["target"].values)

    importances = pd.Series(
        pipe.named_steps["model"].feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    importances.to_csv(f"{DATA_DIR}/feature_importance_{model_name}.csv")
    print(f"\n[FI] Top 10 features ({model_name}):")
    print(importances.head(10).to_string())
    return importances
