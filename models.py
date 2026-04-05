"""
models.py
=========
Define los modelos de machine learning (RandomForest y GradientBoosting)
y funciones auxiliares para entrenamiento y evaluación.

Modelos disponibles:
  - RandomForest: n_estimators=200, max_depth=5
  - GradientBoosting: n_estimators=200, max_depth=3
"""

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import RF_CONFIG, GB_CONFIG, RANDOM_SEED, DATA_DIR

# Configuración de modelos
MODELS = {
    "RandomForest": RandomForestRegressor(**RF_CONFIG),
    "GradientBoosting": GradientBoostingRegressor(**GB_CONFIG),
}


def build_pipeline(model) -> Pipeline:
    """
    Construye un pipeline estándar con una copia fresca del estimador:
      1. Imputación de valores faltantes (mediana)
      2. Escalado estándar (StandardScaler)
      3. Modelo (clonado para evitar compartir estado entre iteraciones)

    El clon garantiza que cada llamada devuelva un Pipeline independiente,
    crítico en el walk-forward donde build_pipeline se llama en cada mes OOS.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   clone(model)),
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
        Nombre del modelo ("RandomForest" o "GradientBoosting")
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
