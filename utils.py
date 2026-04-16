"""
utils.py
========
Funciones compartidas entre múltiples módulos.

Evita código duplicado:
  - Carga de datos
  - Preparación de features
  - Cálculos estándares
"""

import os
import sys
import pandas as pd
import numpy as np
from config import DATA_DIR, ML_TRAIN_EXCLUDE_PERIODS


# ============================================================================
# ENTRENAMIENTO ML (exclusiones de fechas)
# ============================================================================

def ml_train_date_kept(dates: pd.Series) -> pd.Series:
    """
    True donde la fila puede usarse en el fit de modelos supervisados.

    Aplica ML_TRAIN_EXCLUDE_PERIODS de config (p. ej. pandemia 2020-2021).
    No afecta al HMM ni a la descarga de datos.
    """
    if not ML_TRAIN_EXCLUDE_PERIODS:
        return pd.Series(True, index=dates.index)
    ok = pd.Series(True, index=dates.index)
    for start, end in ML_TRAIN_EXCLUDE_PERIODS:
        ts0 = pd.Timestamp(start)
        ts1 = pd.Timestamp(end)
        ok &= ~((dates >= ts0) & (dates <= ts1))
    return ok


# ============================================================================
# CARGAR DATOS
# ============================================================================

def load_etf_prices(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Carga precios de ETFs desde archivo CSV.
    
    Returns:
        DataFrame con índice = fecha, columnas = tickers
    """
    return pd.read_csv(f"{data_dir}/etf_prices.csv", index_col=0, parse_dates=True)


def load_macro(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Carga variables macroeconómicas desde archivo CSV.
    
    Returns:
        DataFrame con índice = fecha, columnas = variables macro
    """
    return pd.read_csv(f"{data_dir}/fred_macro.csv", index_col=0, parse_dates=True)


def load_ff5_factors(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Carga factores Fama-French 5 desde archivo CSV.
    
    Returns:
        DataFrame con índice = fecha, columnas = factores FF5
    """
    return pd.read_csv(f"{data_dir}/ff5_factors.csv", index_col=0, parse_dates=True)


def load_data(data_dir: str = DATA_DIR) -> tuple:
    """
    Carga todos los datos (precios, macro, FF5) de una vez.
    
    Returns:
        (prices_df, macro_df, ff5_df): Tupla de DataFrames
    """
    prices = load_etf_prices(data_dir)
    macro = load_macro(data_dir)
    ff5 = load_ff5_factors(data_dir)
    return prices, macro, ff5


# ============================================================================
# FEATURES
# ============================================================================

def load_panel(regime: bool = True, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Carga el panel de features construido por feature_engineering.py
    
    Formato LONG (panel):
      indice = (date, etf)
      columnas = features + target
    
    Args:
        regime: Si True, carga features_panel_with_regime.csv (incluye detección HMM)
                Si False, carga features_panel.csv (sin regímenes)
        data_dir: Directorio con archivos de datos
        
    Returns:
        DataFrame ordenado por (date, etf)
    """
    fname = "features_panel_with_regime.csv" if regime else "features_panel.csv"
    panel = pd.read_csv(f"{data_dir}/{fname}", parse_dates=["date"])
    panel.sort_values(["date", "etf"], inplace=True)
    return panel


def get_feature_cols(panel: pd.DataFrame) -> list:
    """
    Extrae las columnas de features del panel.

    Excluye únicamente los identificadores y la variable objetivo:
      - date, etf : identificadores del panel
      - target    : exceso de retorno ETF vs SPY en t+1  (variable a predecir)
      - return    : retorno bruto del mes actual (auxiliar, colineal con target)

    Incluye automáticamente todas las features construidas en 02_feature_engineering:
      ETF momentum/vol, rank cross-seccional, exceso vs SPY,
      SPY contexto, macro FRED (VIX, Gold, tipos, spread crédito),
      Fama-French 5, market_regime (si está presente).
    """
    exclude = {"date", "etf", "target", "return"}
    return [c for c in panel.columns if c not in exclude]



def get_oos_dates(panel: pd.DataFrame, oos_start: str, oos_end: str) -> list:
    """
    Extrae fechas únicas del período out-of-sample.
    
    Args:
        panel: DataFrame con columna 'date'
        oos_start: Fecha inicio OOS (str o Timestamp)
        oos_end: Fecha fin OOS (str o Timestamp)
        
    Returns:
        Lista de fechas únicas ordenadas
    """
    mask = (panel["date"] >= oos_start) & (panel["date"] <= oos_end)
    dates = sorted(panel[mask]["date"].unique())
    return dates


# ============================================================================
# FILTRO DE FRECUENCIA
# ============================================================================

def last_friday_of_month(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra predicciones semanales (W-FRI) al último viernes de cada mes.

    Usado por build_portfolio (05_strategy_backtest) y analyze_shorts para
    convertir predicciones semanales en decisiones mensuales de asignación,
    manteniendo estricto anti-leakage (solo usa la última observación
    disponible al cierre de cada mes).

    Args:
        preds: DataFrame con columna 'date' a frecuencia semanal.

    Returns:
        DataFrame filtrado con una fila por (mes × etf), correspondiente al
        último viernes disponible de cada mes.
    """
    preds = preds.copy()
    preds["_month"] = preds["date"].dt.to_period("M")
    preds = preds.loc[
        preds.groupby("_month")["date"].transform("max") == preds["date"]
    ].drop(columns=["_month"])
    return preds


# ============================================================================
# ENTORNO DE EJECUCION
# ============================================================================

def build_runner(caller: str = "") -> str:
    """
    Construye el comando Python correcto para el entorno conda tfm-ml-trading.

    Busca el ejecutable de conda en las rutas estándar de Anaconda/Miniconda
    y verifica que el entorno 'tfm-ml-trading' esté creado.

    Fallback: sys.executable (Python que lanzó el script) si conda o el
    entorno no se encuentran.

    Args:
        caller: Nombre del script que llama (para el mensaje de log).

    Returns:
        Cadena con el comando completo listo para os.system().
    """
    label = f"[{caller}]" if caller else "[runner]"
    for candidate in [
        os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"),
        os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        r"C:\ProgramData\miniconda3\Scripts\conda.exe",
    ]:
        if os.path.exists(candidate):
            envs_dir = os.path.join(os.path.dirname(os.path.dirname(candidate)), "envs")
            if os.path.isdir(os.path.join(envs_dir, "tfm-ml-trading")):
                print(f"{label} Entorno: conda tfm-ml-trading (Python 3.11)")
                return f'"{candidate}" run --no-capture-output -n tfm-ml-trading python'

    print(f"{label} Entorno conda 'tfm-ml-trading' no encontrado. Usando Python del sistema.")
    return f'"{sys.executable}"'
