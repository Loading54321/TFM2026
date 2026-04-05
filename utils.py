"""
utils.py
========
Funciones compartidas entre múltiples módulos.

Evita código duplicado:
  - Carga de datos
  - Preparación de features
  - Cálculos estándares
"""

import pandas as pd
import numpy as np
from config import DATA_DIR


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



# ============================================================================
# UTILIDADES GENERALES
# ============================================================================

def ensure_monthly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que DataFrame tiene frecuencia mensual.
    
    Args:
        df: DataFrame con índice de tiempo
        
    Returns:
        DataFrame (sin cambios si es válido)
        
    Raises:
        ValueError: Si no es frecuencia mensual
    """
    freq = df.index.to_series().diff().dt.days.median()
    if not (25 <= freq <= 35):
        raise ValueError(f"Frecuencia sospechosa: {freq} días (esperado ~30)")
    return df


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
