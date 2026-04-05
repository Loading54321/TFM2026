"""
config.py
=========
Configuración centralizada del proyecto TFM.

Todos los parámetros globales están aquí para fácil mantenimiento:
  - Rutas de datos
  - Períodos de entrenamiento/test
  - Parámetros de HMM
  - Configuración de modelos ML
  - APIs y otras constantes
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DIRECTORIOS
# ============================================================================
DATA_DIR = "data"


# ============================================================================
# PERÍODOS TEMPORALES
# ============================================================================
TRAIN_START = "2008-01-01"      # Inicio período de entrenamiento
TRAIN_END = "2019-12-31"        # Fin período de entrenamiento (día antes OOS_START)
OOS_START = "2020-01-01"        # Inicio out-of-sample (test)
OOS_END = "2024-12-31"          # Fin out-of-sample
DATA_START = "2000-01-01"       # Inicio descarga de datos raw
DATA_END = "2024-12-31"         # Fin descarga de datos raw


# ============================================================================
# CONFIGURACIÓN HMM (Market Regime Detection)
# ============================================================================
# Parámetros del HMM (GaussianHMM con 3 estados) definidos en regime_model.py:
#   N_STATES=3, N_ITER=400, init económica explícita, forward filter causal.
# Aquí solo se mantiene RANDOM_SEED (compartido con los modelos ML).


# ============================================================================
# CONFIGURACIÓN MODELOS MACHINE LEARNING
# ============================================================================
RANDOM_SEED = 42

# Random Forest
RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Gradient Boosting
GB_CONFIG = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 10,
    "random_state": RANDOM_SEED,
}


# ============================================================================
# CONFIGURACIÓN WALK-FORWARD VALIDATION
# ============================================================================
TOP_N = 3                       # Top N ETFs para pata larga (long)
BOTTOM_N = 3                    # Bottom N ETFs para pata corta (short)
MIN_TRAIN_MONTHS = 24           # Historia mínima antes de predecir

# Ventanas de entrenamiento — separadas por funcion:
#   ML models: ventana EXPANSIVA desde TRAIN_START hasta t
#              (usa todos los datos IS disponibles; mas datos = mejor generalizacion)
#   HMM context: ventana fija de HMM_CONTEXT_MONTHS antes de t
#              (filtra el regimen actual solo con historia reciente)
HMM_CONTEXT_MONTHS = 54         # Ventana de contexto del filtro forward HMM (4.5 años)

DEFAULT_MODEL = "RandomForest"


# ============================================================================
# ETFs Y MERCADOS
# ============================================================================
SECTOR_ETFS = [
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Healthcare
    "XLY",  # Consumer Discretionary
    "IYR",  # Real Estate (iShares U.S. Real Estate ETF)
    "VOX",  # Communication Services (Vanguard Comm. Services ETF)
]
BENCHMARK = ["SPY"]
ALL_TICKERS = SECTOR_ETFS + BENCHMARK


# ============================================================================
# VARIABLES MACROECONÓMICAS (FRED API)
# ============================================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_SERIES = {
    # ── Ciclo económico ───────────────────────────────────────────────────────
    "CPIAUCSL"          : "CPI",           # Inflacion (nivel)      -> CPI_YoY en 01
    "UNRATE"            : "Unemployment",  # Desempleo (nivel)      -> Unemp_Chg en 01
    "FEDFUNDS"          : "FedFunds",      # Tasa Fed Funds         -> FedFunds_Chg en 01
    "INDPRO"            : "IndProd",       # Produccion industrial  -> IndProd_YoY en 01
    "GDPC1"             : "GDP",           # PIB real EEUU (trimestral) -> GDP_YoY en 01
                                           # (interpolado a mensual con forward-fill)
    "T10Y2Y"            : "YieldSpread",   # Spread 10yr-2yr (nivel)
    # ── Riesgo y mercado ─────────────────────────────────────────────────────
    "VIXCLS"            : "VIX",           # CBOE VIX (miedo / volatilidad)
    "GOLDAMGBD228NLBM"  : "Gold",          # Precio oro Londres PM (USD/oz)
    "GS3M"              : "T3M",           # T-bill 3 meses (corto plazo mercado)
    "GS10"              : "T10",           # Treasury 10 años (largo plazo)
    "BAMLH0A0HYM2"      : "HY_OAS",       # ICE BofA HY OAS (credito / riesgo)
}


# ============================================================================
# VALIDACIÓN (Anti-leakage)
# ============================================================================
VALIDA_TEMPORAL_ALIGNMENT = True        # Validar que frecuencias sean mensuales
VALIDATE_ANTI_LEAKAGE = True            # Auditoria de data leakage
MIN_DATA_FREQ_DAYS = 25                 # Mínimo días entre observaciones
MAX_DATA_FREQ_DAYS = 35                 # Máximo días entre observaciones
MAX_NA_FORWARD_LOOKING = 0.10           # Max % NaN en posiciones futuras (forward-looking)
