"""
config.py
=========
# v2 final — Abril 2026
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
TRAIN_END = "2019-12-31"        # Fin período de entrenamiento
OOS_START = "2020-01-01"        # Inicio out-of-sample (~260 semanas OOS)
OOS_END = "2024-12-31"          # Fin out-of-sample
DATA_START = "2000-01-01"       # Inicio descarga de datos raw
DATA_END = "2024-12-31"         # Fin descarga de datos raw

# Intervalos excluidos del entrenamiento de modelos supervisados (LightGBM, RF, GB).
# Lista vacía [] → sin exclusión. No altera 01_data_download ni el ajuste EM del HMM.
ML_TRAIN_EXCLUDE_PERIODS = [
    ("2020-01-01", "2021-12-31"),
]


# ============================================================================
# CONFIGURACIÓN HMM (Market Regime Detection)
# ============================================================================
# Parámetros del HMM (GaussianHMM con 3 estados) definidos en regime_model.py:
#   N_STATES=3, N_ITER=500, init económica explícita, forward filter causal.
# Aquí se mantienen los parámetros de ventana compartidos con los walk-forwards.

# Ventana rodante para el ajuste del HMM en 04b_regime_walk_forward.py.
# Justificación: los regímenes de mercado responden a ciclos económicos de 5-7 años.
# 260 semanas ≈ 5 años garantizan:
#   · Al menos 1 ciclo económico completo (Bull + Ranging + Bear)
#   · ~40-50 semanas Bear (15-20 % de 260 w) → suficiente para estimar Bear con 23 params
#   · Sensibilidad a cambios estructurales recientes (no se contaminan datos de 2008
#     cuando se infiere el régimen de 2024)
# La ventana ML sigue siendo EXPANSIVA desde TRAIN_START (distinción intencional).
HMM_REGIME_LOOKBACK = 260      # semanas ≈ 5 años (ventana rodante HMM en 04b)


# ============================================================================
# CONFIGURACIÓN MODELOS MACHINE LEARNING
# ============================================================================
RANDOM_SEED = 42

# Random Forest: 200 árboles, profundidad máxima 5, mínimo 10 muestras por hoja.
RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM (histogram-based gradient boosting, Jansen 2020 cap. 12)
# num_leaves=31 ≈ max_depth=5 en sklearn. lambda_l1/l2: regularización adicional.
LGBM_CONFIG = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 31,
    "max_depth"        : -1,
    "min_child_samples": 10,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "lambda_l1"        : 0.1,
    "lambda_l2"        : 1.0,
    "random_state"     : RANDOM_SEED,
    "n_jobs"           : -1,
    "verbose"          : -1,
}


# ============================================================================
# CONFIGURACIÓN WALK-FORWARD VALIDATION
# ============================================================================
TOP_N = 3                       # Top N ETFs para pata larga (long)
BOTTOM_N = 3                    # Bottom N ETFs para pata corta (short)
MIN_TRAIN_PERIODS = 104         # Historia mínima antes de predecir (semanas ≈ 24 meses)

# Ventanas de entrenamiento — separadas por funcion:
#   ML models: ventana EXPANSIVA desde TRAIN_START hasta t
#              (usa todos los datos IS disponibles; mas datos = mejor generalizacion)
#   HMM context: ventana fija de HMM_CONTEXT_PERIODS antes de t
#              (filtra el regimen actual solo con historia reciente)
HMM_CONTEXT_PERIODS = 235       # Ventana de contexto del filtro forward HMM en 04 (semanas ≈ 4.5 años)
                                # (parámetros HMM fijos en IS; solo se reutiliza para inferencia de estado)

# ── Kelly ─────────────────────────────────────────────────────────────────────
# Fracción del Kelly completo aplicada en 05_strategy_backtest.py.
# Kelly completo (1.0) maximiza el crecimiento logarítmico asintótico.
# Reducir a 0.5 (half-Kelly) reduciría el drawdown ≈50 % a costa de ≈25 % menos
# de retorno esperado si las señales son muy ruidosas (Thorp 2006).
KELLY_FRACTION = 1.0            # fracción Kelly completo para ponderación multi-activo

# Ventana causal (en SEMANAS) para estimar var_i en la ponderación Half-Kelly.
# 36 semanas ≈ 9 meses: suficiente historia para estimar varianza de forma
# estable pero reactiva ante cambios recientes de régimen de volatilidad.
# Usada en 05_strategy_backtest.py y compare_strategies.py.
KELLY_LOOKBACK_WEEKS = 36

# ── Costos de transacción ─────────────────────────────────────────────────────
# Basis points cobrados por cada leg (apertura o cierre de una posición).
# Aplicado en 05_strategy_backtest.py, compare_strategies.py y benchmarks
# long-only / long-short. Fórmula actual:
#   cost_t = n_legs_turned * (COST_BPS / 10_000) / n_positions_held
# siendo n_legs_turned = |simétrica(new, prev)| en cada pata.
COST_BPS = 10


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
    # ── Ciclo económico (series mensuales → resample W-FRI + ffill) ───────────
    "CPIAUCSL"          : "CPI",           # Inflacion (nivel)      -> CPI_YoY en 01
    "UNRATE"            : "Unemployment",  # Desempleo (nivel)      -> Unemp_Chg en 01
    "INDPRO"            : "IndProd",       # Produccion industrial  -> IndProd_YoY en 01
    # ── Tipos de interés (series diarias → resample W-FRI + ffill) ───────────
    "DFF"               : "FedFunds",      # Fed Funds efectiva (diaria) -> FedFunds_Chg
    "DTB3"              : "T3M",           # T-bill 3 meses (diario)
    "DGS10"             : "T10",           # Treasury 10 años (diario)
    "T10Y2Y"            : "YieldSpread",   # Spread 10yr-2yr (diario, nivel)
    # ── Riesgo y mercado (series diarias → resample W-FRI + ffill) ───────────
    "VIXCLS"            : "VIX",           # CBOE VIX (miedo / volatilidad, diario)
    # Gold: LBMA series retiradas de FRED en enero 2022.
    # Se descarga via yfinance (GC=F) en 01_data_download.py.
    "BAMLH0A0HYM2"      : "HY_OAS",       # ICE BofA HY OAS (credito / riesgo, diario)
    # ── Tasa repo y bonos internacionales ────────────────────────────────────
    "SOFR"              : "RepoRate",      # Secured Overnight Financing Rate (diario, desde 2018)
    "IRLTLT01JPM156N"   : "JGB10Y",        # Bono gobierno Japón 10 años (mensual, nivel)
    # ── Crédito corporativo y actividad manufacturera ─────────────────────────
    "BAMLC0A0CM"        : "IG_OAS",        # ICE BofA IG OAS (crédito grado inversión, diario, desde 1997)
    "CFNAI"             : "ISM",           # Chicago Fed NAI (compuesto 85 indicadores, proxy PMI, mensual)
    # ── Ciclo económico / recesiones / condiciones financieras ───────────────
    "JHDUSRGDPBR"       : "recession",            # NBER recession indicator (binario 0/1, mensual)
    "NFCI"              : "financial_conditions",  # Chicago Fed NFCI (semanal)
    "NFCILEVERAGE"      : "leverage",              # NFCI subíndice de apalancamiento (semanal)
    "UMCSENT"           : "sentiment",             # UMich Consumer Sentiment (mensual)
}


# ============================================================================
# VALIDACIÓN (Anti-leakage)
# ============================================================================
MIN_DATA_FREQ_DAYS = 4                  # Mínimo días entre observaciones (datos semanales)
MAX_DATA_FREQ_DAYS = 8                  # Máximo días entre observaciones (datos semanales)
MAX_NA_FORWARD_LOOKING = 0.10           # Max % NaN en posiciones futuras (forward-looking)
