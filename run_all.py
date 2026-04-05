"""
run_all.py
==========
Pipeline completo de la estrategia de rotacion sectorial.

Orden de ejecucion:
  1.  (Opcional) Descarga de datos    -> etf_prices.csv, fred_macro.csv, ff5_factors.csv
  2.  Feature Engineering             -> features_panel.csv
  4.  Walk-Forward RF/GB global       -> predictions_RandomForest.csv
                                         predictions_GradientBoosting.csv
  4b. Walk-Forward 3 RF por Régimen   -> predictions_RegimeRF.csv
  3.  Visualizacion regimenes HMM     -> market_regimes_plot.png + CSVs
  5.  Strategy Backtest               -> metricas + backtest_chart.png
  6.  Signal Evaluation (IC)          -> IC, quintiles + signal_evaluation_plot.png
  C.  Comparacion de estrategias      -> comparison_chart.png + tabla anual

Notas:
  - El paso 04 entrena RF y GB sobre ventana expansiva con HMM integrado.
  - El paso 04b entrena 3 RF especializados (uno por regimen Bear/Ranging/Bull).
  - El paso 03 corre despues (solo genera visualizacion y CSV de respaldo).
  - Para ejecutar solo desde los modelos: python run_models_only.py

Requiere:
  pip install yfinance fredapi hmmlearn scikit-learn pandas_datareader matplotlib seaborn
"""

import os
import sys
import time

_here    = os.path.dirname(os.path.abspath(__file__))
_venv_py = os.path.join(_here, ".venv", "Scripts", "python.exe")
PYTHON   = f'"{_venv_py}"' if os.path.exists(_venv_py) else f'"{sys.executable}"'

STEPS = [
    # ("1.  Descarga de datos",             "01_data_download.py"),   # omitir si ya existe
    ("2.  Feature Engineering",           "02_feature_engineering.py"),
    ("4.  Walk-Forward RF/GB (global)",   "04_walk_forward_training.py"),
    ("4b. Walk-Forward 3 RF por Regimen", "04b_regime_walk_forward.py"),
    ("3.  Visualizacion regimenes HMM",   "03_market_regime_detection.py"),
    ("5.  Strategy Backtest",             "05_strategy_backtest.py"),
    ("6.  Signal Evaluation (IC)",        "06_signal_evaluation.py"),
    ("C.  Comparacion de estrategias",    "compare_strategies.py"),
]

if __name__ == "__main__":
    total_start = time.time()
    for step_name, script in STEPS:
        print(f"\n{'-'*60}")
        print(f"  {step_name}")
        print(f"{'-'*60}")
        t0  = time.time()
        ret = os.system(f"{PYTHON} {script}")
        elapsed = time.time() - t0
        if ret != 0:
            print(f"[ERROR] {script}. Pipeline detenido.")
            break
        print(f"  [OK] Completado en {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\n[DONE] Pipeline finalizado en {total:.0f}s.")
