"""
run_all.py
==========
# v2 final — Abril 2026
Pipeline completo de la estrategia de rotacion sectorial.

Orden de ejecucion:
  1.  (Opcional) Descarga de datos    -> etf_prices.csv, fred_macro.csv, ff5_factors.csv
  2.  Feature Engineering             -> features_panel.csv
  4.  Walk-Forward LightGBM/RF + EDA  -> predictions_LightGBM.csv
                                         predictions_RandomForest.csv
                                         eda_etf_by_regime.csv / .png
  4b. Walk-Forward 3 LGBM por Régimen -> predictions_RegimeLGBM.csv
  3.  Visualizacion regimenes HMM     -> market_regimes_plot.png + CSVs
  5.  Strategy Backtest               -> metricas + backtest_chart.png
  6.  Signal Evaluation (IC)          -> IC, quintiles + signal_evaluation_plot.png
  C.  Comparacion de estrategias      -> comparison_chart.png + tabla anual

Notas:
  - El paso 04 entrena LightGBM y RandomForest sobre ventana expansiva con HMM
    integrado; también ejecuta el EDA de rendimiento de ETFs por régimen.
  - El paso 04b entrena 3 RF especializados (uno por regimen Bear/Ranging/Bull).
  - Para ejecutar solo desde los modelos: python run_models_only.py

Requiere:
  conda env create -f environment.yml   (crea el entorno tfm-ml-trading)
  conda activate tfm-ml-trading         (o simplemente: python run_all.py).
"""

import os
import time

_here = os.path.dirname(os.path.abspath(__file__))

from utils import build_runner

RUNNER = build_runner("run_all")

STEPS = [
    # ("1.  Descarga de datos",             "01_data_download.py"),   # omitir si ya existe
    ("2.  Feature Engineering",           "02_feature_engineering.py"),
    ("4.  Walk-Forward LightGBM/RF + EDA",    "04_walk_forward_training.py"),
    ("4b. Walk-Forward 3 LGBM por Regimen", "04b_regime_walk_forward.py"),
    ("3.  Visualizacion regimenes HMM",     "03_market_regime_detection.py"),
    ("5.  Strategy Backtest",             "05_strategy_backtest.py"),
    ("6.  Signal Evaluation (IC)",        "06_signal_evaluation.py"),
    ("C.  Comparacion de estrategias",    "compare_strategies.py"),
]

if __name__ == "__main__":
    total_start = time.time()
    for step_name, script in STEPS:
        print(f"\n{'-'*60}")
        print(f"  {step_name}")
        print(f"{'-'*60}", flush=True)
        t0  = time.time()
        ret = os.system(f'cd /d "{_here}" && {RUNNER} {script}')
        elapsed = time.time() - t0
        if ret != 0:
            print(f"[ERROR] {script} (exit code {ret}). Pipeline detenido.")
            break
        print(f"  [OK] Completado en {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\n[DONE] Pipeline finalizado en {total:.0f}s.")
