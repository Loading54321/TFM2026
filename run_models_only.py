"""
run_models_only.py
==================
# v2 final — Mayo 2026
Ejecuta el pipeline desde feature engineering (omite la descarga de datos,
que solo es necesaria una vez).

Útil cuando los datos raw ya están descargados (etf_prices.csv, fred_macro.csv,
ff5_factors.csv existen) pero se necesita regenerar el panel de features
(por ejemplo, al añadir nuevas variables).

Orden de ejecución:
  2.  Feature Engineering           -> features_panel.csv
  3.  Visualización regímenes HMM   -> market_regimes_plot.png + CSVs
  4.  Walk-Forward LightGBM/RF      -> predictions_LightGBM.csv
                                       predictions_RandomForest.csv
                                       eda_etf_by_regime.csv / .png
  4b. Walk-Forward RegimeLGBM       -> predictions_RegimeLGBM.csv
  5.  Strategy Backtest             -> métricas + backtest_chart.png
  6.  Signal Evaluation (IC)        -> IC, quintiles + signal_evaluation_plot.png
  C.  Comparación de estrategias    -> comparison_chart.png + tabla anual

Para un pipeline completo (con descarga de datos):
  python run_all.py
"""

import os
import time

_here = os.path.dirname(os.path.abspath(__file__))

from utils import build_runner

RUNNER = build_runner("run_models_only")

STEPS = [
    ("2.  Feature Engineering",             "02_feature_engineering.py"),
    ("3.  Visualización regímenes HMM",     "03_market_regime_detection.py"),
    ("4.  Walk-Forward LightGBM/RF + EDA",  "04_walk_forward_training.py"),
    ("4b. Walk-Forward RegimeLGBM",         "04b_regime_walk_forward.py"),
    ("5.  Strategy Backtest",               "05_strategy_backtest.py"),
    ("6.  Signal Evaluation (IC)",          "06_signal_evaluation.py"),
    ("C.  Comparación de estrategias",      "compare_strategies.py"),
]

if __name__ == "__main__":
    print("\n[INFO] Ejecutando pipeline de modelos (datos ya preparados)\n")
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
    print(f"\n[DONE] Pipeline de modelos finalizado en {total:.0f}s.")
