"""
run_models_only.py
==================
Ejecuta el pipeline DESDE los modelos (salta descarga de datos y feature
engineering, que son los pasos más lentos y solo son necesarios una vez).

Útil cuando los datos ya están preparados (features_panel.csv existe).

Orden de ejecución:
  4.  Walk-Forward RF/GB global  -> predictions_RandomForest.csv
                                    predictions_GradientBoosting.csv
  4b. Walk-Forward 3 RF Régimen  -> predictions_RegimeRF.csv
  3.  Visualización regímenes    -> market_regimes_plot.png + CSVs
  5.  Strategy Backtest          -> métricas + backtest_chart.png
  6.  Signal Evaluation (IC)     -> IC, quintiles + signal_evaluation_plot.png
  C.  Comparación de estrategias -> comparison_chart.png + tabla anual

Para un pipeline completo (con descarga y FE):
  python run_all.py
"""

import os
import sys
import time

_here    = os.path.dirname(os.path.abspath(__file__))
_venv_py = os.path.join(_here, ".venv", "Scripts", "python.exe")
PYTHON   = f'"{_venv_py}"' if os.path.exists(_venv_py) else f'"{sys.executable}"'

STEPS = [
    ("4.  Walk-Forward RF/GB (global)",   "04_walk_forward_training.py"),
    ("4b. Walk-Forward 3 RF por Régimen", "04b_regime_walk_forward.py"),
    ("3.  Visualización regímenes HMM",   "03_market_regime_detection.py"),
    ("5.  Strategy Backtest",             "05_strategy_backtest.py"),
    ("6.  Signal Evaluation (IC)",        "06_signal_evaluation.py"),
    ("C.  Comparación de estrategias",    "compare_strategies.py"),
]

if __name__ == "__main__":
    print("\n[INFO] Ejecutando pipeline de modelos (datos ya preparados)\n")
    total_start = time.time()

    for step_name, script in STEPS:
        print(f"\n{'-'*60}")
        print(f"  {step_name}")
        print(f"{'-'*60}")
        t0  = time.time()
        ret = os.system(f'cd /d "{_here}" && {PYTHON} {script}')
        elapsed = time.time() - t0
        if ret != 0:
            print(f"[ERROR] {script} falló. Pipeline detenido.")
            break
        print(f"  [OK] Completado en {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\n[DONE] Pipeline de modelos finalizado en {total:.0f}s.")
