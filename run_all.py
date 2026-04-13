"""
run_all.py
==========
Pipeline completo de la estrategia de rotacion sectorial.

Orden de ejecucion:
  1.  (Opcional) Descarga de datos    -> etf_prices.csv, fred_macro.csv, ff5_factors.csv
  2.  Feature Engineering             -> features_panel.csv
  3.  Walk-Forward LightGBM/RF/GB      -> predictions_LightGBM.csv
                                         predictions_RandomForest.csv
                                         predictions_GradientBoosting.csv
                                         eda_etf_by_regime.csv / .png
  3b. Walk-Forward 3 RF por Régimen   -> predictions_RegimeRF.csv
  4.  Visualizacion regimenes HMM     -> market_regimes_plot.png + CSVs
  5.  Strategy Backtest               -> metricas + backtest_chart.png
  6.  Signal Evaluation (IC)          -> IC, quintiles + signal_evaluation_plot.png
  C.  Comparacion de estrategias      -> comparison_chart.png + tabla anual

Notas:
  - El paso 03 entrena LightGBM, RF y GB sobre ventana expansiva con HMM integrado;
    también ejecuta el EDA de rendimiento de ETFs por régimen de mercado.
  - El paso 03b entrena 3 RF especializados (uno por regimen Bear/Ranging/Bull).
  - El paso 04 corre despues (solo genera visualizacion y CSV de respaldo).
  - Para ejecutar solo desde los modelos: python run_models_only.py

Requiere:
  conda env create -f environment.yml   (crea el entorno tfm-ml-trading)
  conda activate tfm-ml-trading         (o simplemente: python run_all.py)
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))


def _build_runner() -> str:
    """
    Construye el comando para ejecutar cada paso del pipeline.

    Usa 'conda run -n tfm-ml-trading python' cuando el entorno está disponible.
    Esto activa correctamente el entorno conda (DLLs, PATH, variables de entorno)
    independientemente del Python que haya lanzado run_all.py.

    Fallback: sys.executable si conda o el entorno no se encuentran.
    """
    for candidate in [
        os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"),
        os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        r"C:\ProgramData\miniconda3\Scripts\conda.exe",
    ]:
        if os.path.exists(candidate):
            envs_dir = os.path.join(os.path.dirname(os.path.dirname(candidate)), "envs")
            if os.path.isdir(os.path.join(envs_dir, "tfm-ml-trading")):
                print("[run_all] Entorno: conda tfm-ml-trading (Python 3.11)")
                return f'"{candidate}" run --no-capture-output -n tfm-ml-trading python'

    print("[WARN] Entorno conda 'tfm-ml-trading' no encontrado. Usando Python del sistema.")
    return f'"{sys.executable}"'


RUNNER = _build_runner()

STEPS = [
    # ("1.  Descarga de datos",             "01_data_download.py"),   # omitir si ya existe
    ("2.  Feature Engineering",           "02_feature_engineering.py"),
    ("4.  Walk-Forward LightGBM/RF/GB + EDA", "04_walk_forward_training.py"),
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
