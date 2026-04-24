"""
analyze_shorts.py
=================
Diagnóstico detallado de la pata corta (Bottom-N) del portafolio SEMANAL.

Genera data/shorts_analysis.csv con una fila por (fecha × modelo × ETF bottom-N),
reproduciendo exactamente la misma lógica de selección, filtrado y ponderación
que build_portfolio() en 05_strategy_backtest.py:

  Filtro AND:
    (1) pred < 0
    (2) |pred| > pred_Top1  (predicción del ETF mejor rankeado esa semana)

  Peso: simple_kelly_weights() — Half-Kelly diagonal por activo, normalizado,
        con ventana causal KELLY_LOOKBACK_WEEKS (config.py).

Columnas del CSV:
  date                 viernes W-FRI (fecha de decisión)
  model                LightGBM / RandomForest / RegimeLGBM
  etf                  ticker del candidato a short
  rank                 posición en el ranking esa semana (mayor = peor)
  pred                 predicted_return del modelo
  top1_etf             ticker del ETF #1 esa semana (umbral del filtro)
  pred_top1            predicted_return del ETF #1 (umbral)
  top3_pred_mean       media de pred del Top-N (referencia vs filtro anterior)
  top3_etfs            tickers Top-N separados por |
  cond1_pred_neg       pred < 0
  cond2_abs_gt_top1    |pred| > pred_top1
  passed_filter        cond1 AND cond2
  etf_var_weeks        varianza histórica (ventana KELLY_LOOKBACK_WEEKS) usada en Half-Kelly
  half_kelly_scalar    KELLY_FRACTION * |pred| / var  (antes de normalizar)
  weight               peso final asignado (0 si no pasó el filtro)
  actual_return        retorno real del ETF la semana siguiente (target)
  short_contribution   -weight * actual_return  (>0 = short ganó dinero)
  short_profitable     actual_return < 0  (el ETF bajó: short hubiera ganado)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
from config import DATA_DIR, TOP_N, BOTTOM_N, KELLY_FRACTION, KELLY_LOOKBACK_WEEKS
from utils import weekly_forward_returns

_bt = import_module("05_strategy_backtest")
simple_kelly_weights = _bt.simple_kelly_weights

PRICES_PATH = f"{DATA_DIR}/etf_prices.csv"
OUT_PATH    = f"{DATA_DIR}/shorts_analysis.csv"


def analyze_model(model_name: str, pred_path: str, prices: pd.DataFrame) -> list:
    preds      = pd.read_csv(pred_path, parse_dates=["date"])
    pred_dates = sorted(preds["date"].unique())

    # Retornos t→t+1 alineados al índice semanal de decisión (sin lookahead)
    actual_returns = weekly_forward_returns(prices, pred_dates)

    rows = []

    for date, group in preds.groupby("date"):
        pred_by_etf = group.set_index("etf")["predicted_return"]
        rank_by_etf = group.set_index("etf")["rank"]

        top_etfs    = group.nsmallest(TOP_N,    "rank")["etf"].tolist()
        bottom_etfs = group.nlargest( BOTTOM_N, "rank")["etf"].tolist()

        pred_top1  = float(pred_by_etf[top_etfs[0]])
        top1_etf   = top_etfs[0]
        top3_mean  = float(pred_by_etf[top_etfs].mean())
        top3_str   = "|".join(top_etfs)

        # Retornos reales disponibles (anti-leakage: shift -1 ya aplicado)
        def _actual(etf):
            try:
                r = actual_returns.loc[date, etf]
                return float(r) if pd.notna(r) else np.nan
            except KeyError:
                return np.nan

        # Varianza histórica causal para cada ETF (misma ventana que simple_kelly_weights)
        hist_ret = prices[bottom_etfs].pct_change().dropna()
        hist_ret = hist_ret.loc[hist_ret.index <= date].tail(KELLY_LOOKBACK_WEEKS)

        def _var(etf):
            if etf not in hist_ret.columns:
                return np.nan
            col = hist_ret[etf]
            v = col.iloc[:, 0].var() if isinstance(col, pd.DataFrame) else col.var()
            return float(v)

        # Evaluar filtro para cada ETF del Bottom-3
        cond_results = {}
        for etf in bottom_etfs:
            p      = float(pred_by_etf[etf])
            c1     = p < 0
            c2     = abs(p) > pred_top1
            passed = c1 and c2
            cond_results[etf] = {"c1": c1, "c2": c2, "passed": passed}

        short_etfs_filtered = [e for e in bottom_etfs if cond_results[e]["passed"]]

        # Calcular pesos — solo sobre los que pasaron el filtro
        if short_etfs_filtered:
            # Necesitamos también los long_etfs para llamar simple_kelly_weights correctamente
            long_rets_avail = {
                e: _actual(e) for e in top_etfs
                if not np.isnan(_actual(e))
            }
            long_etfs_avail = list(long_rets_avail.keys())

            if long_etfs_avail:
                _, short_w = simple_kelly_weights(
                    long_etfs_avail, short_etfs_filtered,
                    pred_by_etf, prices, date,
                    lookback=KELLY_LOOKBACK_WEEKS, kelly_fraction=KELLY_FRACTION,
                )
                weight_map = short_w.to_dict()
            else:
                weight_map = {e: 0.0 for e in short_etfs_filtered}
        else:
            weight_map = {}

        # Construir una fila por ETF bottom-3
        for etf in bottom_etfs:
            p       = float(pred_by_etf[etf])
            var_val = _var(etf)
            safe_var = max(var_val, 1e-8) if not np.isnan(var_val) else 1e-8
            hk      = KELLY_FRACTION * abs(p) / safe_var
            actual  = _actual(etf)
            passed  = cond_results[etf]["passed"]
            weight  = float(weight_map.get(etf, 0.0)) if passed else 0.0
            contrib = -weight * actual if not np.isnan(actual) else np.nan

            rows.append({
                "date"              : date,
                "model"             : model_name,
                "etf"               : etf,
                "rank"              : int(rank_by_etf[etf]),
                "pred"              : round(p, 6),
                "top1_etf"          : top1_etf,
                "pred_top1"         : round(pred_top1, 6),
                "top3_pred_mean"    : round(top3_mean, 6),
                "top3_etfs"         : top3_str,
                "cond1_pred_neg"    : cond_results[etf]["c1"],
                "cond2_abs_gt_top1" : cond_results[etf]["c2"],
                "passed_filter"     : passed,
                "etf_var_weeks"     : round(var_val, 8) if not np.isnan(var_val) else np.nan,
                "half_kelly_scalar" : round(hk, 4),
                "weight"            : round(weight, 6),
                "actual_return"     : round(actual, 6) if not np.isnan(actual) else np.nan,
                "short_contribution": round(contrib, 6) if not np.isnan(contrib) else np.nan,
                "short_profitable"  : (actual < 0) if not np.isnan(actual) else np.nan,
            })

    return rows


def main():
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices.sort_index(inplace=True)

    models = {
        "LightGBM"    : f"{DATA_DIR}/predictions_LightGBM.csv",
        "RandomForest": f"{DATA_DIR}/predictions_RandomForest.csv",
        "RegimeLGBM"  : f"{DATA_DIR}/predictions_RegimeLGBM.csv",
    }

    all_rows = []
    for model_name, pred_path in models.items():
        if not os.path.exists(pred_path):
            print(f"[!] No encontrado: {pred_path} — omitido")
            continue
        print(f"  Procesando {model_name}...")
        rows = analyze_model(model_name, pred_path, prices)
        all_rows.extend(rows)
        print(f"    -> {len(rows)} filas generadas "
              f"({sum(r['passed_filter'] for r in rows)} pasaron el filtro, "
              f"{sum(r['short_profitable'] is True for r in rows)} hubieran sido rentables)")

    df = pd.DataFrame(all_rows).sort_values(["model", "date", "rank"], ascending=[True, True, False])

    df.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] CSV guardado: {OUT_PATH}")
    print(f"     {len(df)} filas | {df['model'].nunique()} modelos | {df['date'].nunique()} semanas")

    # Resumen rápido en consola
    print("\n-- Resumen por modelo --------------------------------------------------")
    for model, g in df.groupby("model"):
        total      = len(g)
        passed     = g["passed_filter"].sum()
        profitable = (g["short_profitable"] == True).sum()
        contrib    = g["short_contribution"].sum()
        print(f"  {model:15s} | candidatos: {total:3d} | pasaron filtro: {passed:3d} "
              f"| rentables (real): {profitable:3d} | contrib total: {contrib:+.4f}")

    print("\n-- Top perdidas de la pata corta (short_contribution mas negativo) ----")
    worst = (
        df[df["passed_filter"]]
        .nsmallest(10, "short_contribution")[
            ["date", "model", "etf", "pred", "actual_return", "weight", "short_contribution"]
        ]
    )
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
