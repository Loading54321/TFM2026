"""
compare_strategies.py
=====================
Tabla comparativa anual OOS (2020-2024):
  1. RandomForest   — Kelly Multi-Activo (Top3 largo / Bottom3 corto)
  2. GradientBoosting — Kelly Multi-Activo
  3. RegimeRF       — 3 RF especializados por régimen HMM (Kelly Multi-Activo)
  4. SPY            — benchmark pasivo
  5. Top-3 EW       — Top-3 por predicted_return, pesos iguales, solo largo
  6. Bottom-3 EW    — Bottom-3 por predicted_return, pesos iguales, solo corto
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sin ventana — guarda directamente a disco
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, TOP_N, BOTTOM_N
from importlib import import_module

# Importar desde 05_strategy_backtest (nombre con numero, requiere import_module)
_bt = import_module("05_strategy_backtest")
build_portfolio = _bt.build_portfolio
cagr            = _bt.cagr
sharpe          = _bt.sharpe
max_drawdown    = _bt.max_drawdown


# ── helpers de metricas ───────────────────────────────────────────────────────

def annual_return(series: pd.Series, year: int) -> float:
    s = series[series.index.year == year]
    return float((1 + s).prod() - 1) if len(s) > 0 else np.nan


# ── portafolio equal-weight (long o short) ───────────────────────────────────

def build_ew_portfolio(
    pred_path: str,
    prices_path: str,
    leg: str = "long",
) -> pd.Series:
    """
    Retornos mensuales de un portafolio EW puro, sin Kelly ni costos.
      leg='long' : Top-N ETFs, weight = 1/N cada uno (retorno positivo)
      leg='short': Bottom-N ETFs, weight = 1/N cada uno (retorno negativo = corto)

    Anti-leakage: retornos t+1 via shift(-1), igual que en build_portfolio.
    """
    preds  = pd.read_csv(pred_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    actual = prices.pct_change().shift(-1)

    rows = []
    for date, g in preds.groupby("date"):
        if leg == "long":
            etfs = g.nsmallest(TOP_N, "rank")["etf"].tolist()
            sign = +1.0
        else:
            etfs = g.nlargest(BOTTOM_N, "rank")["etf"].tolist()
            sign = -1.0

        month_rets = []
        for e in etfs:
            try:
                r = actual.loc[date, e]
                if pd.notna(r):
                    month_rets.append(float(r))
            except KeyError:
                pass

        if month_rets:
            rows.append({"date": date, "ret": sign * float(np.mean(month_rets))})

    return (
        pd.DataFrame(rows)
        .set_index("date")
        .sort_index()["ret"]
    )


# ── tabla principal ───────────────────────────────────────────────────────────

def run_comparison():
    prices_path = f"{DATA_DIR}/etf_prices.csv"
    prices      = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    spy_all     = prices["SPY"].pct_change().dropna()

    strategies = {}

    for model in ["RandomForest", "GradientBoosting"]:
        pred_path = f"{DATA_DIR}/predictions_{model}.csv"
        if os.path.exists(pred_path):
            port_df = build_portfolio(pred_path, prices_path, transaction_costs=True)
            strategies[f"{model} Kelly"] = port_df["portfolio_return"]
        else:
            print(f"[!] No encontrado: {pred_path} — omitido")

    # RegimeRF: 3 RF especializados por régimen HMM
    regime_pred_path = f"{DATA_DIR}/predictions_RegimeRF.csv"
    if os.path.exists(regime_pred_path):
        port_regime = build_portfolio(
            regime_pred_path, prices_path, transaction_costs=True
        )
        strategies["RegimeRF Kelly"] = port_regime["portfolio_return"]
    else:
        print(f"[!] No encontrado: {regime_pred_path} — ejecuta 04b_regime_walk_forward.py")

    # Periodo OOS alineado al primer modelo disponible
    first_key = next(iter(strategies), None)
    if first_key is None:
        print("[ERROR] No hay estrategias para comparar.")
        return pd.DataFrame()
    oos_idx = strategies[first_key].index
    strategies["SPY"] = spy_all.reindex(oos_idx).dropna()

    # EW benchmarks usando predicciones de RandomForest (mismo ranking)
    pred_rf  = f"{DATA_DIR}/predictions_RandomForest.csv"
    if os.path.exists(pred_rf):
        ew_long  = build_ew_portfolio(pred_rf, prices_path, leg="long")
        ew_short = build_ew_portfolio(pred_rf, prices_path, leg="short")
        strategies["Top-3 EW Long"]     = ew_long.reindex(oos_idx).dropna()
        strategies["Bottom-3 EW Short"] = ew_short.reindex(oos_idx).dropna()

    # ── tabla año a año + metricas globales ──────────────────────────────────
    years = sorted({d.year for d in oos_idx})

    rows = []
    for label, series in strategies.items():
        row = {"Estrategia": label}
        for y in years:
            row[str(y)] = annual_return(series, y)
        row["CAGR (total)"] = cagr(series)
        row["Sharpe"]       = sharpe(series)
        row["Max DD"]       = max_drawdown(series)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Estrategia")

    # ── formateo ─────────────────────────────────────────────────────────────
    df_fmt = df.copy().astype(object)
    for c in [str(y) for y in years] + ["CAGR (total)"]:
        df_fmt[c] = df[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    df_fmt["Max DD"] = df["Max DD"].map(lambda x: f"{x:.1%}"  if pd.notna(x) else "—")
    df_fmt["Sharpe"] = df["Sharpe"].map(lambda x: f"{x:.2f}"  if pd.notna(x) else "—")

    sep = "=" * 100
    print(f"\n{sep}")
    print("  COMPARACION ANUAL OOS — Retorno compuesto por año + métricas globales")
    print(f"  Periodo OOS: 2020-2024  |  Ranking basado en predicted_return del modelo")
    print(sep)
    print(df_fmt.to_string())
    print(sep)
    print()
    print("Notas:")
    print("  Kelly        = Kelly Multi-Activo: Top3 largo + Bottom3 corto, 6 pesos suman 100%")
    print("  Top-3 EW     = Top-3 por predicted_return, peso 1/3 c/u, solo largo, sin costos tx")
    print("  Bottom-3 EW  = Bottom-3 por predicted_return, peso 1/3 c/u, solo corto, sin costos tx")
    print("  SPY          = Buy & hold S&P500 (benchmark pasivo)")

    # ── grafico de retorno acumulado ──────────────────────────────────────────
    plot_cumulative(strategies, oos_idx)

    return df_fmt


def plot_cumulative(strategies: dict, oos_idx):
    """
    Grafico de retorno acumulado (valor de $1 invertido) para todas las estrategias.
    Incluye lineas verticales en cada inicio de año.
    """
    # Estilos diferenciados por tipo de estrategia
    styles = {
        "RandomForest Kelly"   : dict(lw=2.0, ls="-",  color="#1f77b4"),
        "GradientBoosting Kelly": dict(lw=2.0, ls="--", color="#17becf"),
        "RegimeRF Kelly"       : dict(lw=2.5, ls="-",  color="#9467bd"),
        "SPY"                  : dict(lw=2.5, ls="-",  color="#2ca02c"),
        "Top-3 EW Long"        : dict(lw=2.0, ls="-",  color="#ff7f0e"),
        "Bottom-3 EW Short"    : dict(lw=2.0, ls=":",  color="#d62728"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        "Comparación de Estrategias OOS 2020-2024\n"
        "Retorno acumulado y Drawdown (base $1)",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: retorno acumulado
    ax = axes[0]
    for label, series in strategies.items():
        s = series.reindex(oos_idx).dropna()
        cum = (1 + s).cumprod()
        kw  = styles.get(label, dict(lw=1.5, ls="-"))
        ax.plot(cum.index, cum.values, label=label, **kw)

    # Lineas verticales en inicio de cada año
    for year in range(oos_idx.year.min(), oos_idx.year.max() + 1):
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray",
                   lw=0.6, ls="--", alpha=0.5)

    ax.axhline(1.0, color="black", lw=0.8, ls="-", alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.set_ylabel("Valor (base $1)")
    ax.set_title("Retorno Acumulado")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2: drawdown
    ax = axes[1]
    for label, series in strategies.items():
        s   = series.reindex(oos_idx).dropna()
        cum = (1 + s).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        kw  = styles.get(label, dict(lw=1.5, ls="-"))
        ax.plot(dd.index, dd.values, label=label, **kw)

    for year in range(oos_idx.year.min(), oos_idx.year.max() + 1):
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray",
                   lw=0.6, ls="--", alpha=0.5)

    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{DATA_DIR}/comparison_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Grafico guardado: {out}")


if __name__ == "__main__":
    run_comparison()
