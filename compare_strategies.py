"""
compare_strategies.py
=====================
Tabla comparativa anual OOS (semanal):
  1. LightGBM / RandomForest — Half-Kelly Diagonal Long-Short
  2. RegimeLGBM — 3 LGBM especializados por régimen HMM (Half-Kelly Diagonal Long-Short)
  3. X Top3    — igual modelo, solo pata larga (Half-Kelly Diagonal Long-Only)
  4. SPY       — benchmark pasivo buy & hold
  5. Top-3 EW  — Top-3 por predicted_return, pesos iguales, solo largo

Todas las series tienen frecuencia semanal (W-FRI).  Los costes de transacción
se aplican con COST_BPS por leg, definido en config.py.
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

from config import DATA_DIR, TOP_N, BOTTOM_N, OOS_START, OOS_END, COST_BPS
from utils import weekly_forward_returns
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
    transaction_costs: bool = True,
) -> pd.Series:
    """
    Retornos semanales de un portafolio Equal-Weight puro.
      leg='long' : Top-N ETFs, weight = 1/N cada uno (retorno positivo)
      leg='short': Bottom-N ETFs, weight = 1/N cada uno (retorno negativo = corto)

    Costes de transacción: COST_BPS por nombre entrado/salido, normalizado por
    n_positions_held, igual que en build_portfolio (consistencia entre
    estrategias para que la comparación sea justa).

    Anti-leakage: retornos t+1 vía weekly_forward_returns (shift(-1)).
    """
    preds  = pd.read_csv(pred_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    pred_dates = sorted(preds["date"].unique())
    actual     = weekly_forward_returns(prices, pred_dates)

    rows        = []
    prev_held   = set()
    sign        = 1.0 if leg == "long" else -1.0
    n_select    = TOP_N if leg == "long" else BOTTOM_N

    for date, g in preds.groupby("date"):
        etfs = (
            g.nsmallest(n_select, "rank")["etf"].tolist() if leg == "long"
            else g.nlargest(n_select, "rank")["etf"].tolist()
        )

        rets = []
        held = set()
        for e in etfs:
            try:
                r = actual.loc[date, e]
                if pd.notna(r):
                    rets.append(float(r))
                    held.add(e)
            except KeyError:
                pass

        if not rets:
            continue

        port_ret = sign * float(np.mean(rets))

        if transaction_costs and held:
            turnover = len(held.symmetric_difference(prev_held))
            n_held   = len(held) or 1
            port_ret -= turnover * (COST_BPS / 10_000) / n_held
            prev_held = held

        rows.append({"date": date, "ret": port_ret})

    return (
        pd.DataFrame(rows)
        .set_index("date")
        .sort_index()["ret"]
    )


# ── portafolio Half-Kelly long-only (solo Top-N) ─────────────────────────────

def build_kelly_longonly(
    pred_path: str,
    prices_path: str,
    transaction_costs: bool = True,
) -> pd.Series:
    """
    Portafolio Half-Kelly con solo la pata larga (Top-N ETFs).

    Wrapper sobre build_portfolio(long_only=True): usa exactamente la misma
    lógica de ponderación, retorno t+1 y costes que la estrategia completa
    Long-Short, pero sin pata corta.

    Permite aislar la contribucion de la pata larga vs la long-short completa:
      - Si "X Kelly" >> "X Top3" → el short añade valor
      - Si "X Top3" ≈ "X Kelly" → el short resta o es neutral
    """
    df = build_portfolio(
        pred_path, prices_path,
        transaction_costs=transaction_costs,
        long_only=True,
    )
    return df["portfolio_return"]


# ── tabla principal ───────────────────────────────────────────────────────────

def run_comparison():
    prices_path = f"{DATA_DIR}/etf_prices.csv"
    prices      = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    strategies = {}
    longonly   = {}   # pata larga Half-Kelly Top-N de cada modelo
    shortleg   = {}   # pata corta standalone: -short_return (positivo = ganancia en corto)

    for model in ["LightGBM", "RandomForest"]:
        pred_path = f"{DATA_DIR}/predictions_{model}.csv"
        if os.path.exists(pred_path):
            port_df = build_portfolio(pred_path, prices_path, transaction_costs=True)
            strategies[f"{model} Kelly"]  = port_df["portfolio_return"]
            longonly[f"{model} Top3"]     = build_kelly_longonly(pred_path, prices_path)
            # Pata corta standalone: negamos short_return porque en build_portfolio
            # se resta (port = long - short), pero como inversión aislada la ganancia
            # es justamente esa negación: si el ETF baja, short_return < 0 → -short > 0.
            # Semanas sin posición corta (filtro simétrico) quedan en 0 (efectivo).
            shortleg[f"{model} Short3"]   = -port_df["short_return"]
        else:
            print(f"[!] No encontrado: {pred_path} — omitido")

    # RegimeLGBM: 3 LGBM especializados por régimen HMM
    regime_pred_path = f"{DATA_DIR}/predictions_RegimeLGBM.csv"
    if os.path.exists(regime_pred_path):
        port_regime = build_portfolio(
            regime_pred_path, prices_path, transaction_costs=True
        )
        strategies["RegimeLGBM Kelly"] = port_regime["portfolio_return"]
        longonly["RegimeLGBM Top3"]    = build_kelly_longonly(regime_pred_path, prices_path)
        shortleg["RegimeLGBM Short3"]  = -port_regime["short_return"]
    else:
        print(f"[!] No encontrado: {regime_pred_path} — ejecuta 04b_regime_walk_forward.py")

    # Periodo OOS alineado al primer modelo disponible
    first_key = next(iter(strategies), None)
    if first_key is None:
        print("[ERROR] No hay estrategias para comparar.")
        return pd.DataFrame()
    oos_idx = strategies[first_key].index

    # SPY benchmark: retornos semanales t→t+1 alineados con el portafolio.
    # Usamos weekly_forward_returns para aplicar exactamente la misma convención
    # que build_portfolio (retorno entre t y t+1, NaN al final).
    spy_fwd = weekly_forward_returns(prices[["SPY"]], oos_idx)["SPY"]
    strategies["SPY"] = spy_fwd.dropna()

    # EW Top-N (benchmark simple con los mismos costes que Kelly)
    pred_rf = f"{DATA_DIR}/predictions_RandomForest.csv"
    if os.path.exists(pred_rf):
        ew_long = build_ew_portfolio(pred_rf, prices_path, leg="long")
        strategies["Top-3 EW (RF)"] = ew_long.reindex(oos_idx).dropna()

    # Combinar: primero long-short, luego Top3-only, Short3, benchmarks pasivos
    all_strategies = {}
    for k, v in strategies.items():
        if k not in ("SPY", "Top-3 EW (RF)"):
            all_strategies[k] = v
    for k, v in longonly.items():
        all_strategies[k] = v.reindex(oos_idx).dropna()
    for k, v in shortleg.items():
        # fillna(0): semanas sin posición corta (filtro simétrico) = efectivo, retorno 0
        all_strategies[k] = v.reindex(oos_idx).fillna(0)
    all_strategies["SPY"]          = strategies["SPY"]
    if "Top-3 EW (RF)" in strategies:
        all_strategies["Top-3 EW (RF)"] = strategies["Top-3 EW (RF)"]
    strategies = all_strategies

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
    print(f"  Periodo OOS: {OOS_START[:4]}-{OOS_END[:4]}  |  Ranking basado en predicted_return del modelo")
    print(sep)
    print(df_fmt.to_string())
    print(sep)
    print()
    print("Notas:")
    print(f"  X Kelly      = Half-Kelly Diagonal: Top{TOP_N} largo + Bottom{BOTTOM_N} corto "
          f"(filtro: pred<0 AND |pred|>pred_Top1)")
    print(f"  X Top3       = Solo pata larga: Top{TOP_N} con pesos Half-Kelly Diagonal, sin posiciones cortas")
    print("  X Short3     = Solo pata corta standalone: -short_return del modelo completo")
    print("                 (positivo = ganancia en corto; 0 = semana sin posición por filtro simétrico)")
    print(f"  Top-3 EW(RF) = Top-{TOP_N} RandomForest, peso 1/{TOP_N} c/u, largo, con costos {COST_BPS} bps (benchmark EW)")
    print("  SPY          = Buy & hold S&P500 (benchmark pasivo)")
    print(f"  Frecuencia: semanal (W-FRI) | Costes tx: {COST_BPS} bps por leg | CAGR/Sharpe anualizados × 52")

    # ── grafico de retorno acumulado ──────────────────────────────────────────
    # Portafolio completo Long-Short (Kelly + filtro) + Top-3 long-only + SPY.
    focused_keys = (
        [k for k in strategies if k.endswith("Kelly")]
        + [k for k in strategies if k.endswith("Top3")]
        + ["SPY"]
    )
    focused = {k: strategies[k] for k in focused_keys if k in strategies}
    plot_cumulative(focused, oos_idx)

    return df_fmt


def plot_cumulative(strategies: dict, oos_idx):
    """
    Gráfico de retorno acumulado (valor de $1 invertido):
      · X Kelly  (todos los modelos) — portafolio Long-Short completo con
        Half-Kelly diagonal y filtro: pred<0 AND |pred|>pred_Top1 (línea sólida gruesa)
      · X Top3   (todos los modelos) — pata larga Kelly long-only (línea punteada,
        mismo color que su Kelly, para aislar la contribución del corto)
      · SPY                          — benchmark pasivo
    Dos paneles: retorno acumulado (base $1) y drawdown.
    """
    # Estilos: Kelly sólida gruesa · Top3 punteada (mismo color) · SPY verde
    styles = {
        # ── Portafolio completo Long-Short (sólidas, lw mayor) ───────────────
        "LightGBM Kelly"      : dict(lw=2.2, ls="-",  color="#1f77b4"),
        "RandomForest Kelly"  : dict(lw=2.2, ls="-",  color="#ff7f0e"),
        "RegimeLGBM Kelly"    : dict(lw=2.2, ls="-",  color="#9467bd"),
        # ── Pata larga Top-3 long-only (punteadas, mismo color) ──────────────
        "LightGBM Top3"       : dict(lw=1.5, ls="--", color="#1f77b4"),
        "RandomForest Top3"   : dict(lw=1.5, ls="--", color="#ff7f0e"),
        "RegimeLGBM Top3"     : dict(lw=1.5, ls="--", color="#9467bd"),
        # ── Benchmark ────────────────────────────────────────────────────────
        "SPY"                    : dict(lw=2.5, ls="-",  color="#2ca02c"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        f"Portafolio Long-Short Kelly (—) vs Top-{TOP_N} Long-Only (--) vs SPY  |  Por Modelo\n"
        f"OOS {OOS_START[:4]}–{OOS_END[:4]}  (semanal)  |  Half-Kelly Diagonal · "
        f"Filtro: pred<0 AND |pred|>pred_Top1 · Costos {COST_BPS} bps",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: retorno acumulado ────────────────────────────────────────────
    ax = axes[0]
    for label, series in strategies.items():
        s   = series.reindex(oos_idx).dropna()
        cum = (1 + s).cumprod()
        kw  = styles.get(label, dict(lw=1.5, ls="-", color="gray"))
        ax.plot(cum.index, cum.values, label=label, **kw)

    for year in range(oos_idx.year.min(), oos_idx.year.max() + 1):
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray",
                   lw=0.6, ls="--", alpha=0.45)

    ax.axhline(1.0, color="black", lw=0.8, ls="-", alpha=0.25)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.set_ylabel("Valor (base $1)")
    ax.set_title("Retorno Acumulado")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: drawdown ─────────────────────────────────────────────────────
    ax = axes[1]
    for label, series in strategies.items():
        s   = series.reindex(oos_idx).dropna()
        cum = (1 + s).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        kw  = styles.get(label, dict(lw=1.5, ls="-", color="gray"))
        ax.plot(dd.index, dd.values, label=label, **kw)

    for year in range(oos_idx.year.min(), oos_idx.year.max() + 1):
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color="gray",
                   lw=0.6, ls="--", alpha=0.45)

    ax.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax.fill_between(oos_idx, 0, -0.20, color="red", alpha=0.04)  # zona peligro -20%
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown  (zona roja = −20%)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{DATA_DIR}/comparison_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Grafico guardado: {out}")


if __name__ == "__main__":
    run_comparison()
