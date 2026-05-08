"""
05_strategy_backtest.py
=======================
# v2 final — Abril 2026
Construye el portafolio SEMANAL a partir de las predicciones OOS (W-FRI) y
calcula métricas de performance vs SPY.

Predicciones y rebalanceo: frecuencia semanal (último viernes de cada semana).
La decisión de asignación se toma cada viernes con información <= t; los
retornos se devengan entre t y t+1 (semana siguiente) usando precios W-FRI.

Estrategia Long-Short con Kelly diagonal por activo:
  Long   Top-N  ETFs  (mayor retorno predicho)
  Short  Bottom-N ETFs (menor retorno predicho) — con filtro de validación

  Filtro de la pata corta (condicion AND):
    Un ETF del Bottom-N solo entra en el corto si cumple ambas condiciones:
      (1) pred_bottom < 0          (caída predicha, no solo underperformance)
      (2) |pred_bottom| > pred_Top1  (magnitud de caída supera la subida del mejor long)
    Si no cumple ambas, su peso es 0 (semana long-only parcial o total).

  Universo: todos los ETFs del panel (sin filtro de beta). La selección
    Top-N / Bottom-N se aplica sobre el universo completo de 11 ETFs.

  Ponderación (N = activos que pasaron los filtros, entre 3 y 6):
    1. Peso base:     w_i = 1/N  (positivo para longs, negativo para shorts)
    2. Kelly:         hk_i = KELLY_FRACTION * |pred_i| / var_i  (diagonal, por activo)
    3. Peso ajustado: a_i  = w_i * hk_i
    4. Normalización: w_final_i = a_i / sum(|a_j|)  → suma 100 % del capital

  donde var_i es la varianza histórica causal del activo i (ventana
  KELLY_LOOKBACK_WEEKS semanas).  Fallback a pesos iguales 1/N si todos los
  Kelly son cero.

Retorno semanal = sum(w_long * ret_long_t+1) − sum(|w_short| * ret_short_t+1)


Anti-leakage:
  - Retornos de la semana t+1: prices.reindex(dates).pct_change().shift(-1)
  - var_i para Kelly: sólo precios semanales <= t (causal,
    ventana KELLY_LOOKBACK_WEEKS)
  - Predicciones: ya causales del walk-forward expansivo (date < t)
  - Costes de transacción cobrados en el periodo t+1 sobre el turnover
    respecto al portfolio de t (nunca se mira a t+2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TOP_N, BOTTOM_N, OOS_START, OOS_END,
    KELLY_FRACTION, KELLY_LOOKBACK_WEEKS, COST_BPS,
)
from utils import weekly_forward_returns

# Nº de periodos por año (semanas W-FRI) para anualizar métricas.
WEEKS_PER_YEAR = 52


# ── Metricas ──────────────────────────────────────────────────────────────────

def cagr(returns: pd.Series) -> float:
    n_years = len(returns) / WEEKS_PER_YEAR
    cumret  = (1 + returns).prod()
    return cumret ** (1 / n_years) - 1 if n_years > 0 else np.nan

def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(WEEKS_PER_YEAR)

def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / WEEKS_PER_YEAR
    return (
        (excess.mean() * WEEKS_PER_YEAR) / (returns.std() * np.sqrt(WEEKS_PER_YEAR))
        if returns.std() > 0 else np.nan
    )

def max_drawdown(returns: pd.Series) -> float:
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def calmar(returns: pd.Series) -> float:
    md = max_drawdown(returns)
    return cagr(returns) / abs(md) if md != 0 else np.nan

def performance_summary(returns: pd.Series, label: str = "") -> dict:
    return {
        "Label"      : label,
        "CAGR"       : f"{cagr(returns):.2%}",
        "Vol (ann)"  : f"{annualized_vol(returns):.2%}",
        "Sharpe"     : f"{sharpe(returns):.2f}",
        "Max DD"     : f"{max_drawdown(returns):.2%}",
        "Calmar"     : f"{calmar(returns):.2f}",
        "Best Week"  : f"{returns.max():.2%}",
        "Worst Week" : f"{returns.min():.2%}",
    }


# ── Kelly diagonal por activo ─────────────────────────────────────────────────

def simple_kelly_weights(
    long_etfs: list,
    short_etfs: list,
    pred_by_etf: pd.Series,
    prices: pd.DataFrame,
    current_date: pd.Timestamp,
    lookback: int = KELLY_LOOKBACK_WEEKS,
    kelly_fraction: float = KELLY_FRACTION,
) -> tuple:
    """
    Ponderación Kelly diagonal (por activo) para una cartera long-short.

    Algoritmo:
      N   = len(long_etfs) + len(short_etfs)   (activos que pasaron los filtros)

      Para cada activo i:
        var_i   = varianza histórica causal de retornos SEMANALES (ventana lookback)
        hk_i    = kelly_fraction * |pred_i| / var_i    (escalar Kelly positivo)
        base_i  = 1/N  (positivo para longs, negativo para shorts)
        adj_i   = base_i * hk_i

      Normalización: w_i = adj_i / sum(|adj_j|)  → pesos suman 100 % del capital

    Diferencia vs kelly_weights_multiasset:
      - No usa la matriz de covarianza completa Sigma_eff ni inversión matricial.
      - Solo utiliza la varianza individual de cada activo (diagonal de Sigma).
      - Mas robusto con pocos activos y estimaciones ruidosas de mu.

    Fallback: pesos iguales (1/N longs, -1/N shorts) si la suma de |adj| es 0.

    Returns:
        (long_w, short_w)  par de pd.Series con valores >= 0 cuya suma conjunta
                           es exactamente 1 (normalizados sobre valor absoluto).
    """
    all_etfs = long_etfs + short_etfs
    N = len(all_etfs)

    hist_ret = prices[all_etfs].pct_change().dropna()
    hist_ret = hist_ret.loc[hist_ret.index <= current_date].tail(lookback)

    adjusted = {}
    for i, etf in enumerate(all_etfs):
        pred = float(pred_by_etf[etf])
        if etf in hist_ret.columns:
            col = hist_ret[etf]
            var = float(col.iloc[:, 0].var() if isinstance(col, pd.DataFrame) else col.var())
        else:
            var = 0.0
        var  = max(var, 1e-8)
        hk   = kelly_fraction * abs(pred) / var
        sign = 1.0 if i < len(long_etfs) else -1.0
        adjusted[etf] = sign * (1.0 / N) * hk

    total_abs = sum(abs(v) for v in adjusted.values())
    if total_abs <= 0:
        adjusted = {
            etf: (1.0 / N if i < len(long_etfs) else -1.0 / N)
            for i, etf in enumerate(all_etfs)
        }
        total_abs = 1.0

    w = {etf: v / total_abs for etf, v in adjusted.items()}

    long_w  = pd.Series({e: w[e]       for e in long_etfs})
    short_w = pd.Series({e: abs(w[e])  for e in short_etfs})
    return long_w, short_w


# ── Portfolio Builder ─────────────────────────────────────────────────────────

def build_portfolio(
    predictions_path: str,
    prices_path: str = f"{DATA_DIR}/etf_prices.csv",
    transaction_costs: bool = True,
    kelly_fraction: float = KELLY_FRACTION,
    kelly_lookback: int = KELLY_LOOKBACK_WEEKS,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Construye retornos SEMANALES del portafolio Long-Short con Kelly diagonal.

    Las predicciones son semanales (W-FRI) y se utilizan todas ellas — cada
    viernes se toma una decisión de asignación con información <= t y se cobra
    el retorno realizado entre t y t+1.

    Para cada semana t:

      LONG  — Top-N ETFs (mayor predicted_return)
      SHORT — Bottom-N ETFs que cumplen: pred<0 AND |pred|>pred_Top1
              (desactivado si long_only=True)

      Ponderacion con simple_kelly_weights() — Kelly diagonal:
        N     = activos que pasaron filtros (TOP_N a TOP_N+BOTTOM_N)
        hk_i  = KELLY_FRACTION * |pred_i| / var_i   (por activo, causal)
        w_i   = (1/N * hk_i) / sum(|1/N * hk_j|)   → suma 100 % capital

      Retorno = sum(w_long * ret_long_t+1) - sum(|w_short| * ret_short_t+1)
      donde ret_t+1 = retorno semanal entre viernes t y viernes t+1

    Costes de transacción (si transaction_costs=True):
      COST_BPS por nombre que entra o sale del portfolio, normalizado por
      n_positions_held. Aplicado sobre el retorno del periodo t+1.

    Anti-leakage:
      - Retornos t+1: prices.reindex(dates).pct_change().shift(-1)
      - var_i para Kelly: sólo precios semanales <= t (KELLY_LOOKBACK_WEEKS)
      - Predicciones: ya causales del walk-forward expansivo (date < t)
      - Turnover: referencia al portfolio anterior (t-1), nunca a t+1
    """
    preds  = pd.read_csv(predictions_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    # Universo de fechas de decisión = fechas únicas en predicciones (W-FRI)
    pred_dates     = sorted(preds["date"].unique())
    actual_returns = weekly_forward_returns(prices, pred_dates)
    actual_returns.index.name = "date"

    n_aligned = actual_returns.notna().any(axis=1).sum()
    print(f"  [Validacion] {n_aligned}/{len(pred_dates)} semanas con retorno t+1 disponible "
          f"({n_aligned/len(pred_dates):.1%})")

    results    = []
    prev_long  = set()
    prev_short = set()

    short_n = 0 if long_only else BOTTOM_N

    for date, group in preds.groupby("date"):

        pred_by_etf = group.set_index("etf")["predicted_return"]

        # ── Seleccion de candidatos sobre universo filtrado ───────────────────
        top_etfs    = group.nsmallest(TOP_N,   "rank")["etf"].tolist()
        bottom_etfs = (
            group.nlargest(short_n, "rank")["etf"].tolist() if short_n > 0 else []
        )

        # ── Retornos reales t+1 (anti-leakage) ───────────────────────────────
        def _fetch(etf_list: list) -> dict:
            out = {}
            for etf in etf_list:
                try:
                    r = actual_returns.loc[date, etf]
                    if pd.notna(r):
                        out[etf] = float(r)
                except KeyError:
                    pass
            return out

        long_rets  = _fetch(top_etfs)
        short_rets = _fetch(bottom_etfs)

        if not long_rets:
            continue

        long_etfs = list(long_rets.keys())

        # ── Filtro de la pata corta: condicion AND ────────────────────────────
        # Un ETF del Bottom-N solo entra en el corto si cumple AMBAS:
        #   (1) pred < 0            (caida predicha real, no solo underperformance)
        #   (2) |pred| > pred_Top1  (magnitud supera el mejor long)
        # Si no cumple alguna, su peso es 0 (semana long-only parcial o total).
        if short_rets:
            pred_top1  = float(pred_by_etf[top_etfs[0]])   # rank 1 (mejor long)
            short_etfs = [
                e for e in short_rets.keys()
                if float(pred_by_etf[e]) < 0
                and abs(float(pred_by_etf[e])) > pred_top1
            ]
            short_rets = {e: short_rets[e] for e in short_etfs}
        else:
            short_etfs = []

        # ── Kelly diagonal por activo: posiciones filtradas ──────────────────
        long_w, short_w = simple_kelly_weights(
            long_etfs, short_etfs, pred_by_etf, prices, date,
            lookback=kelly_lookback, kelly_fraction=kelly_fraction,
        )

        # Convertir a dict para indexacion escalar segura en pandas 2.x
        lw = long_w.to_dict()
        sw = short_w.to_dict() if len(short_w) else {}

        long_return  = float(sum(lw[e] * long_rets[e]  for e in long_etfs))
        short_return = float(sum(sw[e] * short_rets[e] for e in short_etfs))
        port_ret     = long_return - short_return

        # ── Costos de transaccion (turnover sobre posiciones con peso > 0) ────
        if transaction_costs:
            new_long  = {e for e in long_etfs  if lw[e] > 0}
            new_short = {e for e in short_etfs if sw.get(e, 0) > 0}
            to_long   = len(new_long.symmetric_difference(prev_long))
            to_short  = len(new_short.symmetric_difference(prev_short))
            n_held    = len(new_long) + len(new_short) or 1
            port_ret -= (to_long + to_short) * (COST_BPS / 10_000) / n_held
            prev_long  = new_long
            prev_short = new_short

        # ── SPY benchmark (retorno t+1) ───────────────────────────────────────
        try:
            spy_ret = actual_returns.loc[date, "SPY"]
            spy_ret = float(spy_ret) if pd.notna(spy_ret) else np.nan
        except KeyError:
            spy_ret = np.nan

        active_long  = [e for e in long_etfs  if lw[e] > 0]
        active_short = [e for e in short_etfs if sw.get(e, 0) > 0]

        results.append({
            "date"            : date,
            "portfolio_return": port_ret,
            "long_return"     : long_return,
            "short_return"    : short_return,
            "spy_return"      : spy_ret,
            "long_holdings"   : ",".join(active_long),
            "short_holdings"  : ",".join(active_short),
            "n_short"         : len(active_short),
            "n_short_filtered": max(0, short_n - len(short_etfs)),
        })

    df = pd.DataFrame(results).set_index("date")
    df.sort_index(inplace=True)

    weeks_with_short = int((df["n_short"] > 0).sum())
    weeks_long_only  = int((df["n_short"] == 0).sum())
    total_filtered   = int(df["n_short_filtered"].sum())
    weeks_partial    = int(((df["n_short"] > 0) & (df["n_short_filtered"] > 0)).sum())
    mode_str = "long-only" if long_only else f"long-short top{TOP_N}/bottom{BOTTOM_N}"
    print(f"  [Portafolio] {len(df)} semanas | Kelly diagonal {mode_str}")
    if not long_only:
        print(f"  [Filtro corto] ETFs bottom filtrados (pred>=0 o |pred|<=pred_Top1): "
              f"{total_filtered} en total "
              f"| semanas long-only: {weeks_long_only} "
              f"| semanas corto parcial: {weeks_partial} "
              f"| semanas corto completo: {weeks_with_short - weeks_partial}")
    return df


# ── Analisis In-Sample ────────────────────────────────────────────────────────

def is_portfolio(
    predictions_path: str,
    prices_path: str = f"{DATA_DIR}/etf_prices.csv",
    is_end: str = "2019-12-31",
) -> pd.DataFrame:
    """Portafolio IS sin costos para medir la señal pura del modelo."""
    preds = pd.read_csv(predictions_path, parse_dates=["date"])
    preds = preds[preds["date"] <= is_end]
    if preds.empty:
        return pd.DataFrame()
    temp_path = f"{DATA_DIR}/_is_preds.csv"
    preds.to_csv(temp_path, index=False)
    return build_portfolio(temp_path, prices_path, transaction_costs=False)


# ── Visualizacion ─────────────────────────────────────────────────────────────

def plot_cumulative(results: dict, title: str = "Backtest Results"):
    """
    results: {label: pd.Series de retornos SEMANALES}
    3 paneles: retorno acumulado, drawdown, rolling Sharpe (52 semanas).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    for label, rets in results.items():
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=label, lw=2)
    ax.set_title("Retorno Acumulado")
    ax.set_ylabel("Valor (base 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, rets in results.items():
        cum = (1 + rets).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, label=label)
    ax.set_title("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for label, rets in results.items():
        rs = rets.rolling(WEEKS_PER_YEAR).apply(
            lambda x: (x.mean() * WEEKS_PER_YEAR) / (x.std() * np.sqrt(WEEKS_PER_YEAR))
            if x.std() > 0 else np.nan
        )
        ax.plot(rs.index, rs.values, label=label, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Rolling Sharpe ({WEEKS_PER_YEAR}w)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{DATA_DIR}/backtest_chart.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> Grafico guardado: {out}")


def print_metrics_table(metrics_list: list):
    df = pd.DataFrame(metrics_list).set_index("Label")
    print("\n" + "=" * 65)
    print(df.to_string())
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_names = ["LightGBM", "RandomForest"]
    prices_path = f"{DATA_DIR}/etf_prices.csv"

    metrics_list = []
    plot_data    = {}
    oos_df       = None

    for model_name in model_names:
        pred_path = f"{DATA_DIR}/predictions_{model_name}.csv"
        if not os.path.exists(pred_path):
            print(f"[!] No encontrado: {pred_path}. Ejecuta primero 04_walk_forward_training.py")
            continue

        print(f"\n{'='*60}")
        print(f" Modelo: {model_name}  |  Kelly Diagonal Top{TOP_N}/Bottom{BOTTOM_N}")
        print(f"{'='*60}")

        oos_df = build_portfolio(pred_path, prices_path)
        label  = f"{model_name} (OOS)"
        plot_data[label] = oos_df["portfolio_return"]
        metrics_list.append(performance_summary(oos_df["portfolio_return"], label))

        is_df = is_portfolio(pred_path, prices_path)
        if not is_df.empty:
            label_is = f"{model_name} (IS)"
            metrics_list.append(performance_summary(is_df["portfolio_return"], label_is))

        long_counts = (
            pd.Series(",".join(oos_df["long_holdings"]).split(","))
            .value_counts().head(5)
        )
        short_str    = ",".join(oos_df["short_holdings"].dropna())
        short_counts = (
            pd.Series(short_str.split(",")).value_counts().head(5)
            if short_str else pd.Series(dtype=int)
        )

        print(f"[OOS] ETFs mas frecuentes LONG:\n{long_counts.to_string()}")
        if not short_counts.empty:
            print(f"[OOS] ETFs mas frecuentes SHORT:\n{short_counts.to_string()}")
        else:
            print("[OOS] Kelly asigno peso 0 a todos los cortos en todas las semanas")

    if oos_df is not None:
        # SPY retornos t→t+1 ya los calcula build_portfolio (columna 'spy_return')
        # alineados semana a semana con el portafolio, bajo la misma convención.
        spy_oos = oos_df["spy_return"].dropna()
        plot_data["SPY (OOS)"] = spy_oos
        metrics_list.append(performance_summary(spy_oos, "SPY (OOS)"))

    print_metrics_table(metrics_list)
    if plot_data:
        plot_cumulative(
            plot_data,
            title=f"Sector Rotation — Kelly Diagonal Top{TOP_N}/Bottom{BOTTOM_N} vs SPY  [OOS {OOS_START[:4]}-{OOS_END[:4]}]",
        )

    print("[OK] Backtest completado.")
