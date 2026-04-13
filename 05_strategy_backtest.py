"""
05_strategy_backtest.py
=======================
Construye el portafolio mensual a partir de las predicciones OOS semanales
y calcula metricas de performance vs SPY.

Predicciones: frecuencia semanal (W-FRI). El portafolio sigue siendo mensual:
  se filtra al último viernes de cada mes para la decisión de asignación.
  Los retornos se calculan sobre precios mensuales (cierre-mes a cierre-mes).

Estrategia Long-Short con Kelly multi-activo (matriz de covarianza completa):
  Long   Top-3 ETFs  (mayor retorno predicho)
  Short  Bottom-3 ETFs  (menor retorno predicho)  -- sin filtros adicionales

  Los 6 activos se ponderan conjuntamente con el criterio de Kelly multi-activo:

    f* = Sigma_eff^{-1} . m_eff

  donde:
    Sigma_eff[i,j] = d_i * d_j * Sigma[i,j]   (Sigma_eff = D Sigma D)
    d_i = +1 para longs, -1 para shorts
    m_eff_i = d_i * predicted_return_i          (retorno esperado de la posicion)
    Sigma    = matriz de covarianza 6x6 de retornos mensuales historicos causales (36m)
               (calculada sobre precios filtrados a frecuencia mensual)

  Se aplica half-Kelly (fraccion 0.5).  Pesos negativos se fijan a 0 (sin
  inversion de direccion).  Los 6 pesos positivos se normalizan para sumar 1
  (100 % del capital mensual).  Fallback a pesos iguales si todos son cero.

Retorno mensual = sum(w_long * ret_long_t+1) - sum(w_short * ret_short_t+1)

Anti-leakage:
  - Predicciones semanales filtradas al último viernes de cada mes
  - Retornos del mes t+1: prices_monthly.pct_change().shift(-1)
  - Sigma para Kelly: solo precios mensuales <= t (causal, ventana de 36 meses)
  - Predicciones: ya causales del walk-forward expansivo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, TOP_N, BOTTOM_N, OOS_START, OOS_END, KELLY_FRACTION

COST_BPS = 10   # costos de transaccion (basis points por operacion, cada leg)


# ── Metricas ──────────────────────────────────────────────────────────────────

def cagr(returns: pd.Series) -> float:
    n_years = len(returns) / 12
    cumret  = (1 + returns).prod()
    return cumret ** (1 / n_years) - 1

def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(12)

def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / 12
    return (excess.mean() * 12) / (returns.std() * np.sqrt(12)) if returns.std() > 0 else np.nan

def max_drawdown(returns: pd.Series) -> float:
    cum  = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def calmar(returns: pd.Series) -> float:
    md = max_drawdown(returns)
    return cagr(returns) / abs(md) if md != 0 else np.nan

def performance_summary(returns: pd.Series, label: str = "") -> dict:
    return {
        "Label"       : label,
        "CAGR"        : f"{cagr(returns):.2%}",
        "Vol (ann)"   : f"{annualized_vol(returns):.2%}",
        "Sharpe"      : f"{sharpe(returns):.2f}",
        "Max DD"      : f"{max_drawdown(returns):.2%}",
        "Calmar"      : f"{calmar(returns):.2f}",
        "Best Month"  : f"{returns.max():.2%}",
        "Worst Month" : f"{returns.min():.2%}",
    }


# ── Kelly Criterion multi-activo ──────────────────────────────────────────────

def kelly_weights_multiasset(
    long_etfs: list,
    short_etfs: list,
    pred_by_etf: pd.Series,
    prices: pd.DataFrame,
    current_date: pd.Timestamp,
    lookback: int = 36,
    kelly_fraction: float = KELLY_FRACTION,
    reg: float = 1e-4,
) -> tuple:
    """
    Criterio de Kelly multi-activo para una cartera long-short de n activos.

    Formula:
        f* = Sigma_eff^{-1} . m_eff

    donde:
      d_i        direccion de la posicion: +1 (long) o -1 (short)
      m_eff_i    retorno esperado de la posicion i:  d_i * predicted_return_i
                   Long:  m_eff = +predicted_return   (>0 si modelo ve subida)
                   Short: m_eff = -predicted_return   (>0 si modelo ve bajada)
      Sigma_eff  covarianza de los retornos efectivos de posicion:
                   Sigma_eff[i,j] = d_i * d_j * Sigma[i,j]
                   En forma matricial: Sigma_eff = D . Sigma . D
                   donde D = diag(d_1,...,d_n)
                 Esto captura que dos activos correlacionados en la misma
                 direccion se penalizan entre si (reducen pesos mutuamente),
                 y que un long y un short correlacionados se benefician.
      Sigma      covarianza 6x6 de retornos mensuales historicos causales
                 (precios ya filtrados a frecuencia mensual, ventana de `lookback` meses)

    Post-procesado:
      - Half-Kelly: f = kelly_fraction * f*  (0.5 por defecto)
      - Pesos negativos -> 0 (Kelly no puede revertir la direccion asignada)
      - Normalizacion: los 6 pesos positivos suman 1 (100 % del capital)
      - Fallback a pesos iguales (1/n) si la suma es <= 0

    Por que half-Kelly:
      El Kelly completo maximiza el crecimiento logaritmico asintoticamente
      pero es muy sensible a errores en la estimacion de mu.  Half-Kelly
      reduce el drawdown ~50 % a costa de ~25 % menos de retorno esperado,
      y es el estandar en gestion cuantitativa con estimaciones ruidosas.

    Causalidad (anti look-ahead):
      Sigma se estima sobre hist_ret.index <= current_date — nunca se usan
      precios posteriores a la fecha de decision.

    Params:
        long_etfs      lista de tickers de la pata larga (Top-N)
        short_etfs     lista de tickers de la pata corta (Bottom-N)
        pred_by_etf    pd.Series (index=ticker) con predicted_return del modelo
        prices         DataFrame de precios (index=fecha, cols=tickers)
        current_date   fecha de decision (causal: Sigma solo hasta esta fecha)
        lookback       meses de historia para estimar Sigma  (default: 36)
        kelly_fraction fraccion del Kelly completo  (default: KELLY_FRACTION=0.5 en config.py)
        reg            regularizacion de la diagonal para evitar singularidad

    Returns:
        (long_w, short_w)  par de pd.Series con pesos >= 0 que suman <= 1
                           y cuya suma conjunta es exactamente 1.
    """
    all_etfs = long_etfs + short_etfs
    n = len(all_etfs)

    # Vector de direcciones: +1 para longs, -1 para shorts
    d = np.array([1.0] * len(long_etfs) + [-1.0] * len(short_etfs))

    # Retornos historicos causales — solo precios <= current_date
    hist_ret = prices[all_etfs].pct_change().dropna()
    hist_ret = hist_ret.loc[hist_ret.index <= current_date].tail(lookback)

    # Matriz de covarianza de retornos crudos (n x n)
    Sigma = hist_ret.cov().values

    # Sigma_eff = D . Sigma . D  (Sigma_eff[i,j] = d_i * d_j * Sigma[i,j])
    D_mat     = np.diag(d)
    Sigma_eff = D_mat @ Sigma @ D_mat

    # Regularizacion: evita singularidad cuando activos estan muy correlacionados
    Sigma_eff += reg * np.eye(n)

    # Retornos esperados efectivos de cada posicion: m_eff_i = d_i * pred_i
    m_eff = d * pred_by_etf[all_etfs].values

    # f* = Sigma_eff^{-1} . m_eff  (solve es mas estable que inv)
    try:
        f_star = np.linalg.solve(Sigma_eff, m_eff)
    except np.linalg.LinAlgError:
        f_star = np.linalg.lstsq(Sigma_eff, m_eff, rcond=None)[0]

    # Half-Kelly y eliminacion de pesos negativos (sin inversion de direccion)
    f = np.clip(kelly_fraction * f_star, 0.0, None)

    total = f.sum()
    if total <= 0:
        # Fallback: pesos iguales si Kelly no produce ninguna señal positiva
        f = np.ones(n) / n
    else:
        f /= total

    f_series = pd.Series(f, index=all_etfs)
    return f_series[long_etfs], f_series[short_etfs]


# ── Portfolio Builder ─────────────────────────────────────────────────────────

def build_portfolio(
    predictions_path: str,
    prices_path: str = f"{DATA_DIR}/etf_prices.csv",
    transaction_costs: bool = True,
    kelly_fraction: float = KELLY_FRACTION,
    kelly_lookback: int = 36,
) -> pd.DataFrame:
    """
    Construye retornos mensuales del portafolio Long-Short con Kelly multi-activo.

    Las predicciones son semanales (W-FRI); se filtran al último viernes de cada
    mes para obtener la señal mensual de asignación. Los precios también se
    filtran a esas mismas fechas mensuales para calcular retornos mes-a-mes
    y estimar la covarianza de Kelly correctamente.

    Para cada mes t (último viernes del mes):

      LONG  — Top-N ETFs (mayor predicted_return)
      SHORT — Bottom-N ETFs (menor predicted_return) — sin filtros adicionales

      Ponderacion conjunta con kelly_weights_multiasset():
        f* = Sigma_eff^{-1} . m_eff   (half-Kelly, normalizados a suma 1)
        Sigma_eff = D . Sigma . D   donde D = diag(+1,...,+1,-1,...,-1)
        Sigma estimada sobre los ultimos kelly_lookback meses causales (<= t)
        usando precios_monthly (retornos mensuales)

      Retorno = sum(w_long * ret_long_t+1) - sum(w_short * ret_short_t+1)
      donde ret_t+1 = retorno mensual entre último viernes de mes t y mes t+1

    Anti-leakage:
      - Predicciones semanales filtradas al último viernes de cada mes
      - Retornos del mes t+1: prices_monthly.pct_change().shift(-1)
      - Sigma para Kelly: solo precios_monthly <= t  (causal, 36 meses)
      - Predicciones: ya causales del walk-forward expansivo
    """
    preds  = pd.read_csv(predictions_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    # Filtrar predicciones semanales al último viernes de cada mes
    preds["_month"] = preds["date"].dt.to_period("M")
    preds = preds.loc[
        preds.groupby("_month")["date"].transform("max") == preds["date"]
    ].drop(columns=["_month"])

    # Filtrar precios a las mismas fechas mensuales (anti-leakage: retornos mes-a-mes)
    monthly_dates  = sorted(preds["date"].unique())
    prices_monthly = prices.reindex(monthly_dates).ffill()

    # Retornos mensuales t+1 (retorno entre mes t y mes t+1)
    actual_returns = prices_monthly.pct_change().shift(-1)
    actual_returns.index.name = "date"

    pred_dates = preds["date"].unique()
    n_aligned  = sum(d in actual_returns.index for d in pred_dates)
    print(f"  [Validacion] {n_aligned}/{len(pred_dates)} periodos mensuales alineados "
          f"({n_aligned/len(pred_dates):.1%})")

    results    = []
    prev_long  = set()
    prev_short = set()

    for date, group in preds.groupby("date"):

        pred_by_etf = group.set_index("etf")["predicted_return"]

        # ── Seleccion de candidatos (sin filtros) ─────────────────────────────
        top_etfs    = group.nsmallest(TOP_N,    "rank")["etf"].tolist()
        bottom_etfs = group.nlargest( BOTTOM_N, "rank")["etf"].tolist()

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

        long_etfs  = list(long_rets.keys())
        short_etfs = list(short_rets.keys())

        # ── Kelly multi-activo: 6 posiciones juntas ───────────────────────────
        # Se pasan precios mensuales: tail(36) = 36 meses de historia
        long_w, short_w = kelly_weights_multiasset(
            long_etfs, short_etfs, pred_by_etf, prices_monthly, date,
            lookback=kelly_lookback, kelly_fraction=kelly_fraction,
        )

        # Convertir a dict para indexacion escalar segura en pandas 2.x
        lw = long_w.to_dict()
        sw = short_w.to_dict()

        long_return  = float(sum(lw[e] * long_rets[e]  for e in long_etfs))
        short_return = float(sum(sw[e] * short_rets[e] for e in short_etfs))
        port_ret     = long_return - short_return

        # ── Costos de transaccion (turnover sobre posiciones con peso > 0) ────
        if transaction_costs:
            new_long  = {e for e in long_etfs  if lw[e] > 0}
            new_short = {e for e in short_etfs if sw[e] > 0}
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

        # ETFs con peso efectivo > 0 para el registro
        active_long  = [e for e in long_etfs  if lw[e] > 0]
        active_short = [e for e in short_etfs if sw[e] > 0]

        results.append({
            "date"            : date,
            "portfolio_return": port_ret,
            "long_return"     : long_return,
            "short_return"    : short_return,
            "spy_return"      : spy_ret,
            "long_holdings"   : ",".join(active_long),
            "short_holdings"  : ",".join(active_short),
            "n_short"         : len(active_short),
        })

    df = pd.DataFrame(results).set_index("date")
    df.sort_index(inplace=True)

    months_with_short = int((df["n_short"] > 0).sum())
    months_long_only  = int((df["n_short"] == 0).sum())
    print(f"  [Portafolio] {len(df)} periodos mensuales | "
          f"Kelly multi-activo top{TOP_N}/bottom{BOTTOM_N} "
          f"| corto activo: {months_with_short}  "
          f"long-only (Kelly=0 en cortos): {months_long_only}")
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
    results: {label: pd.Series de retornos mensuales}
    3 paneles: retorno acumulado, drawdown, rolling Sharpe (12m).
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
        rs = rets.rolling(12).apply(
            lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else np.nan
        )
        ax.plot(rs.index, rs.values, label=label, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling Sharpe (12m)")
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
    model_names = ["LightGBM", "RandomForest", "GradientBoosting"]
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
        print(f" Modelo: {model_name}  |  Kelly Multi-Activo Top{TOP_N}/Bottom{BOTTOM_N}")
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
            print("[OOS] Kelly asigno peso 0 a todos los cortos en todos los meses")

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    if oos_df is not None:
        # SPY retornos mensuales alineados con el portafolio (mismo índice mensual)
        spy_monthly = prices["SPY"].reindex(oos_df.index).ffill()
        spy_oos     = spy_monthly.pct_change().dropna()
        spy_oos     = spy_oos.reindex(oos_df.index).dropna()
        plot_data["SPY (OOS)"] = spy_oos
        metrics_list.append(performance_summary(spy_oos, "SPY (OOS)"))

    print_metrics_table(metrics_list)
    if plot_data:
        plot_cumulative(
            plot_data,
            title=f"Sector Rotation — Kelly Multi-Activo Top{TOP_N}/Bottom{BOTTOM_N} vs SPY  [OOS {OOS_START[:4]}-{OOS_END[:4]}]",
        )

    print("[OK] Backtest completado.")
