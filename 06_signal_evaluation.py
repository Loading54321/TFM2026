"""
06_signal_evaluation.py
=======================
Evaluacion de la calidad de las senales de prediccion del modelo.
Frecuencia de predicciones: semanal (W-FRI).

Basado en el notebook de Stefan Jansen:
  github.com/stefan-jansen/machine-learning-for-trading/blob/main/
  12_gradient_boosting_machines/06_evaluate_trading_signals.ipynb

Metricas implementadas (sin alphalens, puro pandas/scipy):
  1. IC global (Information Coefficient)
     Spearman rank-correlation entre predicted_return y target (retorno t+1).
     IC > 0.05 se considera util en la practica; IC > 0.10 es excelente.

  2. IC Rolling (ventana 52 semanas = 1 año)
     Estabilidad temporal de la senial: un IC que decae sistematicamente
     indica que el modelo pierde poder predictivo en el OOS reciente.

  3. IC por regimen de mercado
     Mide si el modelo funciona mejor en Bull, Ranging o Bear.
     Informacion util para ajustar la estrategia segun el ciclo.

  4. Analisis por quintiles (quantile returns)
     Divide los ETFs predichos en 5 grupos ordenados por predicted_return
     y calcula el retorno medio de cada grupo. Una buena senial muestra
     retornos monotonamente crecientes del Q1 (peor predicho) al Q5 (mejor).

  5. Hit Rate (% de veces que el signo de la prediccion acierta)
     Complementa el IC: un modelo puede tener IC bajo pero hit rate alto.

Modelos evaluados:
  LightGBM, RandomForest, RegimeLGBM (3 LGBM por regimen HMM)

Genera:
  data/signal_evaluation_IC_{model}.csv        IC semanal por modelo
  data/signal_evaluation_quintiles_{model}.csv Retornos por quintil
  data/signal_evaluation_plot.png              Grafico con los 4 paneles

Ejecucion:
  python 06_signal_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, OOS_START, OOS_END, TRAIN_START, TRAIN_END
from regime_model import REGIME_NAMES


# ── Calculo de IC ─────────────────────────────────────────────────────────────

def compute_ic(preds: pd.DataFrame) -> pd.Series:
    """
    IC semanal = Spearman(predicted_return, target) por fecha.

    El IC mide la correlacion de rango entre la prediccion del modelo
    y el retorno real de la semana siguiente. Valores tipicos en la literatura:
      IC > 0.05  : senial debilmente util
      IC > 0.10  : senial moderadamente util
      IC > 0.15  : senial fuerte (raro en mercados eficientes)

    Referencia: Jansen (2020) cap. 12, Grinold & Kahn (2000) "Active Portfolio
    Management" — la Ley Fundamental de la Gestion Activa:
      IR = IC * sqrt(N)
    donde N = numero de activos evaluados por periodo.
    """
    ic_series = {}
    for date, group in preds.groupby("date"):
        if group["target"].isna().all() or group["predicted_return"].isna().all():
            continue
        g = group.dropna(subset=["predicted_return", "target"])
        if len(g) < 3:
            continue
        rho, _ = stats.spearmanr(g["predicted_return"], g["target"])
        ic_series[date] = rho
    return pd.Series(ic_series, name="IC")


def ic_summary(ic: pd.Series, label: str = "") -> dict:
    """
    Resumen estadistico del IC: media, std, t-stat, hit-rate, ICIR.

    ICIR (Information Coefficient Information Ratio):
      ICIR = mean(IC) / std(IC)
      > 0.5 indica consistencia; > 1.0 es excelente.
    """
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        return {}
    mean_ic  = ic_clean.mean()
    std_ic   = ic_clean.std()
    t_stat   = mean_ic / (std_ic / np.sqrt(len(ic_clean))) if std_ic > 0 else np.nan
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    hit_rate = (ic_clean > 0).mean()
    return {
        "Model"    : label,
        "IC Mean"  : f"{mean_ic:.4f}",
        "IC Std"   : f"{std_ic:.4f}",
        "ICIR"     : f"{icir:.3f}",
        "t-stat"   : f"{t_stat:.2f}",
        "Hit Rate" : f"{hit_rate:.1%}",
        "N weeks"  : len(ic_clean),
    }


# ── IC por regimen ─────────────────────────────────────────────────────────────

def ic_by_regime(preds: pd.DataFrame) -> pd.DataFrame:
    """
    IC medio por regimen de mercado.

    Detecta automaticamente la columna de regimen:
      'regime'        -> columna de RegimeLGBM (04b_regime_walk_forward.py)
      'market_regime' -> columna de RF/GB global (04_walk_forward_training.py)
    Si ninguna existe, devuelve DataFrame vacio.
    """
    # Detectar columna de regimen disponible
    regime_col = None
    for candidate in ("regime", "market_regime"):
        if candidate in preds.columns:
            regime_col = candidate
            break
    if regime_col is None:
        return pd.DataFrame()

    records = []
    for date, group in preds.groupby("date"):
        g = group.dropna(subset=["predicted_return", "target"])
        if len(g) < 3:
            continue
        rho, _ = stats.spearmanr(g["predicted_return"], g["target"])
        regime = int(g[regime_col].mode()[0]) if not g[regime_col].isna().all() else -1
        records.append({"date": date, "IC": rho, "regime": regime})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    summary = (
        df.groupby("regime", observed=True)["IC"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "IC Mean", "std": "IC Std", "count": "N"})
        .reset_index()
    )
    summary["Regime"] = summary["regime"].map(REGIME_NAMES).fillna("Unknown")
    return summary[["Regime", "IC Mean", "IC Std", "N"]]


# ── Analisis por quintiles ────────────────────────────────────────────────────

def quantile_returns(preds: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Retorno medio real por quintil de prediccion.

    Cada mes, los ETFs se clasifican en n_quantiles grupos segun
    predicted_return. Se calcula el retorno real medio de cada grupo.
    Una buena senial muestra retornos monotonamente crecientes de Q1 a Q5.

    Q1 = ETFs con menor predicted_return (candidatos SHORT)
    Q5 = ETFs con mayor predicted_return (candidatos LONG)

    Nota: se usa rank(method='first') para romper empates antes de qcut,
    garantizando exactamente n_quantiles bins incluso con pocos activos
    (9 ETFs) o valores repetidos en predicted_return.
    """
    records = []
    for date, group in preds.groupby("date"):
        g = group.dropna(subset=["predicted_return", "target"])
        if len(g) < n_quantiles:
            continue
        g = g.copy()
        # rank(method='first') elimina empates -> qcut produce exactamente n_quantiles bins
        g["quintile"] = pd.qcut(
            g["predicted_return"].rank(method="first"),
            n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
        )
        for q, sub in g.groupby("quintile", observed=True):
            records.append({"date": date, "quintile": str(q), "return": sub["target"].mean()})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.groupby("quintile")["return"].agg(["mean", "std", "count"]).reset_index()


# ── Hit Rate por semana ───────────────────────────────────────────────────────

def hit_rate(preds: pd.DataFrame) -> pd.Series:
    """
    % de ETFs cuyo signo de prediccion coincide con el signo del retorno real.
    Complementa el IC: hit rate > 55% con IC > 0 indica senial util y consistente.
    """
    records = {}
    for date, group in preds.groupby("date"):
        g = group.dropna(subset=["predicted_return", "target"])
        if len(g) == 0:
            continue
        correct = ((g["predicted_return"] > 0) == (g["target"] > 0)).mean()
        records[date] = correct
    return pd.Series(records, name="hit_rate")


# ── Visualizacion ─────────────────────────────────────────────────────────────

def plot_signal_evaluation(
    ic_dict: dict,
    quintile_dict: dict,
    save_path: str,
):
    """
    4 paneles:
      1. IC rolling 52 semanas por modelo
      2. IC semanal con banda ±1 std (rolling 52w)
      3. Retornos por quintil (Q1..Q5) — spread Q5-Q1
      4. Distribucion del IC (histograma)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Evaluacion de Senales — Information Coefficient y Analisis de Quintiles\n"
        "[OOS 2020-2024]  |  frecuencia semanal",
        fontsize=13, fontweight="bold",
    )
    colors = ["#2196f3", "#ff5722", "#4caf50", "#9c27b0"]

    # ── Panel 1: IC rolling 52 semanas ──────────────────────────────────────
    ax = axes[0, 0]
    for (label, ic), color in zip(ic_dict.items(), colors):
        ic_roll = ic.rolling(52).mean()
        ax.plot(ic_roll.index, ic_roll.values, label=label, lw=2, color=color)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(0.05, color="gray", lw=0.6, ls=":", label="IC=0.05 (umbral util)")
    ax.set_title("IC Rolling 52 semanas (1 año)")
    ax.set_ylabel("IC (Spearman)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: IC semanal con banda ────────────────────────────────────────
    ax = axes[0, 1]
    for (label, ic), color in zip(ic_dict.items(), colors):
        roll_mean = ic.rolling(52).mean()
        roll_std  = ic.rolling(52).std()
        ax.plot(ic.index, ic.values, alpha=0.3, lw=0.8, color=color)
        ax.plot(roll_mean.index, roll_mean.values, lw=1.5, color=color, label=label)
        ax.fill_between(
            roll_mean.index,
            roll_mean - roll_std, roll_mean + roll_std,
            alpha=0.12, color=color,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("IC Semanal y Media Rolling 52w ±1σ")
    ax.set_ylabel("IC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Retornos por quintil ─────────────────────────────────────────
    ax = axes[1, 0]
    x = np.arange(5)
    width = 0.35
    for i, (label, qdf) in enumerate(quintile_dict.items()):
        if qdf.empty:
            continue
        means = qdf.set_index("quintile")["mean"]
        bars = ax.bar(
            x + i * width, means.values * 100, width,
            label=label, color=colors[i], alpha=0.8,
        )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Retorno Medio Real por Quintil de Prediccion\n(Q1=peor predicho, Q5=mejor)")
    ax.set_ylabel("Retorno medio semanal (%)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Histograma IC ─────────────────────────────────────────────────
    ax = axes[1, 1]
    for (label, ic), color in zip(ic_dict.items(), colors):
        ic_clean = ic.dropna()
        ax.hist(ic_clean, bins=20, alpha=0.5, color=color, label=f"{label} (mean={ic_clean.mean():.3f})")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(0.05, color="gray", lw=0.8, ls=":", label="IC=0.05")
    ax.set_title("Distribucion del IC Semanal")
    ax.set_xlabel("IC (Spearman)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Grafico guardado: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, period: str = "OOS") -> tuple:
    """Carga predicciones de un modelo y calcula IC, quintiles y hit rate."""
    pred_path = f"{DATA_DIR}/predictions_{model_name}.csv"
    if not os.path.exists(pred_path):
        script = "04b_regime_walk_forward.py" if model_name == "RegimeLGBM" \
                 else "04_walk_forward_training.py"
        print(f"[!] No encontrado: {pred_path}. Ejecuta primero {script}")
        return None, None, None

    preds = pd.read_csv(pred_path, parse_dates=["date"])

    if period == "OOS":
        preds = preds[(preds["date"] >= OOS_START) & (preds["date"] <= OOS_END)]
        label = f"{model_name} OOS"
    elif period == "IS":
        preds = preds[(preds["date"] >= TRAIN_START) & (preds["date"] <= TRAIN_END)]
        label = f"{model_name} IS"
    else:
        label = model_name

    if preds.empty:
        print(f"  [!] Sin datos para periodo {period}")
        return None, None, None

    ic     = compute_ic(preds)
    qdf    = quantile_returns(preds)
    hr     = hit_rate(preds)

    # Resumen estadistico
    summary = ic_summary(ic, label)
    hr_mean = hr.mean()
    summary["Hit Rate (dir)"] = f"{hr_mean:.1%}"

    return ic, qdf, summary


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  EVALUACION DE SENALES — IC Analysis (Jansen 2020, cap. 12)")
    print("=" * 65 + "\n")

    # Modelos a evaluar: RF global, LGBM global y LGBM por regimen HMM
    model_names = ["LightGBM", "RandomForest", "RegimeLGBM"]

    ic_dict       = {}
    quintile_dict = {}
    summaries     = []

    for model_name in model_names:
        pred_path = f"{DATA_DIR}/predictions_{model_name}.csv"
        if not os.path.exists(pred_path):
            print(f"[!] No encontrado: {pred_path} — omitido")
            continue

        print(f"\n[{model_name}] — OOS {OOS_START[:4]}–{OOS_END[:4]}")
        ic, qdf, summary = evaluate_model(model_name, period="OOS")
        if ic is None:
            continue

        ic_dict[model_name]       = ic
        quintile_dict[model_name] = qdf
        summaries.append(summary)

        # IC por regimen (detecta columna automaticamente)
        preds_full = pd.read_csv(pred_path, parse_dates=["date"])
        preds_oos  = preds_full[
            (preds_full["date"] >= OOS_START) & (preds_full["date"] <= OOS_END)
        ]
        regime_df = ic_by_regime(preds_oos)
        if not regime_df.empty:
            print(f"\n  IC por regimen ({model_name}):")
            print(regime_df.to_string(index=False))

        # Guardar IC semanal
        ic_out = f"{DATA_DIR}/signal_evaluation_IC_{model_name}.csv"
        ic.to_csv(ic_out, header=True)
        print(f"  IC guardado: {ic_out}")

        # Guardar quintiles
        if qdf is not None and not qdf.empty:
            q_out = f"{DATA_DIR}/signal_evaluation_quintiles_{model_name}.csv"
            qdf.to_csv(q_out, index=False)
            print(f"  Quintiles guardados: {q_out}")

    # Tabla resumen
    if summaries:
        print("\n" + "=" * 65)
        summary_df = pd.DataFrame(summaries).set_index("Model")
        print(summary_df.to_string())
        print("=" * 65)
        print("\nInterpretacion del IC:")
        print("  IC Mean > 0.05  -> senial debilmente util")
        print("  IC Mean > 0.10  -> senial moderadamente util")
        print("  ICIR    > 0.50  -> IC consistente a lo largo del tiempo")
        print("  Hit Rate > 55%  -> el signo de la prediccion acierta con frecuencia")

    # Grafico comparativo (todos los modelos disponibles)
    if ic_dict:
        plot_signal_evaluation(
            ic_dict,
            quintile_dict,
            save_path=f"{DATA_DIR}/signal_evaluation_plot.png",
        )

    print("\n[OK] Evaluacion de senales completada.")
