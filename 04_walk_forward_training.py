"""
04_walk_forward_training.py
===========================
Walk-Forward con ventanas independientes para HMM y modelos ML.
Frecuencia: semanal (W-FRI). ~260 semanas OOS.

Modelos soportados
──────────────────
  LightGBM     Modelo principal (Jansen 2020, cap. 12, notebook 05).
               Histogram-based gradient boosting: rápido y con regularización
               L1/L2 integrada; superior a sklearn GB en este contexto.
  RandomForest Ensemble de árboles paralelos (baseline sólido).

Arquitectura de ventanas separadas
────────────────────────────────────
  HMM  (régimen) : ventana de contexto de HMM_CONTEXT_PERIODS = 235 semanas.
                   Determina el régimen actual usando solo las últimas ~4.5
                   años de datos del SPY — causal, sin sesgo de regímenes
                   muy antiguos.

  ML   (predicción): ventana EXPANSIVA desde TRAIN_START hasta t.
                     Usa todos los datos IS disponibles; más datos = mejor
                     generalización cross-seccional.

EDA por régimen de mercado
───────────────────────────
  Antes del walk-forward se ejecuta un análisis exploratorio que responde
  al requisito del enunciado: observar qué sectores lideran y rezagan en
  cada fase del ciclo económico (Bear / Ranging / Bull).

  Salidas: data/eda_etf_by_regime.csv  y  data/eda_etf_by_regime.png

Flujo por semana t en OOS:
  1. HMM context (235 semanas):
       X_context = spy_features[ t-235w : t ]
       forward_filter(prior_uniforme) -> regime_t  [causal, sin lookahead]
  2. panel[date=t, market_regime] = regime_t
  3. ML expansivo:
       train = panel[ TRAIN_START : t )
       fit(clone(modelo), train) -> predictions[t]
  4. Rankear ETFs -> Top-3 largo / Bottom-3 corto

Anti-leakage verificado:
  - HMM parámetros (EM) entrenados solo en IS (< OOS_START).
  - regime_t: observaciones únicamente <= t.
  - ML training: date < t en todos los casos.
  - Target = retorno t+1 (shiftado en 02).
  - Modelos ML clonados en cada iteración.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TRAIN_START, OOS_START, OOS_END,
    TOP_N, MIN_TRAIN_PERIODS, HMM_CONTEXT_PERIODS,
    RANDOM_SEED, DEV_MODE, ML_TRAIN_EXCLUDE_PERIODS,
)
from utils import load_panel, get_feature_cols, ml_train_date_kept
from models import MODELS as MODELS_ALL, build_pipeline, LGBM_AVAILABLE
from regime_model import (
    load_spy_features, fit_hmm, label_mapping,
    decode_is_states, get_regime_from_context_window,
    OBS_COLS, REGIME_NAMES,
)


# ── EDA: Rendimiento de ETFs por régimen / ciclo económico ───────────────────

def eda_etf_by_regime(
    panel: pd.DataFrame,
    save_csv : str = f"{DATA_DIR}/eda_etf_by_regime.csv",
    save_plot: str = f"{DATA_DIR}/eda_etf_by_regime.png",
) -> pd.DataFrame:
    """
    Análisis exploratorio de rendimiento de cada ETF por régimen HMM.

    Responde al requisito del enunciado: observar qué sectores lideran
    y rezagan en cada fase del ciclo económico (Bear / Ranging / Bull).
    Usa el panel IS con la columna market_regime asignada por
    03_market_regime_detection.py (Viterbi IS-only, sin leakage OOS).

    Parámetros
    ----------
    panel     : DataFrame panel long con columna 'market_regime'
    save_csv  : ruta de salida del CSV con la tabla pivotada
    save_plot : ruta de salida del gráfico de barras

    Devuelve
    --------
    DataFrame pivotado: filas = ETF, columnas = régimen (Bear/Ranging/Bull)
    con retorno medio semanal del exceso vs SPY (en %).
    """
    print("\n" + "=" * 65)
    print("  EDA — Rendimiento de ETFs por Régimen de Mercado")
    print("=" * 65)

    if "market_regime" not in panel.columns:
        print(
            "  [!] Columna 'market_regime' no encontrada en el panel.\n"
            "      Ejecuta primero 03_market_regime_detection.py.\n"
            "      Omitiendo EDA por régimen."
        )
        return pd.DataFrame()

    is_panel = panel[
        (panel["date"] >= TRAIN_START) & (panel["date"] < OOS_START)
    ].copy()
    is_panel = is_panel[is_panel["market_regime"] >= 0]

    n_periods = is_panel["date"].nunique()
    n_regimes = is_panel["market_regime"].nunique()
    print(f"\n[EDA] Período IS: {TRAIN_START} → {OOS_START}")
    print(f"[EDA] Semanas únicas: {n_periods}  |  Regímenes distintos: {n_regimes}")

    regime_weeks = is_panel.drop_duplicates("date").groupby("market_regime")["date"].count()
    print(f"\n[EDA] Distribución de regímenes (semanas IS):")
    for rid, name in REGIME_NAMES.items():
        n   = int(regime_weeks.get(rid, 0))
        pct = 100 * n / n_periods if n_periods > 0 else 0
        print(f"  {name:10s} ({rid}): {n:4d} semanas  ({pct:5.1f}%)")

    summary = (
        is_panel
        .groupby(["etf", "market_regime"])["target"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )
    summary["regime_name"] = summary["market_regime"].map(REGIME_NAMES)

    pivot_mean = (
        summary
        .pivot(index="etf", columns="regime_name", values="mean")
        [list(REGIME_NAMES.values())]
        * 100
    )

    print(f"\n[EDA] Retorno medio semanal del exceso vs SPY por ETF y régimen (%):")
    print(f"      (target = ret_ETF - ret_SPY en t+1; IS {TRAIN_START[:4]}-{OOS_START[:4]})\n")
    header = f"  {'ETF':6s}  {'Bear':>10s}  {'Ranging':>10s}  {'Bull':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for etf in pivot_mean.index:
        bear    = pivot_mean.loc[etf, "Bear"]    if "Bear"    in pivot_mean.columns else np.nan
        ranging = pivot_mean.loc[etf, "Ranging"] if "Ranging" in pivot_mean.columns else np.nan
        bull    = pivot_mean.loc[etf, "Bull"]    if "Bull"    in pivot_mean.columns else np.nan
        print(f"  {etf:6s}  {bear:>+10.2f}%  {ranging:>+10.2f}%  {bull:>+10.2f}%")

    print(f"\n[EDA] Sectores líderes / rezagados por régimen (exceso vs SPY):")
    for col in ["Bear", "Ranging", "Bull"]:
        if col not in pivot_mean.columns:
            continue
        s        = pivot_mean[col].sort_values(ascending=False)
        leaders  = ", ".join(f"{e}({v:+.2f}%)" for e, v in s.head(3).items())
        laggards = ", ".join(f"{e}({v:+.2f}%)" for e, v in s.tail(3).items())
        print(f"  {col:10s} — Líderes:   {leaders}")
        print(f"  {'':10s}   Rezagados: {laggards}")

    pivot_mean.to_csv(save_csv)
    print(f"\n[EDA] Tabla guardada: {save_csv}")

    _plot_eda(pivot_mean, save_plot)

    return pivot_mean


def _plot_eda(pivot_mean: pd.DataFrame, save_path: str):
    """
    Gráfico de barras agrupadas: retorno medio semanal por ETF y régimen.
    Paleta coherente con el gráfico de regímenes del módulo 03.
    """
    if pivot_mean.empty:
        return

    regime_colors = {"Bear": "#e63946", "Ranging": "#adb5bd", "Bull": "#2dc653"}
    etfs    = list(pivot_mean.index)
    regimes = [c for c in ["Bear", "Ranging", "Bull"] if c in pivot_mean.columns]
    n_etfs  = len(etfs)
    n_reg   = len(regimes)
    x       = np.arange(n_etfs)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, regime in enumerate(regimes):
        vals = pivot_mean[regime].values
        bars = ax.bar(
            x + i * width, vals, width,
            label=regime,
            color=regime_colors.get(regime, "#888888"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if v >= 0 else -0.06),
                    f"{v:+.1f}",
                    ha="center", va="bottom", fontsize=7, color="black",
                )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x + width * (n_reg - 1) / 2)
    ax.set_xticklabels(etfs, fontsize=10)
    ax.set_ylabel("Retorno medio semanal exceso vs SPY (%)", fontsize=11)
    ax.set_title(
        f"EDA — Rendimiento de ETFs sectoriales por Régimen de Mercado (IS {TRAIN_START[:4]}-{OOS_START[:4]})\n"
        "Régimen HMM (Bear / Ranging / Bull)  |  Target = ret_ETF(t+1) - ret_SPY(t+1) [semanal]",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.legend(title="Régimen", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Gráfico guardado: {save_path}")


# ── Inicialización del HMM (una sola vez, antes del loop) ────────────────────

def _build_hmm(data_dir: str) -> tuple:
    """
    Ajusta el HMM sobre datos IS completos y devuelve:
      model              GaussianHMM ajustado (parámetros fijos para todo el OOS)
      mapping            {idx_hmm -> etiqueta_económica}
      spy_df             DataFrame con OBS_COLS indexado por fecha (IS + OOS)
      is_regime_by_date  {fecha_IS: régimen 0/1/2}  (Viterbi IS-only)

    El ajuste usa todos los datos IS para estimar parámetros robustos.
    Durante el OOS los parámetros quedan fijos; solo la inferencia del
    estado cambia (ventana causal de HMM_CONTEXT_PERIODS semanas).
    """
    spy_df  = load_spy_features(data_dir)
    is_mask = spy_df.index < pd.Timestamp(OOS_START)
    X_is    = spy_df[OBS_COLS][is_mask].values

    print(f"[HMM] Observaciones: {OBS_COLS}")
    print(f"[HMM] Entrenando EM sobre {int(is_mask.sum())} semanas IS "
          f"(hasta {spy_df.index[is_mask].max().date()})...")
    model   = fit_hmm(X_is)
    mapping = label_mapping(model)

    if not model.monitor_.converged:
        print("  [WARN] EM no convergió — considera aumentar N_ITER en regime_model.py")

    is_labels         = decode_is_states(model, X_is, mapping)
    is_regime_by_date = dict(zip(spy_df.index[is_mask], is_labels))

    print(f"[HMM] Medias aprendidas (IS):")
    for hmm_s, econ_l in sorted(mapping.items(), key=lambda x: x[1]):
        mu = model.means_[hmm_s]
        print(f"  {REGIME_NAMES[econ_l]:10s}  ret_13w={mu[0]:>8.4f}  vol_13w={mu[1]:>8.4f}")

    return model, mapping, spy_df, is_regime_by_date


# ── Walk-Forward principal ────────────────────────────────────────────────────

def walk_forward_predict(
    panel: pd.DataFrame,
    model_name: str = "LightGBM",
    min_train_periods: int = MIN_TRAIN_PERIODS,
    data_dir: str = DATA_DIR,
) -> pd.DataFrame:
    """
    Loop walk-forward con ventanas independientes para HMM y ML.
    Frecuencia semanal: ~260 iteraciones OOS.

    HMM context window (HMM_CONTEXT_PERIODS semanas):
      En cada semana OOS t, el régimen se determina corriendo el filtro forward
      sobre las últimas HMM_CONTEXT_PERIODS observaciones del SPY hasta t,
      partiendo de un prior uniforme. Los parámetros (means_, covars_,
      transmat_) son los aprendidos en IS y permanecen fijos.

    ML expansiva desde TRAIN_START:
      El modelo ML se entrena en cada semana OOS t sobre todos los datos IS
      disponibles (desde TRAIN_START hasta t-1), garantizando suficiente
      historia para aprender relaciones cross-seccionales estables.

    Parámetros
    ----------
    panel              : DataFrame panel long con features y target
    model_name         : 'LightGBM' | 'RandomForest'
    min_train_periods  : semanas mínimas de historia antes de empezar OOS
    data_dir           : directorio de datos

    Devuelve
    --------
    DataFrame con predicciones OOS (guardado en data/predictions_{model_name}.csv)
    """
    if model_name not in MODELS_ALL:
        available = list(MODELS_ALL.keys())
        raise ValueError(
            f"Modelo '{model_name}' no disponible. "
            f"Opciones: {available}. "
            f"Para LightGBM, instala: pip install lightgbm"
        )

    # ── Inicializar HMM ───────────────────────────────────────────────────────
    hmm_model, mapping, spy_df, is_regime_by_date = _build_hmm(data_dir)

    # ── Preparar panel ────────────────────────────────────────────────────────
    panel = panel.copy()
    panel["market_regime"] = -1

    for date, regime in is_regime_by_date.items():
        panel.loc[panel["date"] == date, "market_regime"] = regime

    feat_cols  = get_feature_cols(panel)
    model_tmpl = MODELS_ALL[model_name]

    has_regime = "market_regime" in feat_cols
    print(f"\n[WF] Modelo ML     : {model_name}  |  ventana EXPANSIVA desde {TRAIN_START}")
    print(f"[WF] Features      : {len(feat_cols)} columnas "
          f"| market_regime={'SI' if has_regime else 'NO'}")
    print(f"[WF] HMM contexto  : ventana causal {HMM_CONTEXT_PERIODS} semanas (prior uniforme)")
    print(f"[WF] OOS           : {OOS_START} -> {OOS_END}\n")

    # ── Fechas OOS ────────────────────────────────────────────────────────────
    oos_dates = sorted(panel.loc[
        (panel["date"] >= OOS_START) & (panel["date"] <= OOS_END), "date"
    ].unique())

    if ML_TRAIN_EXCLUDE_PERIODS and oos_dates:
        t0 = oos_dates[0]
        base = (panel["date"] >= TRAIN_START) & (panel["date"] < t0)
        n_all = int(base.sum())
        n_kept = int((base & ml_train_date_kept(panel["date"])).sum())
        print(
            f"[WF] ML train excluye fechas {ML_TRAIN_EXCLUDE_PERIODS} "
            f"(HMM sin cambios) → {n_all - n_kept} filas panel omitidas "
            f"del train inicial ({n_kept} filas usadas vs {n_all} sin exclusión)\n"
        )

    all_preds      = []
    weeks_skipped  = 0

    for t in oos_dates:

        # ── 1. HMM: régimen causal para la semana t ───────────────────────────
        if spy_df.loc[spy_df.index <= t].shape[0] == 0:
            weeks_skipped += 1
            continue

        regime_t = get_regime_from_context_window(
            hmm_model, spy_df, t, HMM_CONTEXT_PERIODS, mapping
        )
        panel.loc[panel["date"] == t, "market_regime"] = regime_t

        # ── 2. ML: ventana expansiva desde TRAIN_START (sin periodos excluidos)
        train_mask = (
            (panel["date"] >= TRAIN_START) & (panel["date"] < t)
            & ml_train_date_kept(panel["date"])
        )
        train_data = panel[train_mask].dropna(subset=["target"])

        if len(train_data) < min_train_periods * 11:
            weeks_skipped += 1
            continue

        X_train = train_data[feat_cols].values
        y_train = train_data["target"].values

        # ── 3. Features de predicción para date=t ────────────────────────────
        pred_data = panel[panel["date"] == t].copy()
        if pred_data.empty:
            continue
        X_pred = pred_data[feat_cols].values

        # ── 4. Clon fresco del modelo + entrenamiento + predicción ────────────
        pipe = build_pipeline(model_tmpl)
        pipe.fit(X_train, y_train)
        pred_data["predicted_return"] = pipe.predict(X_pred)
        pred_data["rank"] = pred_data["predicted_return"].rank(ascending=False).astype(int)
        all_preds.append(pred_data[["date", "etf", "predicted_return", "rank", "target"]])

        # Imprimir solo en el primer viernes de enero para no saturar la salida
        if t.month == 1 and t.day <= 7:
            top_etfs = list(pred_data.nsmallest(TOP_N, "rank")["etf"])
            print(f"  {t.date()}  regime={REGIME_NAMES.get(regime_t, '?'):8s}"
                  f"  train={len(train_data):5d}  top3={top_etfs}")

    if weeks_skipped > 0:
        print(f"  [Nota] {weeks_skipped} semanas omitidas (datos insuficientes)")

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path    = f"{data_dir}/predictions_{model_name}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"[WF] Predicciones guardadas: {out_path}  ({len(predictions)} filas OOS)")

    return predictions


# ── Feature Importance ────────────────────────────────────────────────────────

def feature_importance_report(
    panel: pd.DataFrame,
    model_name: str = "LightGBM",
) -> pd.Series:
    """
    Entrena el modelo sobre IS completo y reporta importancia de features.

    Para LightGBM usa feature_importances_ gain-based (más robusto ante
    variables con muchas categorías que el impurity-based de RandomForest).
    Para RF y GB usa también feature_importances_ de sklearn.
    Solo para diagnóstico; no afecta las predicciones OOS.
    """
    if model_name not in MODELS_ALL:
        print(f"[FI] Modelo '{model_name}' no disponible, omitiendo.")
        return pd.Series(dtype=float)

    feat_cols = get_feature_cols(panel)
    train = panel[
        (panel["date"] >= TRAIN_START) & (panel["date"] < OOS_START)
    ].dropna(subset=["target"])
    train = train[ml_train_date_kept(train["date"])]

    pipe = build_pipeline(MODELS_ALL[model_name])
    pipe.fit(train[feat_cols].values, train["target"].values)

    importances = pd.Series(
        pipe.named_steps["model"].feature_importances_,
        index=feat_cols,
    ).sort_values(ascending=False)

    out_path = f"{DATA_DIR}/feature_importance_{model_name}.csv"
    importances.to_csv(out_path)

    print(f"\n[FI] Top 10 features ({model_name}) — IS {TRAIN_START[:4]}-{OOS_START[:4]}:")
    print(importances.head(10).to_string())
    print(f"[FI] Guardado: {out_path}")
    return importances


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  04_walk_forward_training.py")
    print("  LightGBM · RandomForest + EDA por régimen")
    print("=" * 65)

    # ── Cargar panel base (sin régimen; se asigna en el walk-forward) ─────────
    panel = load_panel(regime=False)
    print(f"\nPanel base cargado: {panel.shape} | "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    # ── EDA: rendimiento de ETFs por régimen de mercado ───────────────────────
    # Requiere features_panel_with_regime.csv (generado por 03_market_regime_detection.py)
    try:
        panel_with_regime = load_panel(regime=True)
        eda_etf_by_regime(panel_with_regime)
    except FileNotFoundError:
        print(
            "\n[EDA] features_panel_with_regime.csv no encontrado.\n"
            "      Ejecuta 03_market_regime_detection.py primero.\n"
            "      Continuando sin EDA..."
        )

    # ── Walk-Forward: LightGBM (principal) y RandomForest (baseline) ─────────
    models_to_run = list(MODELS_ALL.keys())   # ["RandomForest"] o ["RandomForest","LightGBM"]
    if not LGBM_AVAILABLE:
        print("\n[!] LightGBM no disponible. Instala: pip install lightgbm")

    for model_name in models_to_run:
        print(f"\n{'='*65}")
        print(f"  Walk-Forward: {model_name}")
        print(f"{'='*65}")
        preds = walk_forward_predict(panel, model_name=model_name, min_train_periods=MIN_TRAIN_PERIODS)
        _     = feature_importance_report(panel, model_name=model_name)

    print(f"\n{'='*65}")
    print("  [OK] Walk-forward completado.")
    print(f"  Modelos ejecutados: {', '.join(models_to_run)}")
    print("  Continua con 05_strategy_backtest.py y 06_signal_evaluation.py")
    print("=" * 65)
