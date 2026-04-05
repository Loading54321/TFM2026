"""
04b_regime_walk_forward.py
===========================
Walk-Forward con 3 modelos RandomForest especializados por régimen HMM.

Cada RF aprende únicamente de los períodos históricos que coinciden con su
régimen objetivo (Bear / Ranging / Bull), capturando relaciones distintas
entre features y retornos futuros en cada fase del ciclo económico.

Arquitectura por paso walk-forward (mes OOS t)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Ajustar GaussianHMM en ventana SPY [TRAIN_START, t)  [EM/Baum-Welch]
  2. Forward filter causal → P(régimen | obs_1..s) para cada mes s < t
  3. Para cada régimen k ∈ {0=Bear, 1=Ranging, 2=Bull}:
       • Filtrar observaciones del panel de train donde argmax P == k
         (o donde P(k) >= REGIME_THRESHOLD si el parámetro es > 0)
       • Si |obs_k| >= MIN_REGIME_OBS → entrenar RF_k en esos datos
       • Si no → RF_k = None (se usará el fallback global)
  4. Entrenar RF_global sobre TODOS los datos de train (fallback)
  5. Avanzar forward filter un paso hasta t → régimen_t actual
  6. Predicción en t con RF_{régimen_t} si disponible, o RF_global

Anti-leakage verificado
━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Parámetros HMM (EM)     : estimados solo con datos en [TRAIN_START, t)
  ✓ Etiquetas régimen train  : forward filter causal paso a paso, sin backward
  ✓ Régimen en t             : forward filter avanzado con observación de t
  ✓ ML training              : panel con date < t en todos los casos
  ✓ Target                   : retorno t+1, shiftado en 02_feature_engineering.py

Salida
━━━━━━
  data/predictions_RegimeRF.csv
    Columnas: date, etf, predicted_return, rank, target,
              regime, regime_name, model_used,
              bear_prob, ranging_prob, bull_prob

  Compatible con build_portfolio() de 05_strategy_backtest.py.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TRAIN_START, OOS_START, OOS_END,
    TOP_N, MIN_TRAIN_MONTHS,
)
from utils import load_panel, get_feature_cols
from models import MODELS, build_pipeline
from regime_model import (
    load_spy_features, fit_hmm, label_mapping,
    forward_step, OBS_COLS, N_STATES, REGIME_NAMES,
)


# ── Parámetros del módulo ─────────────────────────────────────────────────────

# Mínimo de filas del panel por régimen para entrenar RF_k.
# Con 11 ETFs y ~10 meses por régimen → ~110 filas.
MIN_REGIME_OBS = 88        # ≈ 8 meses × 11 ETFs

# Si > 0, solo incluir observaciones donde P(régimen=k) >= umbral.
# Si == 0, se usa argmax (el régimen más probable en cada mes).
REGIME_THRESHOLD = 0.0


# ── HMM: ajuste y etiquetado causal ──────────────────────────────────────────

def _fit_and_label_train(
    spy_df: pd.DataFrame,
    t: pd.Timestamp,
) -> tuple:
    """
    Ajusta GaussianHMM en la ventana de entrenamiento [TRAIN_START, t) y
    calcula probabilidades filtradas de régimen para cada fecha de train
    usando únicamente el forward filter causal (sin backward pass).

    Parámetros
    ----------
    spy_df : DataFrame indexado por fecha con columnas OBS_COLS (ret_3m, vol_3m)
    t      : primer mes OOS (fecha de predicción; excluida del entrenamiento)

    Devuelve
    --------
    model        : GaussianHMM ajustado  (None si datos insuficientes)
    mapping      : {índice_hmm -> etiqueta_económica}
    probs_train  : {fecha -> array(3,)} con P(régimen=k | obs_pasadas)
    alpha_end    : vector alpha al final de la ventana de train (array(N_STATES,))
                   Sirve como punto de partida para avanzar hasta t.
    """
    train_spy = spy_df.loc[
        (spy_df.index >= pd.Timestamp(TRAIN_START)) & (spy_df.index < t)
    ].copy()

    if len(train_spy) < 24:
        return None, None, {}, None

    X_train = train_spy[OBS_COLS].values

    # ── 1. Ajuste EM (Baum-Welch) solo con datos de train ────────────────────
    model   = fit_hmm(X_train)
    mapping = label_mapping(model)

    # ── 2. Forward filter causal sobre la ventana de train ───────────────────
    #    alpha_s = P(estados | obs_1..s); solo información pasada en cada paso.
    alpha       = model.startprob_.copy()
    probs_train = {}

    for date in train_spy.index:
        obs   = train_spy.loc[date, OBS_COLS].values
        alpha = forward_step(model, alpha, obs)

        # Reordenar de índices HMM (arbitrarios) a etiquetas económicas (0/1/2)
        econ_probs = np.zeros(N_STATES)
        for hmm_s, econ_l in mapping.items():
            econ_probs[econ_l] = alpha[hmm_s]
        probs_train[date] = econ_probs.copy()

    return model, mapping, probs_train, alpha.copy()


def _regime_at_t(
    model,
    mapping: dict,
    alpha_end: np.ndarray,
    spy_df: pd.DataFrame,
    t: pd.Timestamp,
) -> tuple:
    """
    Avanza el forward filter un paso hasta la fecha t y devuelve el
    régimen actual junto con el vector de probabilidades.

    alpha_end es el alpha al final de la ventana de train, garantizando
    que el estado en t se infiere de forma estrictamente causal.

    Devuelve
    --------
    regime_t : int  — régimen más probable en t  (0=Bear, 1=Ranging, 2=Bull)
    probs_t  : array(3,) — probabilidades de cada régimen en t
    """
    if t in spy_df.index:
        obs = spy_df.loc[t, OBS_COLS].values
    else:
        # Último punto disponible como proxy (raro con datos mensuales)
        available = spy_df.loc[spy_df.index <= t]
        obs = available.iloc[-1][OBS_COLS].values if not available.empty \
              else np.zeros(len(OBS_COLS))

    alpha_t = forward_step(model, alpha_end, obs)

    econ_probs = np.zeros(N_STATES)
    for hmm_s, econ_l in mapping.items():
        econ_probs[econ_l] = alpha_t[hmm_s]

    return int(np.argmax(econ_probs)), econ_probs


# ── Walk-Forward principal ────────────────────────────────────────────────────

def walk_forward_regime_models(
    panel: pd.DataFrame,
    min_train_months: int  = MIN_TRAIN_MONTHS,
    regime_threshold: float = REGIME_THRESHOLD,
    min_regime_obs: int    = MIN_REGIME_OBS,
    data_dir: str          = DATA_DIR,
) -> pd.DataFrame:
    """
    Loop walk-forward con 3 RF especializados por régimen + fallback global.

    Por cada mes OOS t:
      1. Ajusta HMM en [TRAIN_START, t)  →  parámetros aprendidos del pasado
      2. Etiqueta régimen de cada mes de train con forward filter causal
      3. Entrena RF_k sobre los meses donde argmax régimen == k (k=0,1,2)
      4. Entrena RF_global sobre todos los meses de train (fallback)
      5. Avanza filter hasta t → régimen_t
      6. Predice con RF_{régimen_t} o RF_global si RF_k no disponible

    Parámetros
    ----------
    panel            : DataFrame panel long (date × etf) con features y target
    min_train_months : meses mínimos de entrenamiento antes de empezar OOS
    regime_threshold : umbral de probabilidad (0 = argmax, >0 = threshold)
    min_regime_obs   : filas mínimas del panel por régimen para entrenar RF_k
    data_dir         : directorio de datos

    Devuelve
    --------
    DataFrame con predicciones OOS  (guardado también en predictions_RegimeRF.csv)
    """
    spy_df    = load_spy_features(data_dir)
    feat_cols = get_feature_cols(panel)
    rf_tmpl   = MODELS["RandomForest"]
    n_etfs    = panel["etf"].nunique()

    oos_dates = sorted(panel.loc[
        (panel["date"] >= OOS_START) & (panel["date"] <= OOS_END), "date"
    ].unique())

    print(f"\n{'='*65}")
    print(f"  Walk-Forward — 3 RF Especializados por Régimen HMM")
    print(f"{'='*65}")
    print(f"  Features         : {len(feat_cols)} columnas")
    print(f"  ETFs             : {n_etfs}")
    print(f"  OOS              : {OOS_START} → {OOS_END}  ({len(oos_dates)} meses)")
    print(f"  MIN_REGIME_OBS   : {min_regime_obs} filas  "
          f"(≈{min_regime_obs // n_etfs} meses × {n_etfs} ETFs)")
    print(f"  REGIME_THRESHOLD : "
          f"{'argmax' if regime_threshold == 0 else f'{regime_threshold:.2f} (umbral prob)'}")
    print()

    all_preds      = []
    months_skipped = 0
    usage_count    = {REGIME_NAMES[k]: 0 for k in range(N_STATES)}
    usage_count["global_fallback"] = 0

    for i, t in enumerate(oos_dates):

        # ── 1. HMM: ajuste + etiquetado causal de train ──────────────────────
        model, mapping, probs_train, alpha_end = _fit_and_label_train(spy_df, t)

        if model is None:
            months_skipped += 1
            continue

        # ── 2. Panel de entrenamiento para este paso ──────────────────────────
        train_mask = (panel["date"] >= TRAIN_START) & (panel["date"] < t)
        train_data = panel[train_mask].dropna(subset=["target"]).copy()

        if len(train_data) < min_train_months * n_etfs:
            months_skipped += 1
            continue

        # Mapear fecha → régimen argmax  (o probabilidad si threshold > 0)
        date_to_regime = {
            d: int(np.argmax(p)) for d, p in probs_train.items()
        }
        train_data["_regime"] = (
            train_data["date"].map(date_to_regime).fillna(-1).astype(int)
        )

        # ── 3. Entrenar RF_k por cada régimen ─────────────────────────────────
        regime_models = {}

        for k in range(N_STATES):
            if regime_threshold > 0:
                # Obs donde P(régimen=k) supera el umbral
                high_conf_dates = {
                    d for d, p in probs_train.items() if p[k] >= regime_threshold
                }
                regime_data = train_data[train_data["date"].isin(high_conf_dates)]
            else:
                # Obs donde el régimen k es el más probable (argmax)
                regime_data = train_data[train_data["_regime"] == k]

            n_obs = len(regime_data)

            if n_obs >= min_regime_obs:
                pipe_k = build_pipeline(rf_tmpl)
                pipe_k.fit(regime_data[feat_cols].values, regime_data["target"].values)
                regime_models[k] = pipe_k
            else:
                regime_models[k] = None  # insuficiente → usará fallback

        # ── 4. Fallback global (todos los datos de train) ─────────────────────
        pipe_global = build_pipeline(rf_tmpl)
        pipe_global.fit(train_data[feat_cols].values, train_data["target"].values)

        # ── 5. Régimen actual en t (forward filter un paso más) ───────────────
        regime_t, probs_t = _regime_at_t(model, mapping, alpha_end, spy_df, t)

        # ── 6. Seleccionar modelo activo ──────────────────────────────────────
        if regime_models.get(regime_t) is not None:
            active_pipe  = regime_models[regime_t]
            model_label  = REGIME_NAMES[regime_t]
        else:
            active_pipe  = pipe_global
            model_label  = "global_fallback"

        usage_count[model_label] = usage_count.get(model_label, 0) + 1

        # ── 7. Predicción sobre los ETFs del mes t ────────────────────────────
        pred_data = panel[panel["date"] == t].copy()
        if pred_data.empty:
            continue

        pred_data["predicted_return"] = active_pipe.predict(
            pred_data[feat_cols].values
        )
        pred_data["rank"] = (
            pred_data["predicted_return"].rank(ascending=False).astype(int)
        )
        pred_data["regime"]       = regime_t
        pred_data["regime_name"]  = REGIME_NAMES[regime_t]
        pred_data["model_used"]   = model_label
        pred_data["bear_prob"]    = round(float(probs_t[0]), 4)
        pred_data["ranging_prob"] = round(float(probs_t[1]), 4)
        pred_data["bull_prob"]    = round(float(probs_t[2]), 4)

        all_preds.append(pred_data[[
            "date", "etf", "predicted_return", "rank", "target",
            "regime", "regime_name", "model_used",
            "bear_prob", "ranging_prob", "bull_prob",
        ]])

        # ── Log anual ─────────────────────────────────────────────────────────
        if t.month == 1 or i == 0:
            top_etfs   = list(pred_data.nsmallest(TOP_N, "rank")["etf"])
            obs_counts = {
                REGIME_NAMES[k]: int((train_data["_regime"] == k).sum())
                for k in range(N_STATES)
            }
            print(
                f"  {t.date()}  "
                f"régimen={REGIME_NAMES[regime_t]:8s}  "
                f"modelo={model_label:16s}  "
                f"top3={top_etfs}  "
                f"[Bear={obs_counts['Bear']:3d}  "
                f"Rang={obs_counts['Ranging']:3d}  "
                f"Bull={obs_counts['Bull']:3d}]"
            )

    # ── Resumen de uso ────────────────────────────────────────────────────────
    total_predicted = sum(usage_count.values())
    print(f"\n[WF-Régimen] Meses predichos : {total_predicted}")
    print(f"[WF-Régimen] Meses omitidos  : {months_skipped}")
    print(f"[WF-Régimen] Uso de modelos  :")
    for label, cnt in sorted(usage_count.items()):
        if cnt > 0:
            pct = 100 * cnt / max(total_predicted, 1)
            print(f"    {label:20s}: {cnt:3d} meses  ({pct:5.1f}%)")

    if not all_preds:
        print("[!] No se generaron predicciones. Verifica los datos.")
        return pd.DataFrame()

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path    = f"{data_dir}/predictions_RegimeRF.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\n[WF-Régimen] Predicciones guardadas: {out_path}  "
          f"({len(predictions)} filas OOS)")

    return predictions


# ── Importancia de features por régimen (diagnóstico IS) ──────────────────────

def regime_feature_importance(
    panel: pd.DataFrame,
    data_dir: str    = DATA_DIR,
    min_regime_obs: int = MIN_REGIME_OBS,
) -> dict:
    """
    Entrena un RF IS completo por régimen y reporta importancia de features.

    Usa las etiquetas de régimen guardadas en features_panel_with_regime.csv
    (generadas por 03_market_regime_detection.py).  Solo para diagnóstico;
    no afecta las predicciones OOS.

    Devuelve
    --------
    dict {nombre_régimen -> pd.Series con importancias ordenadas}
    """
    from models import feature_importance_report as _fi

    # Cargar panel IS con etiquetas de régimen (ya calculadas por 03)
    panel_with_regime = load_panel(regime=True, data_dir=data_dir)
    feat_cols         = get_feature_cols(panel_with_regime)
    results           = {}

    for k, name in REGIME_NAMES.items():
        regime_panel = panel_with_regime[panel_with_regime["market_regime"] == k]
        n_obs = len(regime_panel[
            (regime_panel["date"] >= TRAIN_START) & (regime_panel["date"] < OOS_START)
        ])

        print(f"\n[FI-Régimen] {name} (k={k}) — {n_obs} obs IS")
        if n_obs < min_regime_obs:
            print(f"  → Datos insuficientes ({n_obs} < {min_regime_obs}), omitido")
            continue

        imp = _fi(regime_panel, "RandomForest", TRAIN_START, OOS_START, feat_cols)
        out = f"{data_dir}/feature_importance_RF_{name}.csv"
        imp.to_csv(out)
        print(f"  → Guardado: {out}")
        results[name] = imp

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cargar panel base (sin columna market_regime):
    #   El régimen se calcula internamente en walk_forward_regime_models()
    #   con el HMM re-ajustado en cada paso, garantizando causalidad estricta.
    panel = load_panel(regime=False)
    print(f"Panel cargado: {panel.shape}  |  "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    preds = walk_forward_regime_models(panel)

    if not preds.empty:
        print(f"\n[OK] Walk-Forward Régimen completado.")
        print(f"     Predicciones OOS : {len(preds)} filas")
        print(f"\n     Distribución de regímenes detectados (meses únicos):")
        regime_dist = (
            preds.drop_duplicates("date")
            .groupby("regime_name")["date"].count()
            .rename("meses")
        )
        for rname, cnt in regime_dist.items():
            pct = 100 * cnt / regime_dist.sum()
            print(f"       {rname:10s}: {cnt:3d} meses  ({pct:5.1f}%)")

        print(f"\n     Importancia de features por régimen (IS):")
        regime_feature_importance(panel)
