"""
04b_regime_walk_forward.py
===========================
# v2 final — Mayo 2026
Walk-Forward con un LGBM que recibe el regimen HMM como feature.

El regimen de mercado (Bear / Ranging / Bull) se incorpora al modelo como
cuatro variables adicionales en lugar de entrenar modelos separados por regimen:

  market_regime  — regimen mas probable (int: 0=Bear, 1=Ranging, 2=Bull;
                   -1 para semanas anteriores a la ventana HMM rodante)
  bear_prob      — P(Bear    | observaciones pasadas), forward filter causal
  ranging_prob   — P(Ranging | observaciones pasadas), forward filter causal
  bull_prob      — P(Bull    | observaciones pasadas), forward filter causal

Ventajas frente a modelos separados por regimen:
  + Usa todos los datos IS (no descarta 2/3 del panel por regimen).
  + Las probabilidades soft transmiten la incertidumbre de regimen al modelo.
  + El LGBM aprende interacciones cruzadas entre features y regimen.
  + Para semanas antiguas sin régimen calculado (fuera de la ventana HMM
    rodante), market_regime=-1 y probs=1/3 señalizan incertidumbre máxima.

Arquitectura por paso walk-forward (semana OOS t)
---------------------------------------------------
  1. Ajustar GaussianHMM en ventana rodante SPY [t - HMM_REGIME_LOOKBACK, t)
     (EM/Baum-Welch; solo datos < t)
  2. Forward filter causal -> P(regimen | obs_1..s) para cada semana s en ventana
  3. Anadir market_regime, bear_prob, ranging_prob, bull_prob a train_data:
       - Semanas dentro de la ventana: probabilidades del forward filter
       - Semanas fuera de la ventana: market_regime=-1, probs=1/3 (maxima incert.)
  4. Entrenar 1 LGBM sobre TODOS los datos de train (77 features = 73 base + 4 regimen)
  5. Avanzar forward filter un paso hasta t -> regimen_t actual
  6. Inyectar features de regimen en pred_data y predecir con el LGBM

Anti-leakage verificado
------------------------
  - Parametros HMM (EM)     : estimados solo con datos en [t - HMM_LOOKBACK, t)
  - Etiquetas regimen train  : forward filter causal paso a paso, sin backward
  - Regimen en t             : forward filter avanzado con observacion de t
  - ML training              : panel con date < t en todos los casos
  - Target                   : retorno t+1, shiftado en 02_feature_engineering.py

Salida
------
  data/predictions_RegimeLGBM.csv
    Columnas: date, etf, predicted_return, rank, target,
              regime, regime_name, bear_prob, ranging_prob, bull_prob

  Compatible con build_portfolio() de 05_strategy_backtest.py.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TRAIN_START, OOS_START, OOS_END,
    TOP_N, MIN_TRAIN_PERIODS, HMM_REGIME_LOOKBACK,
)
from utils import load_panel, get_feature_cols, ml_train_date_kept
from models import MODELS, build_pipeline
from regime_model import (
    load_spy_features, fit_hmm, label_mapping,
    forward_step, OBS_COLS, N_STATES, REGIME_NAMES,
)


# ── Columnas de regimen añadidas dinamicamente ────────────────────────────────
REGIME_FEAT_COLS = ["market_regime", "bear_prob", "ranging_prob", "bull_prob"]

# Valor asignado a semanas fuera de la ventana HMM rodante (sin regimen calculado)
_UNKNOWN_REGIME = -1
_UNIFORM_PROB   = 1.0 / N_STATES


# ── HMM: ajuste y etiquetado causal ──────────────────────────────────────────

def _fit_and_label_train(
    spy_df: pd.DataFrame,
    t: pd.Timestamp,
    hmm_lookback: int = HMM_REGIME_LOOKBACK,
) -> tuple:
    """
    Ajusta GaussianHMM en ventana rodante [t - hmm_lookback, t) y calcula
    probabilidades de regimen con forward filter causal para cada fecha
    de esa ventana (solo informacion disponible hasta cada punto s < t).

    Ventana HMM rodante vs ML expansiva (separación intencional):
      HMM  [t - hmm_lookback, t)  aprox. 5 años (260 semanas):
           Captura la dinámica de régimen RECIENTE; las relaciones Bear/Bull
           de 2008 no contaminan la inferencia de 2024.
      ML   [TRAIN_START, t)       aprox. 12-16 años (expansiva):
           Relaciones cross-seccionales features<->retornos son más estables;
           más datos mejoran la generalización del modelo.

    Devuelve
    --------
    model        : GaussianHMM ajustado  (None si datos insuficientes)
    mapping      : {indice_hmm -> etiqueta_economica}
    probs_train  : {fecha -> array(3,)} con P(regimen=k | obs_pasadas)
    alpha_end    : vector alpha al final de la ventana HMM (array(N_STATES,))
    """
    candidates = spy_df.loc[spy_df.index < t]
    train_spy  = candidates.tail(hmm_lookback).copy()

    if len(train_spy) < 104:   # minimo ~2 anyos para EM estable
        return None, None, {}, None

    X_train = train_spy[OBS_COLS].values
    model   = fit_hmm(X_train)
    mapping = label_mapping(model)

    alpha       = model.startprob_.copy()
    probs_train = {}

    for date in train_spy.index:
        obs   = train_spy.loc[date, OBS_COLS].values
        alpha = forward_step(model, alpha, obs)

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
    Avanza el forward filter un paso hasta t y devuelve el regimen actual.

    Devuelve
    --------
    regime_t : int  — regimen mas probable en t  (0=Bear, 1=Ranging, 2=Bull)
    probs_t  : array(3,) — probabilidades de cada regimen en t
    """
    if t in spy_df.index:
        obs = spy_df.loc[t, OBS_COLS].values
    else:
        available = spy_df.loc[spy_df.index <= t]
        obs = available.iloc[-1][OBS_COLS].values if not available.empty \
              else np.zeros(len(OBS_COLS))

    alpha_t = forward_step(model, alpha_end, obs)

    econ_probs = np.zeros(N_STATES)
    for hmm_s, econ_l in mapping.items():
        econ_probs[econ_l] = alpha_t[hmm_s]

    return int(np.argmax(econ_probs)), econ_probs


# ── Walk-Forward principal ────────────────────────────────────────────────────

def walk_forward_regime_model(
    panel: pd.DataFrame,
    min_train_periods: int = MIN_TRAIN_PERIODS,
    data_dir: str          = DATA_DIR,
) -> pd.DataFrame:
    """
    Loop walk-forward con un LGBM que incluye el regimen HMM como feature.
    Frecuencia semanal; OOS definido por OOS_START/OOS_END en config.py.

    Por cada semana OOS t:
      1. Ajusta HMM en ventana rodante [t - HMM_REGIME_LOOKBACK, t)
      2. Forward filter causal -> probabilidades de regimen para cada semana train
      3. Anade market_regime, bear_prob, ranging_prob, bull_prob a train_data
         (semanas fuera de la ventana: market_regime=-1, probs=1/3)
      4. Entrena 1 LGBM con 73 features base + 4 features de regimen = 77 total
      5. Avanza filter hasta t -> regimen_t + probs_t
      6. Inyecta features de regimen en pred_data y predice

    Devuelve
    --------
    DataFrame con predicciones OOS (guardado en predictions_RegimeLGBM.csv)
    """
    spy_df    = load_spy_features(data_dir)
    feat_cols = get_feature_cols(panel)
    feat_cols = [c for c in feat_cols if c not in REGIME_FEAT_COLS]
    all_feat_cols = feat_cols + REGIME_FEAT_COLS

    lgbm_tmpl = MODELS["LightGBM"]
    n_etfs    = panel["etf"].nunique()

    oos_dates = sorted(panel.loc[
        (panel["date"] >= OOS_START) & (panel["date"] <= OOS_END), "date"
    ].unique())

    print(f"\n{'='*65}")
    print(f"  Walk-Forward — LGBM con Regimen como Feature")
    print(f"{'='*65}")
    print(f"  Features base    : {len(feat_cols)} columnas")
    print(f"  Features regimen : {REGIME_FEAT_COLS}")
    print(f"  Total features   : {len(all_feat_cols)} columnas")
    print(f"  ETFs             : {n_etfs}")
    print(f"  OOS              : {OOS_START} -> {OOS_END}  ({len(oos_dates)} semanas)")
    print(f"  Ventana HMM      : rodante {HMM_REGIME_LOOKBACK} semanas "
          f"(aprox. {HMM_REGIME_LOOKBACK // 52:.1f} años)  [re-ajuste EM por semana OOS]")
    print(f"  Ventana ML       : expansiva desde {TRAIN_START}  [todos los datos IS]")
    print()

    all_preds     = []
    weeks_skipped = 0

    for i, t in enumerate(oos_dates):

        # ── 1. HMM: ajuste + etiquetado causal ───────────────────────────────
        model, mapping, probs_train, alpha_end = _fit_and_label_train(spy_df, t)

        if model is None:
            weeks_skipped += 1
            continue

        # ── 2. Panel de entrenamiento ─────────────────────────────────────────
        train_mask = (
            (panel["date"] >= TRAIN_START) & (panel["date"] < t)
            & ml_train_date_kept(panel["date"])
        )
        train_data = panel[train_mask].dropna(subset=["target"]).copy()

        if len(train_data) < min_train_periods * n_etfs:
            weeks_skipped += 1
            continue

        # ── 3. Anadir features de regimen a train_data ────────────────────────
        # Semanas en ventana HMM: probabilidades del forward filter.
        # Semanas fuera de ventana (datos IS mas antiguos): market_regime=-1,
        # probs=1/3 para indicar incertidumbre maxima de regimen.
        train_data["market_regime"] = train_data["date"].map(
            lambda d: int(np.argmax(probs_train[d])) if d in probs_train
                      else _UNKNOWN_REGIME
        ).astype(int)
        train_data["bear_prob"] = train_data["date"].map(
            lambda d: float(probs_train[d][0]) if d in probs_train else _UNIFORM_PROB
        )
        train_data["ranging_prob"] = train_data["date"].map(
            lambda d: float(probs_train[d][1]) if d in probs_train else _UNIFORM_PROB
        )
        train_data["bull_prob"] = train_data["date"].map(
            lambda d: float(probs_train[d][2]) if d in probs_train else _UNIFORM_PROB
        )

        # ── 4. Entrenar 1 LGBM con todas las features + regimen ───────────────
        pipe = build_pipeline(lgbm_tmpl)
        pipe.fit(train_data[all_feat_cols].values, train_data["target"].values)

        # ── 5. Regimen actual en t ────────────────────────────────────────────
        regime_t, probs_t = _regime_at_t(model, mapping, alpha_end, spy_df, t)

        # ── 6. Prediccion con features de regimen_t inyectadas ────────────────
        pred_data = panel[panel["date"] == t].copy()
        if pred_data.empty:
            continue

        pred_data["market_regime"] = regime_t
        pred_data["bear_prob"]    = round(float(probs_t[0]), 4)
        pred_data["ranging_prob"] = round(float(probs_t[1]), 4)
        pred_data["bull_prob"]    = round(float(probs_t[2]), 4)

        pred_data["predicted_return"] = pipe.predict(pred_data[all_feat_cols].values)
        pred_data["rank"] = pred_data["predicted_return"].rank(ascending=False).astype(int)
        pred_data["regime"]      = regime_t
        pred_data["regime_name"] = REGIME_NAMES[regime_t]

        all_preds.append(pred_data[[
            "date", "etf", "predicted_return", "rank", "target",
            "regime", "regime_name", "bear_prob", "ranging_prob", "bull_prob",
        ]])

        # ── Log anual (primer viernes de enero) ───────────────────────────────
        if (t.month == 1 and t.day <= 7) or i == 0:
            top_etfs = list(pred_data.nsmallest(TOP_N, "rank")["etf"])
            print(
                f"  {t.date()}  "
                f"regimen={REGIME_NAMES[regime_t]:8s}  "
                f"train={len(train_data):5d} filas  "
                f"top3={top_etfs}"
            )

    print(f"\n[WF-Regime] Semanas predichas : {len(oos_dates) - weeks_skipped}")
    print(f"[WF-Regime] Semanas omitidas  : {weeks_skipped}")

    if not all_preds:
        print("[!] No se generaron predicciones. Verifica los datos.")
        return pd.DataFrame()

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path    = f"{data_dir}/predictions_RegimeLGBM.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\n[WF-Regime] Predicciones guardadas: {out_path}  ({len(predictions)} filas OOS)")

    return predictions


# ── Importancia de features (diagnostico IS) ──────────────────────────────────

def regime_feature_importance(
    panel: pd.DataFrame,
    data_dir: str = DATA_DIR,
) -> None:
    """
    Entrena el LGBM IS completo con features de regimen y reporta importancia.
    Usa regimenes de features_panel_with_regime.csv (generado por 03_market_regime_detection).
    Solo diagnostico; no afecta predicciones OOS.
    """
    from config import TRAIN_START, TRAIN_END

    try:
        panel_regime = load_panel(regime=True, data_dir=data_dir)
    except FileNotFoundError:
        print("  [!] features_panel_with_regime.csv no encontrado. Omitiendo diagnostico IS.")
        return

    feat_cols = get_feature_cols(panel_regime)
    feat_cols = [c for c in feat_cols if c not in REGIME_FEAT_COLS]

    # Generar columnas de probabilidad como proxy (one-hot) cuando no estan disponibles
    if "bear_prob" not in panel_regime.columns:
        panel_regime["bear_prob"]    = (panel_regime["market_regime"] == 0).astype(float)
        panel_regime["ranging_prob"] = (panel_regime["market_regime"] == 1).astype(float)
        panel_regime["bull_prob"]    = (panel_regime["market_regime"] == 2).astype(float)

    all_feat = feat_cols + [c for c in REGIME_FEAT_COLS if c in panel_regime.columns]

    train = panel_regime[
        (panel_regime["date"] >= TRAIN_START) & (panel_regime["date"] <= TRAIN_END)
    ].dropna(subset=["target"])
    train = train[ml_train_date_kept(train["date"])]

    pipe = build_pipeline(MODELS["LightGBM"])
    pipe.fit(train[all_feat].values, train["target"].values)

    importances = pd.Series(
        pipe.named_steps["model"].feature_importances_,
        index=all_feat
    ).sort_values(ascending=False)

    out = f"{data_dir}/feature_importance_RegimeLGBM.csv"
    importances.to_csv(out)
    print(f"\n[FI] Top 10 features (RegimeLGBM) — IS {TRAIN_START}-{TRAIN_END}:")
    print(importances.head(10).to_string())
    print(f"  -> Guardado: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel = load_panel(regime=False)
    print(f"Panel cargado: {panel.shape}  |  "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    preds = walk_forward_regime_model(panel)

    if not preds.empty:
        print(f"\n[OK] Walk-Forward RegimeLGBM completado.")
        print(f"     Predicciones OOS : {len(preds)} filas")

        regime_dist = (
            preds.drop_duplicates("date")
            .groupby("regime_name")["date"].count()
            .rename("semanas")
        )
        print(f"\n     Distribucion de regimenes detectados (semanas unicas):")
        for rname, cnt in regime_dist.items():
            pct = 100 * cnt / regime_dist.sum()
            print(f"       {rname:10s}: {cnt:3d} semanas  ({pct:5.1f}%)")

        print(f"\n     Importancia de features (diagnostico IS):")
        regime_feature_importance(panel)
