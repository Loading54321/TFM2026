"""
04_walk_forward_training.py
===========================
Walk-Forward con ventanas independientes para HMM y modelos ML.

Arquitectura de ventanas separadas
───────────────────────────────────
  HMM  (regimen) : ventana de contexto de HMM_CONTEXT_MONTHS = 54 meses
                   Determina el regimen actual usando solo los ultimos
                   4.5 anos de datos del SPY — adaptable sin cargar sesgo
                   de regimenes muy antiguos.

  ML   (prediccion): ventana EXPANSIVA desde TRAIN_START hasta t
                     Usa todos los datos IS disponibles; mas datos = mejor
                     generalizacion cross-seccional para RF y GBT.

Flujo por mes t en OOS:
  1. HMM context (54m):
       X_context = spy_features[ t-54m : t ]
       forward_filter(prior_uniforme) -> regime_t  [causal, sin lookahead]

  2. panel[date=t, market_regime] = regime_t

  3. ML expansivo:
       train = panel[ TRAIN_START : t )
       fit(clone(modelo), train) -> predictions[t]

  4. Rankear ETFs -> Top-3 largo / Bottom-3 corto

Anti-leakage verificado:
  - HMM parametros (EM) entrenados solo en IS (< OOS_START).
  - regime_t: observaciones unicamente <= t.
  - ML training: date < t en todos los casos.
  - Target = retorno t+1 (shiftado en 02).
  - Modelos ML clonados en cada iteracion.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TRAIN_START, OOS_START, OOS_END,
    TOP_N, MIN_TRAIN_MONTHS, HMM_CONTEXT_MONTHS,
    DEFAULT_MODEL,
)
from utils import load_panel, get_feature_cols
from models import MODELS, build_pipeline
from regime_model import (
    load_spy_features, fit_hmm, label_mapping,
    decode_is_states, get_regime_from_context_window,
    OBS_COLS, REGIME_NAMES,
)


# ── Inicializacion del HMM (una sola vez, antes del loop) ────────────────────

def _build_hmm(data_dir: str) -> tuple:
    """
    Ajusta el HMM sobre datos IS completos y devuelve:
      model              GaussianHMM ajustado (parametros fijos para todo el OOS)
      mapping            {idx_hmm -> etiqueta_economica}
      spy_df             DataFrame con OBS_COLS indexado por fecha (IS + OOS)
      is_regime_by_date  {fecha_IS: regimen 0/1/2}  (Viterbi IS-only)

    El ajuste del HMM usa todos los datos IS para estimar parametros robustos
    (medias, covarianzas, transiciones).  Durante el OOS, los parametros
    quedan fijos; solo la INFERENCIA del estado cambia (ventana de 24m).
    """
    spy_df  = load_spy_features(data_dir)
    is_mask = spy_df.index < pd.Timestamp(OOS_START)
    X_is    = spy_df[OBS_COLS][is_mask].values

    print(f"[HMM] Observaciones: {OBS_COLS}")
    print(f"[HMM] Entrenando EM sobre {int(is_mask.sum())} meses IS "
          f"(hasta {spy_df.index[is_mask].max().date()})...")
    model   = fit_hmm(X_is)
    mapping = label_mapping(model)

    if not model.monitor_.converged:
        print("  [WARN] EM no convergio — considera aumentar N_ITER en regime_model.py")

    # Regimenes IS via Viterbi (IS-only, sin contaminacion OOS)
    is_labels         = decode_is_states(model, X_is, mapping)
    is_regime_by_date = dict(zip(spy_df.index[is_mask], is_labels))

    col0, col1 = OBS_COLS
    print(f"[HMM] Medias aprendidas (IS):")
    print(f"  {'Regimen':10s}  {col0:>8s}  {col1:>8s}")
    for hmm_s, econ_l in sorted(mapping.items(), key=lambda x: x[1]):
        mu = model.means_[hmm_s]
        print(f"  {REGIME_NAMES[econ_l]:10s}  {mu[0]:>8.4f}  {mu[1]:>8.4f}")

    return model, mapping, spy_df, is_regime_by_date


# ── Walk-Forward principal ────────────────────────────────────────────────────

def walk_forward_predict(
    panel: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    min_train_months: int = MIN_TRAIN_MONTHS,
    data_dir: str = DATA_DIR,
) -> pd.DataFrame:
    """
    Loop walk-forward con ventanas independientes para HMM y ML.

    HMM context window (HMM_CONTEXT_MONTHS = 24):
      En cada mes OOS t, el regimen se determina corriendo el filtro forward
      sobre las ultimas 24 observaciones del SPY hasta t, partiendo de un
      prior uniforme.  Los parametros del modelo (means_, covars_, transmat_)
      son los aprendidos en IS y permanecen fijos.

    ML rolling window (TRAIN_WINDOW_MONTHS = 96):
      El modelo ML se entrena en cada mes OOS t sobre los datos de los
      ultimos 96 meses antes de t, garantizando suficiente historia para
      aprender relaciones cross-seccionales estables.
    """
    # ── Inicializar HMM ───────────────────────────────────────────────────────
    hmm_model, mapping, spy_df, is_regime_by_date = _build_hmm(data_dir)

    # ── Preparar panel ────────────────────────────────────────────────────────
    panel = panel.copy()
    panel["market_regime"] = -1

    # IS: regimenes via Viterbi (una sola pasada, ya calculados)
    for date, regime in is_regime_by_date.items():
        panel.loc[panel["date"] == date, "market_regime"] = regime

    feat_cols  = get_feature_cols(panel)
    model_tmpl = MODELS[model_name]

    has_regime = "market_regime" in feat_cols
    print(f"\n[WF] Features: {len(feat_cols)} columnas "
          f"| market_regime={'SI' if has_regime else 'NO'}")
    print(f"[WF] Modelo ML     : {model_name}  |  ventana EXPANSIVA desde {TRAIN_START}")
    print(f"[WF] HMM contexto  : ventana causal {HMM_CONTEXT_MONTHS} meses (prior uniforme)")
    print(f"[WF] OOS           : {OOS_START} -> {OOS_END}\n")

    # ── Fechas OOS ────────────────────────────────────────────────────────────
    oos_dates = sorted(panel.loc[
        (panel["date"] >= OOS_START) & (panel["date"] <= OOS_END), "date"
    ].unique())

    all_preds      = []
    months_skipped = 0

    for t in oos_dates:

        # ── 1. HMM: regimen via ventana de 24 meses (causal) ─────────────────
        if spy_df.loc[spy_df.index <= t].shape[0] == 0:
            months_skipped += 1
            continue

        regime_t = get_regime_from_context_window(
            hmm_model, spy_df, t, HMM_CONTEXT_MONTHS, mapping
        )

        # ── 2. Actualizar panel con regimen_t para date=t ─────────────────────
        panel.loc[panel["date"] == t, "market_regime"] = regime_t

        # ── 3. ML: ventana expansiva desde TRAIN_START ───────────────────────
        train_mask = (panel["date"] >= TRAIN_START) & (panel["date"] < t)
        train_data = panel[train_mask].dropna(subset=["target"])

        if len(train_data) < min_train_months * 9:
            months_skipped += 1
            continue

        X_train = train_data[feat_cols].values
        y_train = train_data["target"].values

        # ── 4. Features de prediccion para date=t ────────────────────────────
        pred_data = panel[panel["date"] == t].copy()
        if pred_data.empty:
            continue
        X_pred = pred_data[feat_cols].values

        # ── 5. Clon fresco del modelo + entrenamiento + prediccion ────────────
        pipe = build_pipeline(model_tmpl)
        pipe.fit(X_train, y_train)
        pred_data["predicted_return"] = pipe.predict(X_pred)
        pred_data["rank"] = pred_data["predicted_return"].rank(ascending=False).astype(int)
        all_preds.append(pred_data[["date", "etf", "predicted_return", "rank", "target"]])

        if t.month == 1:
            top_etfs = list(pred_data.nsmallest(TOP_N, "rank")["etf"])
            print(f"  {t.date()}  regime={REGIME_NAMES.get(regime_t, '?'):8s}"
                  f"  train={len(train_data):5d}  top3={top_etfs}")

    if months_skipped > 0:
        print(f"  [Nota] {months_skipped} meses omitidos (datos insuficientes)")

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path    = f"{data_dir}/predictions_{model_name}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"[WF] Predicciones guardadas: {out_path}  ({len(predictions)} filas OOS)")

    return predictions


# ── Feature Importance (IS completo) ──────────────────────────────────────────

def feature_importance_report(model_name: str = "RandomForest") -> pd.Series:
    """
    Entrena el modelo sobre datos IS completos (panel con regimen de 03)
    e imprime las features mas importantes.
    Solo para diagnostico; no afecta las predicciones OOS.
    """
    panel     = load_panel(regime=True)
    feat_cols = get_feature_cols(panel)
    from models import feature_importance_report as _fi
    return _fi(panel, model_name, TRAIN_START, OOS_START, feat_cols)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    panel = load_panel(regime=False)
    print(f"Panel base cargado: {panel.shape} | "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    for model_name in ["RandomForest", "GradientBoosting"]:
        preds = walk_forward_predict(panel, model_name=model_name)
        _     = feature_importance_report(model_name=model_name)

    print("\n[OK] Walk-forward completado para ambos modelos.")
