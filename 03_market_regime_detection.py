"""
03_market_regime_detection.py
=============================
Visualizacion de regimenes de mercado y generacion de CSV de respaldo.

Guarda:
  data/market_regimes.csv                  (etiqueta por fecha)
  data/features_panel_with_regime.csv      (panel para auditoria)
  data/market_regimes_plot.png             (grafico)

Decodificacion en este script (solo visualizacion / diagnostico):
  IS  -> Viterbi sobre IS unicamente  (sin contaminacion OOS)
  OOS -> Viterbi sobre secuencia completa IS+OOS  (backward pass global)
         El modelo sigue entrenado SOLO en IS; el Viterbi completo da la
         visualizacion mas limpia e interpretable para el TFM.

Decodificacion en el walk-forward (04_walk_forward_training.py):
  OOS -> Forward filter causal, un paso por semana  [garantia de no lookahead]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, OOS_START
from regime_model import (
    load_spy_features, fit_hmm, label_mapping,
    decode_full_viterbi,
    hmm_diagnostics,
    OBS_COLS, N_STATES, REGIME_NAMES, REGIME_COLORS,
)


# ── Visualizacion ─────────────────────────────────────────────────────────────

def _plot_regimes(prices: pd.DataFrame, regime: pd.Series, save_path: str):
    """Precio del SPY con fondo coloreado por regimen y linea de volatilidad."""
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Panel superior: precio SPY
    ax = axes[0]
    aligned_spy = prices["SPY"].reindex(regime.index)
    ax.plot(regime.index, aligned_spy.values, color="black", lw=1.5, label="SPY", zorder=3)

    for date, reg in regime.items():
        ax.axvspan(
            date,
            date + pd.DateOffset(months=1),
            color=REGIME_COLORS[reg],
            alpha=0.28,
            zorder=1,
        )

    patches = [
        mpatches.Patch(color=REGIME_COLORS[i], alpha=0.7, label=REGIME_NAMES[i])
        for i in range(N_STATES)
    ]
    patches.append(plt.Line2D([0], [0], color="black", lw=1.5, label="SPY"))
    ax.legend(handles=patches, loc="upper left", fontsize=10)
    ax.set_title(
        "Regimenes de mercado — Gaussian HMM  (ret_13w + vol_13w)  |  "
        "Train IS-only · Visualizacion: Viterbi global",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Precio SPY (USD)")
    ax.grid(True, alpha=0.3)

    # Panel inferior: regimen numerico para ver transiciones
    ax2 = axes[1]
    ax2.step(regime.index, regime.values, where="post", color="#555555", lw=1)
    ax2.fill_between(
        regime.index, regime.values, step="post",
        color="#888888", alpha=0.25,
    )
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Bear", "Ranging", "Bull"], fontsize=9)
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Regimen")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Grafico guardado: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  REGIME DETECTION — Gaussian HMM  (Bear / Ranging / Bull)")
    print("=" * 60 + "\n")

    df      = load_spy_features(DATA_DIR)
    X       = df[OBS_COLS].values
    is_mask = df.index < pd.Timestamp(OOS_START)
    X_is    = X[is_mask]
    X_oos   = X[~is_mask]
    X_all   = X                        # IS + OOS para Viterbi global

    n_is, n_oos = int(is_mask.sum()), int((~is_mask).sum())
    print(f"[HMM] Observaciones: {OBS_COLS}")
    print(f"[HMM] Entrenando EM con {n_is} semanas IS (hasta {df.index[is_mask].max().date()})")
    print(f"      IS  -> Viterbi IS-only  ({n_is} semanas)")
    print(f"      OOS -> Viterbi global IS+OOS para visualizacion ({n_oos} semanas)")

    model   = fit_hmm(X_is)
    mapping = label_mapping(model)

    # ── Diagnostico del modelo (BIC, log-lik, convergencia) ──────────────────
    diag = hmm_diagnostics(model, X_is)
    print(f"\n[HMM] Diagnosticos del modelo (IS):")
    print(f"  Log-likelihood : {diag['log_likelihood']}")
    print(f"  BIC            : {diag['bic']}  (k={diag['n_params']} params, n={diag['n_obs']} obs)")
    print(f"  Convergencia   : {'SI' if diag['converged'] else 'NO (aumentar N_ITER)'}  "
          f"({diag['n_iter_done']} iteraciones EM)")

    if not model.monitor_.converged:
        print("  [WARN] EM no convergio — considera aumentar N_ITER en regime_model.py")

    # Decodificar con Viterbi global (IS+OOS) para visualizacion limpia
    all_labels = decode_full_viterbi(model, X_all, mapping)
    regime     = pd.Series(all_labels, index=df.index, name="market_regime")

    # Diagnostico de parametros aprendidos
    print("\n[HMM] Parametros aprendidos (IS):")
    print(f"  {'Regimen':10s}  {'ret_13w':>8s}  {'vol_13w':>8s}  {'std_r13w':>10s}  {'std_v13w':>10s}")
    for hmm_s, econ_l in sorted(mapping.items(), key=lambda x: x[1]):
        mu   = model.means_[hmm_s]
        cov  = model.covars_[hmm_s]
        s_r  = float(np.sqrt(cov[0, 0]))
        s_v  = float(np.sqrt(cov[1, 1]))
        print(f"  {REGIME_NAMES[econ_l]:10s}  {mu[0]:>8.4f}  {mu[1]:>8.4f}  {s_r:>10.4f}  {s_v:>10.4f}")

    # Matriz de transicion aprendida
    print("\n[HMM] Matriz de transicion aprendida:")
    print(f"  {'':12s}  {'->Bear':>8s}  {'->Rang':>8s}  {'->Bull':>8s}")
    for hmm_s, econ_l in sorted(mapping.items(), key=lambda x: x[1]):
        row = model.transmat_[hmm_s]
        # reordenar columnas segun mapping
        row_econ = np.zeros(N_STATES)
        for hmm_j, econ_j in mapping.items():
            row_econ[econ_j] = row[hmm_j]
        print(f"  {REGIME_NAMES[econ_l]:12s}  {row_econ[0]:>8.3f}  {row_econ[1]:>8.3f}  {row_econ[2]:>8.3f}")

    # Distribucion de regimenes
    counts = regime.value_counts().sort_index()
    print(f"\n[Distribucion] {len(regime)} semanas totales:")
    for rid, name in REGIME_NAMES.items():
        n   = int(counts.get(rid, 0))
        pct = 100 * n / len(regime)
        print(f"  {name:10s} ({rid}): {n:4d} semanas  ({pct:5.1f}%)")

    # Guardar market_regimes.csv
    regime_df = regime.reset_index()
    regime_df.columns = ["date", "market_regime"]
    regime_df.to_csv(f"{DATA_DIR}/market_regimes.csv", index=False)
    print(f"\n[Save] {DATA_DIR}/market_regimes.csv  ({len(regime_df)} filas)")

    # Guardar features_panel_with_regime.csv
    panel = pd.read_csv(f"{DATA_DIR}/features_panel.csv", parse_dates=["date"])
    if "market_regime" in panel.columns:
        panel.drop(columns=["market_regime"], inplace=True)
    panel = panel.merge(regime_df, on="date", how="left")
    panel["market_regime"] = panel["market_regime"].fillna(-1).astype(int)
    panel.to_csv(f"{DATA_DIR}/features_panel_with_regime.csv", index=False)
    print(f"[Save] {DATA_DIR}/features_panel_with_regime.csv  {panel.shape}")

    # Visualizacion
    prices = pd.read_csv(f"{DATA_DIR}/etf_prices.csv", index_col=0, parse_dates=True)
    prices.sort_index(inplace=True)
    _plot_regimes(prices, regime, f"{DATA_DIR}/market_regimes_plot.png")

    print("\n[OK] Deteccion de regimenes completada.")


if __name__ == "__main__":
    main()
