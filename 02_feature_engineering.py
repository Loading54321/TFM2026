"""
02_feature_engineering.py
==========================
# v2 final — Abril 2026
Construye el dataset final en formato LONG (panel):
  índice  = (date, etf)
  columnas = features + target

Frecuencia: semanal (último viernes de cada semana, W-FRI + ffill universal).

TARGET — exceso de retorno vs SPY (t+1 semana)
────────────────────────────────────────────────
  target = return_ETF(t+1) - return_SPY(t+1)

  Por qué exceso vs SPY y no retorno absoluto:
    La estrategia es de ROTACIÓN SECTORIAL: el objetivo es predecir qué
    sectores lo harán MEJOR QUE EL MERCADO, no la dirección del mercado.
    Con retorno absoluto el modelo aprende principalmente la dirección del
    SPY (que es igual para todos los ETFs en una semana dada) y no la
    diferencia entre sectores.
    Con exceso vs SPY, el target varía entre ETFs en cada semana y el modelo
    puede aprender la rotación sectorial real.

REGLA ANTI-LEAKAGE
───────────────────
  features  = información disponible al cierre de la semana t (viernes)
  target    = retorno_etf(t+1) - retorno_SPY(t+1)  [realizado en t+1]
  El shift se aplica POR ETF para evitar mezclar filas entre activos.

FEATURES — tres bloques
───────────────────────
  1. ETF-específicas (varían por ETF y por fecha):
       momentum_1w, momentum_3w, momentum_4w, momentum_7w, momentum_8w, momentum_13w, momentum_26w, momentum_52w,
       momentum_52_4 (retorno acumulado 52w excluyendo últimas 4w),
       vol_3w, vol_7w, vol_13w, vol_26w, vol_52w (volatilidad rolling anualizada)
       excess_ret_1w  = momentum_1w  - spy_ret_1w   (rendimiento relativo 1w)
       excess_ret_13w = momentum_13w - spy_ret_13w  (rendimiento relativo 13w ≈ trim.)
       excess_ret_52w = momentum_52w - spy_ret_52w  (rendimiento relativo 52w)

  2. Cross-seccionales por fecha (rank dentro del corte transversal):
       momentum_1w_rank, momentum_3w_rank, momentum_4w_rank, momentum_7w_rank, momentum_8w_rank,
       momentum_13w_rank, momentum_26w_rank, momentum_52w_rank,
       vol_3w_rank, vol_7w_rank, vol_13w_rank, vol_26w_rank, vol_52w_rank
       [0=peor, 1=mejor dentro del grupo de ETFs esa semana]

  3. Macro / mercado (iguales para todos los ETFs esa semana):
       SPY: spy_ret_1w, spy_ret_4w, spy_ret_52w
       FRED: CPI_YoY, IndProd_YoY, Unemployment, Unemp_Chg,
             FedFunds, FedFunds_Chg, YieldSpread,
             VIX, VIX_Chg, Gold_ret_1w, Gold_ret_4w,
             T3M, T10, Term_Spread_10_3m, T10_Chg,
             HY_OAS, HY_OAS_Chg,
             RepoRate, RepoRate_Chg (SOFR, desde 2018),
             JGB10Y, JGB10Y_Chg, US_JP_Spread (bono Japón 10Y + spread vs T10),
             IG_OAS, IG_OAS_Chg (ICE BofA IG credit spread),
             ISM, ISM_Chg (Chicago Fed NAI, proxy PMI manufacturero),
             Oil_ret_1w, Oil_ret_4w (WTI vía FRED DCOILWTICO),
             recession (NBER binario, nivel),
             financial_conditions (NFCI semanal, nivel),
             leverage (NFCI subíndice apalancamiento semanal, nivel),
             sentiment (UMich Consumer Sentiment mensual, nivel),
             recession_diff, yield_curve_diff, financial_conditions_diff,
             leverage_diff, sentiment_diff, empleo_diff, inflacion_diff
       Fama-French 5: Mkt-RF, SMB, HML, RMW, CMA, RF
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, SECTOR_ETFS, TRAIN_START,
    MIN_DATA_FREQ_DAYS, MAX_DATA_FREQ_DAYS, MAX_NA_FORWARD_LOOKING,
)
from utils import load_data


# ── Validación temporal ────────────────────────────────────────────────────────

def _validate_temporal_alignment(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    ff5: pd.DataFrame,
    sector_etfs: list,
) -> None:
    """
    Comprueba que todos los datos están sincronizados semanalmente
    y que los ETFs requeridos están presentes en el CSV de precios.
    """
    price_freq = prices.index.to_series().diff().dt.days.median()
    assert MIN_DATA_FREQ_DAYS <= price_freq <= MAX_DATA_FREQ_DAYS, \
        f"Precios: frecuencia sospechosa ({price_freq} días)"

    macro_freq = macro.index.to_series().diff().dt.days.median()
    assert MIN_DATA_FREQ_DAYS <= macro_freq <= MAX_DATA_FREQ_DAYS, \
        f"Macro: frecuencia sospechosa ({macro_freq} días)"

    ff5_freq = ff5.index.to_series().diff().dt.days.median()
    assert MIN_DATA_FREQ_DAYS <= ff5_freq <= MAX_DATA_FREQ_DAYS, \
        f"FF5: frecuencia sospechosa ({ff5_freq} días)"

    common_start = max(prices.index[0], macro.index[0], ff5.index[0])
    common_end   = min(prices.index[-1], macro.index[-1], ff5.index[-1])
    print(f"  [OK] Periodo comun : {common_start.date()} → {common_end.date()}")
    print(f"  [OK] Precios       : {prices.shape[0]} semanas, {prices.shape[1]} series")
    print(f"  [OK] Macro (FRED)  : {macro.shape[0]} semanas, {macro.shape[1]} series")
    print(f"  [OK] FF5           : {ff5.shape[0]} semanas, {ff5.shape[1]} series")

    # ETFs obligatorios
    missing = [e for e in sector_etfs if e not in prices.columns]
    if missing:
        raise ValueError(f"ETFs faltantes en etf_prices.csv: {missing}. "
                         f"Ejecuta 01_data_download.py para descargarlos.")

    # Comprobar NaN solo en el periodo de entrenamiento (TRAIN_START en adelante).
    prices_train = prices[prices.index >= pd.Timestamp(TRAIN_START)]
    for etf in sector_etfs:
        pct_null = prices_train[etf].isna().mean()
        if pct_null > 0:
            print(f"  [WARN] {etf}: {pct_null:.1%} NaN en periodo de entrenamiento")
        assert pct_null < 0.05, (
            f"{etf}: demasiados NaN en periodo de entrenamiento ({pct_null:.1%}). "
            f"Verifica que {etf} existe desde {TRAIN_START}."
        )

    # Macro: advertir (no fallar) si alguna columna tiene muchos NaN
    macro_nulls = macro.isna().mean()
    problematic = macro_nulls[macro_nulls >= MAX_NA_FORWARD_LOOKING]
    if not problematic.empty:
        print(f"  [WARN] Columnas macro con >{MAX_NA_FORWARD_LOOKING:.0%} NaN:")
        for col, pct in problematic.items():
            print(f"    {col}: {pct:.1%}")


# ── Features ETF-específicas ───────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    """
    RSI de Wilder usando suavizado exponencial (EWM con alpha = 1/period).

    Devuelve valores en [0, 100]:
      < 30  → territorio de sobreventa  (posible recuperación)
      > 70  → territorio de sobrecompra (posible corrección)

    Causal: solo usa historia hasta el punto t (sin lookahead).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_etf_features(prices: pd.DataFrame, sector_etfs: list) -> pd.DataFrame:
    """
    Genera features de momentum, volatilidad y RSI por ETF a frecuencia semanal:
      momentum_1w   retorno 1 semana
      momentum_3w   retorno acumulado 3 semanas
      momentum_4w   retorno acumulado 4 semanas (≈ 1 mes)
      momentum_7w   retorno acumulado 7 semanas
      momentum_8w   retorno acumulado 8 semanas (≈ 2 meses)
      momentum_13w  retorno acumulado 13 semanas (≈ 1 trimestre)
      momentum_26w  retorno acumulado 26 semanas (≈ 6 meses)
      momentum_52w  retorno acumulado 52 semanas (≈ 1 año)
      momentum_52_4 momentum_52w − momentum_4w  (momentum anual excluyendo el último mes)
      vol_3w        volatilidad rolling anualizada, ventana 3 semanas
      vol_7w        volatilidad rolling anualizada, ventana 7 semanas
      vol_13w       volatilidad rolling anualizada, ventana 13 semanas
      vol_26w       volatilidad rolling anualizada, ventana 26 semanas
      vol_52w       volatilidad rolling anualizada, ventana 52 semanas
      rsi_9w        RSI Wilder 9 semanas (≈ 2 meses, táctico)
      rsi_14w       RSI Wilder 14 semanas (≈ 3.5 meses, estándar)
      rsi_26w       RSI Wilder 26 semanas (≈ 6 meses, medio plazo)

    Todas las features son causales: usan datos disponibles al cierre de la semana t.

    Beta (beta_52w, beta_26w):
      Covarianza rolling del retorno del ETF con el SPY dividida por la varianza
      rolling del SPY. Causal: en t solo usa retornos [t-window+1, t].
      Un beta > 1 indica que el ETF amplifica los movimientos del mercado;
      < 1 indica que los amortigua. Utiles para que el modelo condicione la
      prediccion al grado de sensibilidad sistematica del sector.
    """
    returns = prices[sector_etfs].pct_change()
    spy_r   = prices["SPY"].pct_change()
    records = []

    for etf in sector_etfs:
        r           = returns[etf].dropna()
        p           = prices[etf].reindex(r.index)
        spy_aligned = spy_r.reindex(r.index)
        spy_var_52  = spy_aligned.rolling(52).var().replace(0, np.nan)
        spy_var_26  = spy_aligned.rolling(26).var().replace(0, np.nan)

        d = pd.DataFrame(index=r.index)
        d["etf"]           = etf
        d["return"]        = r
        d["momentum_1w"]   = r
        d["momentum_3w"]   = r.rolling(3).sum()
        d["momentum_4w"]   = r.rolling(4).sum()
        d["momentum_7w"]   = r.rolling(7).sum()
        d["momentum_8w"]   = r.rolling(8).sum()
        d["momentum_13w"]  = r.rolling(13).sum()
        d["momentum_26w"]  = r.rolling(26).sum()
        d["momentum_52w"]  = r.rolling(52).sum()
        d["momentum_52_4"] = r.rolling(52).sum() - r.rolling(4).sum()
        d["vol_3w"]        = r.rolling(3).std()  * np.sqrt(52)
        d["vol_7w"]        = r.rolling(7).std()  * np.sqrt(52)
        d["vol_13w"]       = r.rolling(13).std() * np.sqrt(52)
        d["vol_26w"]       = r.rolling(26).std() * np.sqrt(52)
        d["vol_52w"]       = r.rolling(52).std() * np.sqrt(52)
        d["rsi_9w"]        = _rsi(p, 9)
        d["rsi_14w"]       = _rsi(p, 14)
        d["rsi_26w"]       = _rsi(p, 26)
        d["beta_52w"]      = r.rolling(52).cov(spy_aligned) / spy_var_52
        d["beta_26w"]      = r.rolling(26).cov(spy_aligned) / spy_var_26
        records.append(d)

    panel = pd.concat(records)
    panel.index.name = "date"
    panel.reset_index(inplace=True)

    # ffill de gaps puntuales por ETF (festivos o datos ausentes esporádicos)
    feat_cols = ["momentum_1w", "momentum_3w", "momentum_4w", "momentum_7w", "momentum_8w", "momentum_13w",
                 "momentum_26w", "momentum_52w", "momentum_52_4",
                 "vol_3w", "vol_7w", "vol_13w", "vol_26w", "vol_52w",
                 "rsi_9w", "rsi_14w", "rsi_26w",
                 "beta_52w", "beta_26w"]
    panel.sort_values(["etf", "date"], inplace=True)
    panel[feat_cols] = panel.groupby("etf")[feat_cols].ffill()

    return panel


# ── Features cross-seccionales ────────────────────────────────────────────────

def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Añade el rank percentil de cada ETF dentro de su corte transversal (semana t).

    ret_Xw_rank = 0  →  peor retorno del grupo esa semana
    ret_Xw_rank = 1  →  mejor retorno del grupo

    rsi_Xw_rank: rank del RSI dentro del grupo esa semana.
      rank alto (≈1) → ETF con mayor RSI relativo (más sobrecomprado en el grupo)
      rank bajo (≈0) → ETF con menor RSI relativo (más sobrevendido en el grupo)
    """
    for col in ["momentum_1w", "momentum_3w", "momentum_4w", "momentum_7w", "momentum_8w", "momentum_13w",
                "momentum_26w", "momentum_52w",
                "vol_3w", "vol_7w", "vol_13w", "vol_26w", "vol_52w",
                "rsi_9w", "rsi_14w", "rsi_26w",
                "beta_52w", "beta_26w"]:
        if col in panel.columns:
            panel[f"{col}_rank"] = (
                panel.groupby("date")[col]
                .rank(pct=True)
            )
    return panel


# ── Exceso de retorno vs SPY ──────────────────────────────────────────────────

def add_excess_return_features(
    panel: pd.DataFrame,
    spy_ret_1w: pd.Series,
    spy_ret_13w: pd.Series,
    spy_ret_52w: pd.Series,
) -> pd.DataFrame:
    """
    Añade retorno del ETF MENOS retorno del SPY en el mismo período:
      excess_ret_1w  = momentum_1w  - SPY_ret_1w   al cierre de la semana t
      excess_ret_13w = momentum_13w - SPY_ret_13w  últimas 13 semanas hasta t
      excess_ret_52w = momentum_52w - SPY_ret_52w  últimas 52 semanas hasta t

    Miden si el sector ha estado superando o rezagándose del mercado.
    """
    def _series_to_df(s: pd.Series, val_col: str) -> pd.DataFrame:
        df = s.reset_index()
        df.columns = ["date", val_col]
        return df

    panel = panel.merge(_series_to_df(spy_ret_1w,  "_spy1w"),  on="date", how="left")
    panel = panel.merge(_series_to_df(spy_ret_13w, "_spy13w"), on="date", how="left")
    panel = panel.merge(_series_to_df(spy_ret_52w, "_spy52w"), on="date", how="left")

    panel["excess_ret_1w"]  = panel["momentum_1w"]  - panel["_spy1w"]
    panel["excess_ret_13w"] = panel["momentum_13w"] - panel["_spy13w"]
    panel["excess_ret_52w"] = panel["momentum_52w"] - panel["_spy52w"]

    panel.drop(columns=["_spy1w", "_spy13w", "_spy52w"], inplace=True)
    return panel


# ── Pipeline principal ─────────────────────────────────────────────────────────

def build_feature_matrix(sector_etfs: list = None) -> pd.DataFrame:
    """
    Construye el panel final con ESTRICTO ANTI-LEAKAGE a frecuencia semanal.

    Target: excess_return_ETF(t+1) - excess_return_SPY(t+1)
      = retorno relativo al mercado en la semana siguiente.

    Devuelve DataFrame en formato long (date × etf) listo para el modelo.
    """
    if sector_etfs is None:
        sector_etfs = SECTOR_ETFS    # 11 ETFs definidos en config.py

    prices, macro, ff5 = load_data()

    # Normalizar nombres de índice a 'date'
    for df in (prices, macro, ff5):
        df.index.name = "date"

    # ── 1. Validación ──────────────────────────────────────────────────────────
    print("[FE] Validando sincronizacion temporal...")
    _validate_temporal_alignment(prices, macro, ff5, sector_etfs)

    # ── 2. Series SPY para features y target ──────────────────────────────────
    spy_pct    = prices["SPY"].pct_change()
    spy_ret_1w  = spy_pct                           # retorno 1w SPY en t
    spy_ret_4w  = spy_pct.rolling(4).sum()          # retorno acum 4w SPY en t
    spy_ret_13w = spy_pct.rolling(13).sum()         # retorno acum 13w SPY en t
    spy_ret_52w = spy_pct.rolling(52).sum()         # retorno acum 52w SPY en t
    spy_ret_t1  = spy_pct.shift(-1)                 # retorno SPY en t+1 (para target)

    spy_ret_t1.index.name = "date"
    spy_ret_t1 = spy_ret_t1.rename("_spy_t1")

    # ── 3. Features ETF-específicas ───────────────────────────────────────────
    print("[FE] Calculando features de ETF (ventanas semanales)...")
    panel = compute_etf_features(prices, sector_etfs)

    # ── 4. Features cross-seccionales (rank por fecha) ────────────────────────
    print("[FE] Calculando features cross-seccionales (rank por fecha)...")
    panel = add_cross_sectional_features(panel)

    # ── 5. Exceso de retorno vs SPY (features) ────────────────────────────────
    print("[FE] Calculando excesos de retorno vs SPY...")
    panel = add_excess_return_features(panel, spy_ret_1w, spy_ret_13w, spy_ret_52w)

    # ── 6. SPY return (feature de contexto de mercado) ────────────────────────
    spy_feat = pd.DataFrame({
        "date"        : spy_ret_1w.index,
        "spy_ret_1w"  : spy_ret_1w.values,
        "spy_ret_4w"  : spy_ret_4w.values,
        "spy_ret_52w" : spy_ret_52w.values,
    })
    panel = panel.merge(spy_feat, on="date", how="left")

    # ── 7. Macro FRED ────────────────────────────────────────────────────────
    print("[FE] Mergeando macro FRED...")
    macro_reset = macro.reset_index()
    if macro_reset.columns[0] != "date":
        macro_reset.rename(columns={macro_reset.columns[0]: "date"}, inplace=True)
    panel = panel.merge(macro_reset, on="date", how="left")

    # ── 8. Factores Fama-French 5 ─────────────────────────────────────────────
    print("[FE] Mergeando factores Fama-French...")
    ff5_reset = ff5.reset_index()
    date_col  = "Date" if "Date" in ff5_reset.columns else ff5_reset.columns[0]
    ff5_reset.rename(columns={date_col: "date"}, inplace=True)
    panel = panel.merge(ff5_reset, on="date", how="left")

    # ── 9. Target: exceso de retorno ETF vs SPY en t+1 ────────────────────────
    print("[FE] Calculando target (exceso de retorno vs SPY en t+1)...")
    panel.sort_values(["etf", "date"], inplace=True)

    # Retorno ETF en t+1 (shift por ETF para anti-leakage)
    panel["_etf_ret_t1"] = panel.groupby("etf")["return"].shift(-1)

    # Retorno SPY en t+1 (mismo para todos los ETFs esa semana)
    spy_t1_df = spy_ret_t1.reset_index()
    spy_t1_df.columns = ["date", "_spy_t1"]
    panel = panel.merge(spy_t1_df, on="date", how="left")

    # Target = exceso de retorno del ETF sobre el SPY la semana siguiente
    panel["target"] = panel["_etf_ret_t1"] - panel["_spy_t1"]
    panel.drop(columns=["_etf_ret_t1", "_spy_t1"], inplace=True)

    # ── 10. Limpieza y orden de columnas ──────────────────────────────────────
    n_before = len(panel)
    panel.dropna(subset=["target"], inplace=True)
    n_after  = len(panel)
    print(f"[FE] Filas sin target eliminadas: {n_before - n_after} "
          f"(ultima semana de cada ETF)")

    id_cols      = ["date", "etf"]
    target_col   = ["target"]
    feature_cols = [c for c in panel.columns
                    if c not in id_cols + target_col + ["return"]]
    panel = panel[id_cols + feature_cols + target_col]

    # ── 11. Resumen ───────────────────────────────────────────────────────────
    print(f"\n[FE] Dataset final: {panel.shape}  |  "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"[FE] ETFs incluidos ({len(sector_etfs)}): {sector_etfs}")
    print(f"[FE] Target: exceso de retorno ETF vs SPY en t+1 (semanal)")

    _etf_base  = {"momentum_1w","momentum_3w","momentum_4w","momentum_7w","momentum_8w","momentum_13w","momentum_26w","momentum_52w",
                  "momentum_52_4",
                  "vol_3w","vol_7w","vol_13w","vol_26w","vol_52w",
                  "rsi_9w","rsi_14w","rsi_26w",
                  "beta_52w","beta_26w"}
    _etf_rank  = {f"{c}_rank" for c in ["momentum_1w","momentum_3w","momentum_4w","momentum_7w","momentum_8w",
                                         "momentum_13w","momentum_26w","momentum_52w",
                                         "vol_3w","vol_7w","vol_13w","vol_26w","vol_52w",
                                         "rsi_9w","rsi_14w","rsi_26w",
                                         "beta_52w","beta_26w"]}
    _etf_exc   = {"excess_ret_1w","excess_ret_13w","excess_ret_52w"}
    _spy_cols  = {"spy_ret_1w","spy_ret_4w","spy_ret_52w"}
    _ff5_cols  = {"Mkt-RF","SMB","HML","RMW","CMA","RF"}
    _macro_cols = {
        "CPI_YoY","IndProd_YoY","Unemployment","Unemp_Chg",
        "FedFunds","FedFunds_Chg","YieldSpread",
        "VIX","VIX_Chg","Gold_ret_1w","Gold_ret_4w",
        "T3M","T10","Term_Spread_10_3m","T10_Chg","HY_OAS","HY_OAS_Chg",
        "RepoRate","RepoRate_Chg","JGB10Y","JGB10Y_Chg","US_JP_Spread",
        "IG_OAS","IG_OAS_Chg",
        "ISM","ISM_Chg",
        "Oil_ret_1w","Oil_ret_4w",
        # Recesión, condiciones financieras, apalancamiento y sentimiento
        "recession","financial_conditions","leverage","sentiment",
        "recession_diff","yield_curve_diff","financial_conditions_diff",
        "leverage_diff","sentiment_diff","empleo_diff","inflacion_diff",
    }

    feat_set    = set(feature_cols)
    etf_base    = sorted(feat_set & _etf_base)
    etf_rank    = sorted(feat_set & _etf_rank)
    etf_exc     = sorted(feat_set & _etf_exc)
    spy_feats   = sorted(feat_set & _spy_cols)
    ff5_feats   = sorted(feat_set & _ff5_cols)
    macro_feats = sorted(feat_set & _macro_cols)
    other_feats = sorted(feat_set - _etf_base - _etf_rank - _etf_exc
                         - _spy_cols - _ff5_cols - _macro_cols)

    print(f"\n[FE] Features ({len(feature_cols)} total):")
    print(f"  ETF momentum/vol ({len(etf_base)})  : {etf_base}")
    print(f"  ETF rank CS      ({len(etf_rank)})  : {etf_rank}")
    print(f"  ETF exceso SPY   ({len(etf_exc)})   : {etf_exc}")
    print(f"  SPY contexto     ({len(spy_feats)}) : {spy_feats}")
    print(f"  Fama-French 5    ({len(ff5_feats)}) : {ff5_feats}")
    print(f"  Macro/Riesgo     ({len(macro_feats)}): {macro_feats}")
    if other_feats:
        print(f"  Otros            ({len(other_feats)}): {other_feats}")

    # NaN por feature
    feats_nan = [(f, panel[f].isna().mean()*100) for f in feature_cols
                 if panel[f].isna().mean() > 0]
    if feats_nan:
        print("\n[FE] Features con NaN (se manejan por ffill; NaN iniciales son normales):")
        for feat, pct in sorted(feats_nan, key=lambda x: -x[1]):
            print(f"  {feat:28s}  nulls={pct:5.1f}%")

    # Estadístico descriptivo del target
    t = panel["target"]
    print(f"\n[FE] Target (exceso vs SPY) — estadisticos:")
    print(f"  media={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")
    print(f"  % semanas con exceso positivo: {(t > 0).mean():.1%}")

    panel.to_csv(f"{DATA_DIR}/features_panel.csv", index=False)
    print(f"\n  → Guardado: {DATA_DIR}/features_panel.csv")
    return panel


if __name__ == "__main__":
    panel = build_feature_matrix()
