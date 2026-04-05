"""
02_feature_engineering.py
==========================
Construye el dataset final en formato LONG (panel):
  índice  = (date, etf)
  columnas = features + target

TARGET — exceso de retorno vs SPY (t+1)
─────────────────────────────────────────
  target = return_ETF(t+1) - return_SPY(t+1)

  Por qué exceso vs SPY y no retorno absoluto:
    La estrategia es de ROTACIÓN SECTORIAL: el objetivo es predecir qué
    sectores lo harán MEJOR QUE EL MERCADO, no la dirección del mercado.
    Con retorno absoluto el modelo aprende principalmente la dirección del
    SPY (que es igual para todos los ETFs en un mes dado) y no la diferencia
    entre sectores → IC negativo / señal invertida.
    Con exceso vs SPY, el target varía entre ETFs en cada mes y el modelo
    puede aprender la rotación sectorial real.

REGLA ANTI-LEAKAGE
───────────────────
  features  = información disponible al cierre del mes t
  target    = retorno_etf(t+1) - retorno_SPY(t+1)  [realizado en t+1]
  El shift se aplica POR ETF para evitar mezclar filas entre activos.

FEATURES — tres bloques
───────────────────────
  1. ETF-específicas (varían por ETF y por fecha):
       ret_1m, ret_3m, ret_6m, ret_12m, momentum_12_1, vol_6m
       excess_ret_1m  = ret_1m  - spy_ret_1m   (rendimiento relativo 1m)
       excess_ret_3m  = ret_3m  - spy_ret_3m   (rendimiento relativo 3m)
       excess_ret_12m = ret_12m - spy_ret_12m  (rendimiento relativo 12m)

  2. Cross-seccionales por fecha (rank dentro del corte transversal):
       ret_1m_rank, ret_3m_rank, ret_6m_rank, ret_12m_rank, vol_6m_rank
       [0=peor, 1=mejor dentro del grupo de ETFs ese mes]
       Estas features capturan la posición RELATIVA de cada ETF ese mes,
       que es exactamente lo que el modelo necesita para rankear sectores.

  3. Macro / mercado (iguales para todos los ETFs ese mes):
       SPY: spy_ret_1m, spy_ret_3m, spy_ret_12m
       FRED: CPI_YoY, IndProd_YoY, Unemployment, Unemp_Chg,
             FedFunds, FedFunds_Chg, YieldSpread, GDP_YoY
       Mercado/riesgo: VIX, VIX_Chg, Gold_ret_1m, Gold_ret_3m,
                       T3M, T10, Term_Spread_10_3m, T10_Chg,
                       HY_OAS, HY_OAS_Chg
       Fama-French 5: Mkt-RF, SMB, HML, RMW, CMA, RF
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, SECTOR_ETFS,
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
    Comprueba que todos los datos están sincronizados mensualmente
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
    print(f"  [OK] Precios       : {prices.shape[0]} filas, {prices.shape[1]} series")
    print(f"  [OK] Macro (FRED)  : {macro.shape[0]} filas, {macro.shape[1]} series")
    print(f"  [OK] FF5           : {ff5.shape[0]} filas, {ff5.shape[1]} series")

    # ETFs obligatorios
    missing = [e for e in sector_etfs if e not in prices.columns]
    if missing:
        raise ValueError(f"ETFs faltantes en etf_prices.csv: {missing}. "
                         f"Ejecuta 01_data_download.py para descargarlos.")
    for etf in sector_etfs:
        pct_null = prices[etf].isna().mean()
        assert pct_null < 0.05, f"{etf}: demasiados NaN ({pct_null:.1%})"

    # Macro: advertir (no fallar) si alguna columna tiene muchos NaN
    macro_nulls   = macro.isna().mean()
    problematic   = macro_nulls[macro_nulls >= MAX_NA_FORWARD_LOOKING]
    if not problematic.empty:
        print(f"  [WARN] Columnas macro con >{MAX_NA_FORWARD_LOOKING:.0%} NaN "
              f"(se imputarán por mediana en el pipeline ML):")
        for col, pct in problematic.items():
            print(f"    {col}: {pct:.1%}")


# ── Features ETF-específicas ───────────────────────────────────────────────────

def compute_etf_features(prices: pd.DataFrame, sector_etfs: list) -> pd.DataFrame:
    """
    Genera features de momentum y volatilidad por ETF:
      ret_1m, ret_3m, ret_6m, ret_12m,
      momentum_12_1 (retorno acumulado 12m excluyendo el último mes),
      vol_6m        (volatilidad rolling anualizada, ventana 6m)

    Todas las features son causales: usan datos disponibles al cierre de t.
    """
    returns = prices[sector_etfs].pct_change()
    records = []

    for etf in sector_etfs:
        r = returns[etf].dropna()
        d = pd.DataFrame(index=r.index)
        d["etf"]           = etf
        d["return"]        = r
        d["ret_1m"]        = r
        d["ret_3m"]        = r.rolling(3).sum()
        d["ret_6m"]        = r.rolling(6).sum()
        d["ret_12m"]       = r.rolling(12).sum()
        d["momentum_12_1"] = r.rolling(12).sum() - r
        d["vol_6m"]        = r.rolling(6).std() * np.sqrt(12)
        records.append(d)

    panel = pd.concat(records)
    panel.index.name = "date"
    panel.reset_index(inplace=True)
    return panel


# ── Features cross-seccionales ────────────────────────────────────────────────

def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Añade el rank percentil de cada ETF dentro de su corte transversal (mes t).

    Estas features capturan la POSICIÓN RELATIVA de cada sector ese mes:
      ret_1m_rank = 0  →  peor retorno del grupo en el último mes
      ret_1m_rank = 1  →  mejor retorno del grupo

    Son especialmente útiles para un modelo cross-seccional con solo 11 ETFs
    porque eliminan el componente común del mercado (que afecta a todos
    por igual) y dejan solo la variación entre sectores.
    """
    for col in ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_6m"]:
        if col in panel.columns:
            panel[f"{col}_rank"] = (
                panel.groupby("date")[col]
                .rank(pct=True)
            )
    return panel


# ── Exceso de retorno vs SPY ──────────────────────────────────────────────────

def add_excess_return_features(
    panel: pd.DataFrame,
    spy_ret_1m: pd.Series,
    spy_ret_3m: pd.Series,
    spy_ret_12m: pd.Series,
) -> pd.DataFrame:
    """
    Añade retorno del ETF MENOS retorno del SPY en el mismo período:
      excess_ret_1m  = ret_1m  - SPY_ret_1m   al cierre de t
      excess_ret_3m  = ret_3m  - SPY_ret_3m   últimos 3m hasta t
      excess_ret_12m = ret_12m - SPY_ret_12m  últimos 12m hasta t

    Miden si el sector ha estado superando o rezagándose del mercado,
    que es exactamente la señal de momentum relativo más útil para
    estrategias de rotación sectorial.
    """
    spy_1m  = spy_ret_1m.rename("_spy1m")
    spy_3m  = spy_ret_3m.rename("_spy3m")
    spy_12m = spy_ret_12m.rename("_spy12m")

    panel = panel.merge(spy_1m.reset_index(),  on="date", how="left")
    panel = panel.merge(spy_3m.reset_index(),  on="date", how="left")
    panel = panel.merge(spy_12m.reset_index(), on="date", how="left")

    panel["excess_ret_1m"]  = panel["ret_1m"]  - panel["_spy1m"]
    panel["excess_ret_3m"]  = panel["ret_3m"]  - panel["_spy3m"]
    panel["excess_ret_12m"] = panel["ret_12m"] - panel["_spy12m"]

    panel.drop(columns=["_spy1m", "_spy3m", "_spy12m"], inplace=True)
    return panel


# ── Pipeline principal ─────────────────────────────────────────────────────────

def build_feature_matrix(sector_etfs: list = None) -> pd.DataFrame:
    """
    Construye el panel final con ESTRICTO ANTI-LEAKAGE.

    Target: excess_return_ETF(t+1) - excess_return_SPY(t+1)
      = retorno relativo al mercado en el mes siguiente.

    Devuelve DataFrame en formato long (date × etf) listo para el modelo.
    """
    if sector_etfs is None:
        sector_etfs = SECTOR_ETFS    # 11 ETFs definidos en config.py

    prices, macro, ff5 = load_data()

    # ── 1. Validación ──────────────────────────────────────────────────────────
    print("[FE] Validando sincronizacion temporal...")
    _validate_temporal_alignment(prices, macro, ff5, sector_etfs)

    # ── 2. Series SPY para features y target ──────────────────────────────────
    spy_pct    = prices["SPY"].pct_change()
    spy_ret_1m  = spy_pct                          # retorno 1m SPY en t
    spy_ret_3m  = spy_pct.rolling(3).sum()         # retorno acum 3m SPY en t
    spy_ret_12m = spy_pct.rolling(12).sum()        # retorno acum 12m SPY en t
    spy_ret_t1  = spy_pct.shift(-1)                # retorno SPY en t+1 (para target)

    spy_ret_t1.index.name = "date"
    spy_ret_t1 = spy_ret_t1.rename("_spy_t1")

    # ── 3. Features ETF-específicas ───────────────────────────────────────────
    print("[FE] Calculando features de ETF...")
    panel = compute_etf_features(prices, sector_etfs)

    # ── 4. Features cross-seccionales (rank por fecha) ────────────────────────
    print("[FE] Calculando features cross-seccionales (rank por fecha)...")
    panel = add_cross_sectional_features(panel)

    # ── 5. Exceso de retorno vs SPY (features) ────────────────────────────────
    print("[FE] Calculando excesos de retorno vs SPY...")
    panel = add_excess_return_features(panel, spy_ret_1m, spy_ret_3m, spy_ret_12m)

    # ── 6. SPY return (feature de contexto de mercado) ────────────────────────
    spy_feat = pd.DataFrame({
        "date"        : spy_ret_1m.index,
        "spy_ret_1m"  : spy_ret_1m.values,
        "spy_ret_3m"  : spy_ret_3m.values,
        "spy_ret_12m" : spy_ret_12m.values,
    })
    panel = panel.merge(spy_feat, on="date", how="left")

    # ── 7. Macro FRED ─────────────────────────────────────────────────────────
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

    # Retorno SPY en t+1 (mismo para todos los ETFs ese mes)
    spy_t1_df = spy_ret_t1.reset_index()
    spy_t1_df.columns = ["date", "_spy_t1"]
    panel = panel.merge(spy_t1_df, on="date", how="left")

    # Target = exceso de retorno del ETF sobre el SPY el mes siguiente
    panel["target"] = panel["_etf_ret_t1"] - panel["_spy_t1"]
    panel.drop(columns=["_etf_ret_t1", "_spy_t1"], inplace=True)

    # ── 10. Limpieza y orden de columnas ──────────────────────────────────────
    n_before = len(panel)
    panel.dropna(subset=["target"], inplace=True)
    n_after  = len(panel)
    print(f"[FE] Filas sin target eliminadas: {n_before - n_after} "
          f"(ultimo mes de cada ETF)")

    id_cols      = ["date", "etf"]
    target_col   = ["target"]
    feature_cols = [c for c in panel.columns
                    if c not in id_cols + target_col + ["return"]]
    panel = panel[id_cols + feature_cols + target_col]

    # ── 11. Resumen ───────────────────────────────────────────────────────────
    print(f"\n[FE] Dataset final: {panel.shape}  |  "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"[FE] ETFs incluidos ({len(sector_etfs)}): {sector_etfs}")
    print(f"[FE] Target: exceso de retorno ETF vs SPY en t+1")

    _etf_base  = {"ret_1m","ret_3m","ret_6m","ret_12m","momentum_12_1","vol_6m"}
    _etf_rank  = {f"{c}_rank" for c in ["ret_1m","ret_3m","ret_6m","ret_12m","vol_6m"]}
    _etf_exc   = {"excess_ret_1m","excess_ret_3m","excess_ret_12m"}
    _spy_cols  = {"spy_ret_1m","spy_ret_3m","spy_ret_12m"}
    _ff5_cols  = {"Mkt-RF","SMB","HML","RMW","CMA","RF"}
    _macro_cols = {
        "CPI_YoY","IndProd_YoY","Unemployment","Unemp_Chg",
        "FedFunds","FedFunds_Chg","YieldSpread","GDP_YoY",
        "VIX","VIX_Chg","Gold_ret_1m","Gold_ret_3m",
        "T3M","T10","Term_Spread_10_3m","T10_Chg","HY_OAS","HY_OAS_Chg",
    }

    feat_set   = set(feature_cols)
    etf_base   = sorted(feat_set & _etf_base)
    etf_rank   = sorted(feat_set & _etf_rank)
    etf_exc    = sorted(feat_set & _etf_exc)
    spy_feats  = sorted(feat_set & _spy_cols)
    ff5_feats  = sorted(feat_set & _ff5_cols)
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
        print("\n[FE] Features con NaN (imputados por mediana en el pipeline ML):")
        for feat, pct in sorted(feats_nan, key=lambda x: -x[1]):
            print(f"  {feat:28s}  nulls={pct:5.1f}%")

    # Estadístico descriptivo del target
    t = panel["target"]
    print(f"\n[FE] Target (exceso vs SPY) — estadisticos:")
    print(f"  media={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")
    print(f"  % meses con exceso positivo: {(t > 0).mean():.1%}")

    panel.to_csv(f"{DATA_DIR}/features_panel.csv", index=False)
    print(f"\n  → Guardado: {DATA_DIR}/features_panel.csv")
    return panel


if __name__ == "__main__":
    panel = build_feature_matrix()
