"""
data_download.py
================
Descarga y guarda los datos crudos:
  - ETFs sectoriales (yfinance, mensual)
  - Variables macroeconómicas (FRED API)
  - Factores Fama & French 5 (pandas_datareader)

Todos los datos se guardan en /data/
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from fredapi import Fred
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import (
    FRED_API_KEY, DATA_START, DATA_END, DATA_DIR,
    SECTOR_ETFS, BENCHMARK, ALL_TICKERS, FRED_SERIES
)

os.makedirs(DATA_DIR, exist_ok=True)


# 1. ETFs
def download_etfs() -> pd.DataFrame:
    """
    Descarga precios ajustados de cierre a frecuencia mensual.
    Si la descarga falla y ya existe etf_prices.csv con datos, mantiene el anterior.
    """
    prices_path = f"{DATA_DIR}/etf_prices.csv"
    
    print("[ETF] Descargando precios mensuales...")
    try:
        raw = yf.download(
            ALL_TICKERS,
            start=DATA_START,
            end=DATA_END,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )["Close"]

        # Asegurar columnas correctas
        raw.index = pd.to_datetime(raw.index).to_period("M").to_timestamp("M")
        raw.sort_index(inplace=True)
        
        # Validar que tenemos datos
        if raw.shape[0] < 100:  # Al menos 100 meses de datos
            raise ValueError(f"Muy pocos datos descargados: {raw.shape[0]} filas")
            
        raw.to_csv(prices_path)
        print(f"  -> {raw.shape[0]} filas, {raw.shape[1]} columnas guardadas en {prices_path}")
        return raw
    except Exception as e:
        print(f"  [WARN] Descarga falló: {e}")
        # Intentar cargar archivo anterior
        if os.path.exists(prices_path):
            try:
                raw = pd.read_csv(prices_path, index_col=0, parse_dates=True)
                print(f"  -> Usando archivo anterior: {raw.shape[0]} filas, {raw.shape[1]} columnas")
                return raw
            except:
                raise ValueError(f"No hay datos: descarga falló y {prices_path} no existe o está corrupto")
        else:
            raise
    return raw


# 2. Variables macro (FRED)
def download_fred() -> pd.DataFrame:
    """
    Descarga series de FRED y aplica transformaciones anti-leakage.

    Series y transformaciones:
      CPI          -> CPI_YoY      (variacion % interanual, sin lookahead)
      IndProd      -> IndProd_YoY  (variacion % interanual)
      Unemployment -> Unemp_Chg    (diff mensual) + nivel
      FedFunds     -> FedFunds_Chg (diff mensual) + nivel
      YieldSpread  -> nivel (10yr-2yr, ya es un diferencial)

      VIX          -> nivel + VIX_Chg (diff mensual)
                      Mide el miedo del mercado; util para diferenciar
                      periodos de panico (Bear) de calma (Bull/Ranging)

      Gold         -> Gold_ret_1m, Gold_ret_3m (retornos 1m y 3m)
                      Activo refugio / señal de inflacion; se convierte a
                      retorno para eliminar la tendencia de precio

      T3M, T10     -> niveles + Term_Spread_10_3m (T10 - T3M) + T10_Chg
                      T10-T3M es mejor predictor de recesion que T10-T2Y
                      (la curva de rendimientos a 3m-10a tiene mayor poder
                      predictivo segun Estrella & Mishkin 1998)

      HY_OAS       -> nivel + HY_OAS_Chg (diff mensual)
                      Spread de credito HY: widening = estres financiero;
                      util para distinguir Bear de Ranging
    """
    print("[FRED] Descargando variables macroeconómicas...")
    fred = Fred(api_key=FRED_API_KEY)
    frames = {}

    for series_id, name in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=DATA_START,
                observation_end=DATA_END,
            )
            s = s.resample("ME").last()
            s.index = s.index.to_period("M").to_timestamp("M")
            frames[name] = s
            print(f"  [OK] {series_id:25s} -> '{name}'  ({len(s)} obs)")
        except Exception as e:
            print(f"  [WARN] {series_id}: {e}")

    df = pd.DataFrame(frames)
    df.sort_index(inplace=True)

    # ── GDP: interpolacion trimestral -> mensual ──────────────────────────────
    # GDPC1 es trimestral; se interpola linealmente a mensual y luego
    # se aplica forward-fill para garantizar causalidad estricta:
    # en cada mes solo se usa el ultimo dato trimestral publicado.
    if "GDP" in df.columns:
        # Crear indice mensual completo del rango de datos
        monthly_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="ME"
        )
        monthly_idx = monthly_idx.to_period("M").to_timestamp("M")
        # Reindexar: NaN entre trimestres, luego forward-fill (anti-leakage)
        df["GDP"] = df["GDP"].reindex(monthly_idx).ffill()
        df = df.reindex(monthly_idx)

    # ── Transformaciones ─────────────────────────────────────────────────────

    # Series existentes
    df["CPI_YoY"]       = df["CPI"].pct_change(12)
    df["IndProd_YoY"]   = df["IndProd"].pct_change(12)
    df["Unemp_Chg"]     = df["Unemployment"].diff(1)
    df["FedFunds_Chg"]  = df["FedFunds"].diff(1)
    df.drop(columns=["CPI", "IndProd"], inplace=True)

    # PIB real: variacion interanual (GDP_YoY)
    # Se usa pct_change(4) sobre el nivel trimestral antes de interpolar
    # pero aqui trabajamos con el nivel ya interpolado, por lo que
    # pct_change(12) da la variacion anual en el nivel mensual.
    if "GDP" in df.columns:
        df["GDP_YoY"] = df["GDP"].pct_change(12)
        df.drop(columns=["GDP"], inplace=True)

    # VIX
    if "VIX" in df.columns:
        df["VIX_Chg"] = df["VIX"].diff(1)

    # Gold: precio -> retornos (elimina tendencia de precio)
    if "Gold" in df.columns:
        df["Gold_ret_1m"] = df["Gold"].pct_change(1)
        df["Gold_ret_3m"] = df["Gold"].pct_change(3)
        df.drop(columns=["Gold"], inplace=True)

    # Tipos de interes: curva de rendimientos
    if "T3M" in df.columns and "T10" in df.columns:
        df["Term_Spread_10_3m"] = df["T10"] - df["T3M"]   # pendiente curva
        df["T10_Chg"]           = df["T10"].diff(1)        # cambio tipos largos

    # Credito HY
    if "HY_OAS" in df.columns:
        df["HY_OAS_Chg"] = df["HY_OAS"].diff(1)

    df.to_csv(f"{DATA_DIR}/fred_macro.csv")
    feature_list = [c for c in df.columns]
    print(f"\n  -> {df.shape} guardadas en {DATA_DIR}/fred_macro.csv")
    print(f"  -> Features macro: {feature_list}")
    return df


# 3. Factores Fama & French
def download_ff5() -> pd.DataFrame:
    """
    Descarga los 5 factores de Fama-French (mensual) desde
    pandas_datareader (no requiere API key).
    Devuelve retornos en decimal (no %).
    """
    print("[FF5] Descargando factores Fama-French 5...")
    ff5 = pdr.famafrench.FamaFrenchReader(
        "F-F_Research_Data_5_Factors_2x3",
        start=DATA_START, end=DATA_END
    ).read()[0]

    ff5.index = ff5.index.to_timestamp("M")
    ff5 = ff5 / 100   # convertir de % a decimal
    ff5.sort_index(inplace=True)

    ff5.to_csv(f"{DATA_DIR}/ff5_factors.csv")
    print(f"  -> {ff5.shape} guardados en {DATA_DIR}/ff5_factors.csv")
    return ff5


# Main
if __name__ == "__main__":
    etf_prices = download_etfs()
    fred_macro  = download_fred()
    ff5_factors = download_ff5()
    print("\n[OK] Datos descargados correctamente en /data/")
