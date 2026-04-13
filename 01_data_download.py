"""
01_data_download.py
===================
Descarga y guarda los datos crudos a frecuencia semanal (último viernes, ffill universal):
  - ETFs sectoriales (yfinance, diario → resample W-FRI + ffill)
  - Variables macroeconómicas (FRED API, diario o mensual → resample W-FRI + ffill)
  - Oro (yfinance, diario → resample W-FRI + ffill)
  - Factores Fama & French 5 (mensual → resample W-FRI + ffill)

Todos los datos se guardan en /data/
"""

import io
import os
import zipfile
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
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
    Descarga precios ajustados de cierre diarios y resamplea al último viernes de
    cada semana (W-FRI) con forward-fill para cubrir festivos/gaps.
    Si la descarga falla y ya existe etf_prices.csv con datos, mantiene el anterior.
    """
    prices_path = f"{DATA_DIR}/etf_prices.csv"

    print("[ETF] Descargando precios diarios y resampleando a W-FRI...")
    try:
        raw = yf.download(
            ALL_TICKERS,
            start=DATA_START,
            end=DATA_END,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]

        raw.index = pd.to_datetime(raw.index)
        raw.sort_index(inplace=True)

        # Resamplear al último viernes de cada semana; ffill para festivos
        prices = raw.resample("W-FRI").last().ffill()

        if prices.shape[0] < 100:
            raise ValueError(f"Muy pocos datos descargados: {prices.shape[0]} filas")

        prices.to_csv(prices_path)
        print(f"  -> {prices.shape[0]} semanas, {prices.shape[1]} columnas guardadas en {prices_path}")
        return prices
    except Exception as e:
        print(f"  [WARN] Descarga falló: {e}")
        if os.path.exists(prices_path):
            try:
                prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
                print(f"  -> Usando archivo anterior: {prices.shape[0]} filas, {prices.shape[1]} columnas")
                return prices
            except Exception:
                raise ValueError(
                    f"No hay datos: descarga falló y {prices_path} no existe o está corrupto"
                )
        else:
            raise


# 2. Variables macro (FRED)
def download_fred() -> pd.DataFrame:
    """
    Descarga series de FRED y aplica transformaciones, todo sobre índice semanal W-FRI.

    Series diarias (resample W-FRI directo):
      DFF       -> FedFunds  -> FedFunds_Chg  (diff semanal)
      DTB3      -> T3M       -> nivel + Term_Spread_10_3m + T10_Chg
      DGS10     -> T10       -> nivel + Term_Spread_10_3m + T10_Chg
      T10Y2Y    -> YieldSpread (nivel, diferencial 10yr-2yr)
      VIXCLS    -> VIX       -> VIX_Chg (diff semanal)
      BAMLH0A0HYM2 -> HY_OAS -> HY_OAS_Chg (diff semanal)

    Series mensuales (resample W-FRI + ffill):
      CPIAUCSL  -> CPI       -> CPI_YoY     (pct_change 52 semanas)
      UNRATE    -> Unemployment -> Unemp_Chg (diff 4 semanas ≈ mensual)
      INDPRO    -> IndProd   -> IndProd_YoY  (pct_change 52 semanas)
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
            s.index = pd.to_datetime(s.index)
            # Resamplear al último viernes de cada semana; ffill llena festivos/gaps
            s = s.resample("W-FRI").last().ffill()
            frames[name] = s
            print(f"  [OK] {series_id:25s} -> '{name}'  ({len(s)} obs semanales)")
        except Exception as e:
            print(f"  [WARN] {series_id}: {e}")

    df = pd.DataFrame(frames)
    df.sort_index(inplace=True)

    # ── Transformaciones ──────────────────────────────────────────────────────

    # Series mensuales ffilled → YoY con 52 semanas
    if "CPI" in df.columns:
        df["CPI_YoY"] = df["CPI"].pct_change(52)
        df.drop(columns=["CPI"], inplace=True)

    if "IndProd" in df.columns:
        df["IndProd_YoY"] = df["IndProd"].pct_change(52)
        df.drop(columns=["IndProd"], inplace=True)

    # Desempleo: diff(4) captura cambio mes-a-mes en índice semanal ffilled
    if "Unemployment" in df.columns:
        df["Unemp_Chg"] = df["Unemployment"].diff(4)

    # Series diarias → diff semanal (diff(1))
    if "FedFunds" in df.columns:
        df["FedFunds_Chg"] = df["FedFunds"].diff(1)

    if "VIX" in df.columns:
        df["VIX_Chg"] = df["VIX"].diff(1)

    if "T3M" in df.columns and "T10" in df.columns:
        df["Term_Spread_10_3m"] = df["T10"] - df["T3M"]
        df["T10_Chg"] = df["T10"].diff(1)

    if "HY_OAS" in df.columns:
        df["HY_OAS_Chg"] = df["HY_OAS"].diff(1)

    df.to_csv(f"{DATA_DIR}/fred_macro.csv")
    feature_list = list(df.columns)
    print(f"\n  -> {df.shape} guardadas en {DATA_DIR}/fred_macro.csv")
    print(f"  -> Features macro: {feature_list}")
    return df


# 2b. Oro via yfinance (FRED retiró la serie LBMA en enero 2022)
def download_gold() -> pd.DataFrame:
    """
    Descarga el precio diario del oro (GC=F) y resamplea a W-FRI + ffill.
    Calcula retornos 1 semana y 4 semanas (≈ 1 mes).
    """
    print("[GOLD] Descargando oro (GC=F) via yfinance (diario → W-FRI)...")
    raw = yf.download(
        "GC=F", start=DATA_START, end=DATA_END,
        interval="1d", auto_adjust=True, progress=False
    )["Close"]

    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().rename("Gold")

    # Resamplear al último viernes; ffill para festivos
    raw = raw.resample("W-FRI").last().ffill()

    gold_ret_1w = raw.pct_change(1).rename("Gold_ret_1w")
    gold_ret_4w = raw.pct_change(4).rename("Gold_ret_4w")

    n_valid = gold_ret_1w.notna().sum()
    print(f"  -> Gold_ret_1w / Gold_ret_4w: {n_valid} obs válidas")
    return pd.DataFrame({"Gold_ret_1w": gold_ret_1w, "Gold_ret_4w": gold_ret_4w})


# 3. Factores Fama & French
def download_ff5() -> pd.DataFrame:
    """
    Descarga los 5 factores de Fama-French (mensual) directamente desde
    el sitio de Kenneth French y resamplea a W-FRI + ffill.
    Devuelve retornos en decimal (no %).
    """
    FF5_URL = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    print("[FF5] Descargando factores Fama-French 5 desde Kenneth French website...")
    resp = requests.get(FF5_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw_text = z.read(csv_name).decode("utf-8", errors="replace")

    # El CSV de French tiene cabecera de texto libre antes de los datos;
    # los datos empiezan cuando la primera columna es un entero de 6 dígitos (YYYYMM).
    lines = raw_text.splitlines()
    data_lines = []
    header = None
    for line in lines:
        parts = line.strip().split(",")
        if not parts or not parts[0].strip():
            continue
        if parts[0].strip().isdigit() and len(parts[0].strip()) == 6:
            data_lines.append(line)
        elif header is None and any(
            kw in line for kw in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        ):
            header = line

    ff5 = pd.read_csv(
        io.StringIO((header or "Date,Mkt-RF,SMB,HML,RMW,CMA,RF") + "\n" + "\n".join(data_lines))
    )
    ff5.columns = [c.strip() for c in ff5.columns]
    ff5["Date"] = pd.to_datetime(ff5["Date"].astype(str), format="%Y%m")
    ff5 = ff5.set_index("Date")
    ff5 = ff5.apply(pd.to_numeric, errors="coerce") / 100   # % -> decimal

    # Filtrar rango de fechas del proyecto
    ff5 = ff5.loc[
        (ff5.index >= pd.Timestamp(DATA_START)) &
        (ff5.index <= pd.Timestamp(DATA_END))
    ]
    ff5.sort_index(inplace=True)

    # Resamplear a W-FRI + ffill para alinear con el resto del pipeline
    ff5 = ff5.resample("W-FRI").last().ffill()

    ff5.to_csv(f"{DATA_DIR}/ff5_factors.csv")
    print(f"  -> {ff5.shape} guardados en {DATA_DIR}/ff5_factors.csv")
    return ff5


# Main
if __name__ == "__main__":
    etf_prices  = download_etfs()
    fred_macro  = download_fred()
    gold_df     = download_gold()
    ff5_factors = download_ff5()

    # Incorporar Gold al CSV macro (merge sobre índice fecha)
    macro_path = f"{DATA_DIR}/fred_macro.csv"
    macro_df   = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro_df = macro_df.join(gold_df, how="left")
    macro_df.to_csv(macro_path)
    print(f"  -> Gold añadido a {macro_path}  "
          f"(Gold_ret_1w NaN: {macro_df['Gold_ret_1w'].isna().sum()})")

    print("\n[OK] Datos descargados correctamente en /data/ (frecuencia semanal W-FRI)")
