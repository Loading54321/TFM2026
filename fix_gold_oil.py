"""Restore Gold columns in fred_macro.csv via yfinance (retry after rate limit)."""
import pandas as pd
import yfinance as yf
from config import DATA_DIR, DATA_START, DATA_END

macro_path = f"{DATA_DIR}/fred_macro.csv"
macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)

print(f"Current Gold_ret_1w non-NaN: {macro_df['Gold_ret_1w'].notna().sum() if 'Gold_ret_1w' in macro_df.columns else 'MISSING'}")

print("Downloading GC=F via yfinance...")
raw = yf.download("GC=F", start=DATA_START, end=DATA_END,
                  interval="1d", auto_adjust=True, progress=False)["Close"]
if isinstance(raw, pd.DataFrame):
    raw = raw.squeeze()
raw.index = pd.to_datetime(raw.index)
raw = raw.sort_index()
n = raw.notna().sum()
print(f"  Downloaded {n} valid obs")

if n > 100:
    raw = raw.resample("W-FRI").last().ffill()
    macro_df["Gold_ret_1w"] = raw.pct_change(1)
    macro_df["Gold_ret_4w"] = raw.pct_change(4)
    macro_df.to_csv(macro_path)
    print(f"  Saved. Gold_ret_1w non-NaN: {macro_df['Gold_ret_1w'].notna().sum()}")
else:
    print("  Still rate-limited. Gold left as-is (NaN).")
