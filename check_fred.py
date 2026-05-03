from fredapi import Fred
import os
from dotenv import load_dotenv
load_dotenv()
fred = Fred(api_key=os.getenv('FRED_API_KEY', ''))
# Gold London fixing prices
for sid in ['GOLDAMGBD228NLBM', 'GOLD', 'GOLDBLN', 'GVZCLS', 'WPU1022']:
    try:
        s = fred.get_series(sid, observation_start='2000-01-01', observation_end='2001-01-01')
        info = fred.get_series_info(sid)
        title = info['title']
        print(f'OK: {sid} -> {len(s)} obs | {title}')
    except Exception as e:
        print(f'FAIL: {sid} -> {e}')
