import pandas as pd

def filter_data_by_market_hours(df) -> pd.DataFrame:
    df['timedelta'] = pd.to_timedelta(df['TIME_M'])
    market_open = pd.Timedelta(hours=9, minutes=30)
    market_close = pd.Timedelta(hours=16, minutes=15) # include 15 minutes after close to capture closing auction
    return df[(df['timedelta'] >= market_open) & (df['timedelta'] < market_close)]