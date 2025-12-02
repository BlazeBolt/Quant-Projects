import pandas as pd
import yfinance as yf
from config import START_DATE, END_DATE


def download_panel(tickers: list[str],
                   start: pd.Timestamp | None = None,
                   end: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Download OHLCV panel from yfinance and return stacked MultiIndex (date, ticker).
    """
    start = pd.to_datetime(start or START_DATE)
    end = pd.to_datetime(end or END_DATE)

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
    ).stack()
    df.index.names = ["date", "ticker"]
    df.columns = df.columns.str.lower()
    return df


def download_daily_close(tickers: list[str],
                         start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.DataFrame:
    """
    Daily close price matrix: index=date, columns=tickers
    """
    raw = yf.download(tickers=tickers, start=start, end=end)
    # When multiple tickers: MultiIndex (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # single column → make it 2D
        close = raw[["Close"]]
    return close.dropna(axis=1, how="all")