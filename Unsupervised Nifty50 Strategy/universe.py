import requests
import pandas as pd
from config import NIFTY_WIKI_URL, YAHOO_EXCHANGE_SUFFIX


def get_nifty50_universe() -> pd.DataFrame:
    """Scrape NIFTY 50 constituents and build Yahoo Finance tickers."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/114.0 Safari/537.36"
        )
    }
    response = requests.get(NIFTY_WIKI_URL, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(response.text)

    nifty = [t for t in tables if "Symbol" in t.columns][0].copy()
    nifty["Symbol"] = nifty["Symbol"].astype(str).str.strip()
    nifty["YahooSymbol"] = nifty["Symbol"].str.upper() + YAHOO_EXCHANGE_SUFFIX
    return nifty