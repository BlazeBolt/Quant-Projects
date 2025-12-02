# config.py
import datetime as dt

NIFTY_WIKI_URL = "https://en.wikipedia.org/wiki/NIFTY_50"
YAHOO_EXCHANGE_SUFFIX = ".NS"

START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2023, 9, 27)  # for backtest; for paper trading use dt.date.today()

TECH_RSI_LENGTH = 20
TECH_BB_LENGTH = 20
TECH_ATR_LENGTH = 14

MONTHLY_LAGS = [1, 2, 3, 6, 9, 12]
OUTLIER_CUTOFF = 0.005

ROLLING_LIQ_YEARS = 5
MIN_LIQ_MONTHS = 12
MAX_LIQ_RANK = 50  # keep liquid names

KMEANS_CLUSTERS = 4

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

MAX_WEIGHT_PER_STOCK = 0.10
OPT_FREQ_DAYS = 252

BENCH_TICKER = "^NSEI"