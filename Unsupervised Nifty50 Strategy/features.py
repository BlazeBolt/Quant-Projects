import numpy as np
import pandas as pd
import pandas_ta
from config import (
    TECH_RSI_LENGTH,
    TECH_BB_LENGTH,
    TECH_ATR_LENGTH,
    MONTHLY_LAGS,
    OUTLIER_CUTOFF,
    ROLLING_LIQ_YEARS,
    MIN_LIQ_MONTHS,
    MAX_LIQ_RANK,
)


def add_technical_features(df: pd.DataFrame,
                           price_col: str = "close") -> pd.DataFrame:
    """Add GK vol, RSI, BBands, ATR, MACD and inr_volume to (date, ticker) panel."""
    df = df.copy()

    # Garman–Klass volatility
    df["garman_klass_vol"] = (
        ((np.log(df["high"]) - np.log(df["low"])) ** 2) / 2
        - (2 * np.log(2) - 1) * ((np.log(df[price_col]) - np.log(df["open"])) ** 2)
    )

    # RSI
    df["rsi"] = df.groupby(level="ticker")[price_col].transform(
        lambda x: pandas_ta.rsi(close=x, length=TECH_RSI_LENGTH)
    )

    # Bollinger Bands on log(1+price)
    def bb_low(x):
        return pandas_ta.bbands(close=np.log1p(x), length=TECH_BB_LENGTH).iloc[:, 0]

    def bb_mid(x):
        return pandas_ta.bbands(close=np.log1p(x), length=TECH_BB_LENGTH).iloc[:, 1]

    def bb_high(x):
        return pandas_ta.bbands(close=np.log1p(x), length=TECH_BB_LENGTH).iloc[:, 2]

    df["bb_low"] = df.groupby(level="ticker")[price_col].transform(bb_low)
    df["bb_mid"] = df.groupby(level="ticker")[price_col].transform(bb_mid)
    df["bb_high"] = df.groupby(level="ticker")[price_col].transform(bb_high)

    # ATR normalized per stock
    def compute_atr(stock_data: pd.DataFrame) -> pd.Series:
        atr = pandas_ta.atr(
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data[price_col],
            length=TECH_ATR_LENGTH,
        )
        return atr.sub(atr.mean()).div(atr.std())

    df["atr"] = df.groupby(level="ticker", group_keys=False).apply(compute_atr)

    # MACD normalized per stock
    def compute_macd(close: pd.Series) -> pd.Series:
        macd_df = pandas_ta.macd(close=close)
        if macd_df is None or macd_df.empty:
            return pd.Series(index=close.index, data=np.nan)
        macd = macd_df.iloc[:, 0]
        return macd.sub(macd.mean()).div(macd.std())

    df["macd"] = df.groupby(level="ticker", group_keys=False)[price_col].apply(
        compute_macd
    )

    # Liquidity proxy: close * volume in INR millions
    df["inr_volume"] = (df[price_col] * df["volume"]) / 1e6

    return df


def build_monthly_panel(df: pd.DataFrame,
                        price_col: str = "close") -> pd.DataFrame:
    """
    Convert daily panel to monthly (date, ticker):
    - price: month-end close
    - inr_volume: monthly average
    - features: month-end values
    """
    df = df.copy()

    feature_cols = [
        c
        for c in df.columns
        if c not in ["inr_volume", "volume", "open", "high", "low", price_col]
    ]

    monthly_inr = (
        df["inr_volume"]
        .unstack("ticker")
        .resample("M")
        .mean()
        .stack("ticker")
        .to_frame("inr_volume")
    )

    monthly_price = (
        df[price_col]
        .unstack("ticker")
        .resample("M")
        .last()
        .stack("ticker")
        .to_frame("price")
    )

    monthly_features = (
        df[feature_cols]
        .unstack("ticker")
        .resample("M")
        .last()
        .stack("ticker")
    )

    data = pd.concat([monthly_price, monthly_inr, monthly_features], axis=1).dropna()
    return data


def apply_liquidity_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Keep most liquid names using rolling average inr_volume."""
    data = data.copy()
    window = ROLLING_LIQ_YEARS * 12

    data["inr_volume"] = (
        data["inr_volume"]
        .unstack("ticker")
        .rolling(window, min_periods=MIN_LIQ_MONTHS)
        .mean()
        .stack()
    )

    data["inr_vol_rank"] = data.groupby("date")["inr_volume"].rank(ascending=False)
    data = data[data["inr_vol_rank"] < MAX_LIQ_RANK].drop(
        ["inr_volume", "inr_vol_rank"], axis=1
    )
    return data


def add_lagged_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add monthly lagged returns on 'price' for each ticker.
    """
    def calculate_returns(df_: pd.DataFrame) -> pd.DataFrame:
        for lag in MONTHLY_LAGS:
            col = f"return_{lag}m"
            x = df_["price"].pct_change(lag)
            x = x.pipe(
                lambda s: s.clip(
                    lower=s.quantile(OUTLIER_CUTOFF),
                    upper=s.quantile(1 - OUTLIER_CUTOFF),
                )
            )
            df_[col] = (1 + x) ** (1 / lag) - 1
        return df_

    data = (
        data.groupby(level="ticker", group_keys=False)
        .apply(calculate_returns)
        .dropna()
    )
    return data