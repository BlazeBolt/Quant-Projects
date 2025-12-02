import pandas as pd
import numpy as np


def cumulative_from_log(log_returns: pd.Series) -> pd.Series:
    return np.exp(log_returns.cumsum()) - 1


def annualized_return(log_returns: pd.Series, freq: int = 252) -> float:
    mean_daily = log_returns.mean()
    return np.exp(mean_daily * freq) - 1


def annualized_vol(log_returns: pd.Series, freq: int = 252) -> float:
    return log_returns.std() * np.sqrt(freq)


def sharpe_ratio(log_returns: pd.Series, freq: int = 252, rf: float = 0.0) -> float:
    ann_ret = annualized_return(log_returns, freq)
    ann_vol = annualized_vol(log_returns, freq)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol


def max_drawdown(log_returns: pd.Series) -> float:
    cum = np.exp(log_returns.cumsum())
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd.min()
