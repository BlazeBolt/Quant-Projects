"""
Time-series pipeline for index forecasting and backtesting.
Features:
- clean data download and handling
- log-return transformation and modeling on returns
- stationarity check and differencing logic
- train/validation split and walk-forward evaluation
- baseline (naive) comparison
- error metrics: RMSE, MAE, MAPE, directional accuracy
- risk/PnL framing: simple strategy converting forecasts to signals, PnL, Sharpe, max drawdown
- modular functions, clear main entry, and configuration at top

Notes:
- Requires: pandas, numpy, matplotlib, yfinance, statsmodels, pmdarima (optional). If pmdarima is unavailable, code falls back to manual ARIMA order or grid search using statsmodels.
- Do NOT run `pip install` inside the script; add dependencies to requirements.txt for your repo.

Usage example:
    python quant_arima_pipeline.py

"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import pmdarima; if unavailable, we still run using SARIMAX with simple heuristics.
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

# ---------------------------
# Config / Hyperparameters
# ---------------------------

@dataclass
class Config:
    ticker: str = '^NSEI'
    lookback_days: int = 365  # data window
    forecast_horizon: int = 5  # forecast horizon in business days for each step
    walk_forward_steps: int = 20  # how many rolling forecasts to make (walk-forward)
    train_initial_days: int = 200  # initial train size (days)
    seasonal_m: Optional[int] = None  # seasonality (e.g., 5 for weekly in business-days) or None
    arima_order: Tuple[int, int, int] = (2, 0, 2)  # fallback order when auto_arima unavailable
    verbose: bool = True
    transaction_cost: float = 0.0005  # per-trade proportional cost (0.05%)

cfg = Config()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Utility functions
# ---------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and return a DataFrame with DateTimeIndex.
    Ensures business-day frequency and fills missing days with forward fill.
    """
    logger.info(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker or date range.")

    # Keep only expected columns and ensure DateTimeIndex
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis('Date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Reindex to business days to make forecasting index consistent
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(idx)
    # Forward fill prices, fill initial NA by backfill if necessary
    df['Close'] = df['Close'].ffill().bfill()
    df[['Open', 'High', 'Low', 'Volume']] = df[['Open', 'High', 'Low', 'Volume']].ffill().bfill()

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log price and log returns. Drop NaNs and keep DateTimeIndex.
    """
    df = df.copy()
    df['log_close'] = np.log(df['Close'])
    df['return'] = df['log_close'].diff()  # log return
    df = df.dropna()
    return df


def check_stationarity(series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
    """Run ADF test and return dictionary with results.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    out = {
        'adf_stat': result[0],
        'pvalue': result[1],
        'used_lag': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] <= significance
    }
    return out


def train_test_split_timeseries(series: pd.Series, train_size: int) -> Tuple[pd.Series, pd.Series]:
    """Split series by number of observations for train size. Returns train, test.
    """
    if train_size >= len(series):
        raise ValueError("train_size must be smaller than series length")
    train = series.iloc[:train_size].copy()
    test = series.iloc[train_size:].copy()
    return train, test


def fit_arima_auto(series: pd.Series, cfg: Config) -> Any:
    """Fit ARIMA using pmdarima.auto_arima if available, else fallback to SARIMAX with cfg.arima_order.
    Returns a fitted model-like object with `predict(start, end)`.
    """
    if HAS_PMDARIMA:
        logger.info("Fitting auto_arima (pmdarima)")
        # seasonal setting based on cfg.seasonal_m
        seasonal = cfg.seasonal_m is not None and cfg.seasonal_m > 1
        model = auto_arima(series, start_p=0, start_q=0, max_p=4, max_q=4, m=cfg.seasonal_m or 1,
                           seasonal=seasonal, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        return model
    else:
        logger.info("pmdarima not found — falling back to statsmodels SARIMAX with order %s" % (cfg.arima_order,))
        model = SARIMAX(series, order=cfg.arima_order, enforce_stationarity=False, enforce_invertibility=False)
        fit_res = model.fit(disp=False)
        return fit_res


def forecast_model(fitted, steps: int) -> pd.Series:
    """Unified forecasting wrapper for pmdarima and statsmodels results.
    Returns a pandas Series of length `steps`.

    This tries several common interfaces in order:
      1) pmdarima-style .predict(n_periods=steps) or .predict(steps)
      2) statsmodels ResultsWrapper .get_forecast(steps=...).predicted_mean
      3) statsmodels .predict(start, end) using model/data endog length
      4) fallback to .predict(steps) if available
    """
    # 1) Try pmdarima-style predict(n_periods)
    try:
        if hasattr(fitted, 'predict'):
            # first try keyword arg (pmdarima)
            try:
                preds = fitted.predict(n_periods=steps)
                return pd.Series(np.asarray(preds))
            except TypeError:
                # try positional argument
                try:
                    preds = fitted.predict(steps)
                    return pd.Series(np.asarray(preds))
                except TypeError:
                    pass
    except Exception:
        pass

    # 2) Try statsmodels get_forecast
    try:
        if hasattr(fitted, 'get_forecast'):
            forecast_res = fitted.get_forecast(steps=steps)
            preds = forecast_res.predicted_mean
            return pd.Series(np.asarray(preds))
    except Exception:
        pass

    # 3) Try statsmodels predict(start, end) using available length
    try:
        # many statsmodels result objects provide .model.endog
        if hasattr(fitted, 'model') and hasattr(fitted.model, 'endog'):
            start = len(fitted.model.endog)
            end = start + steps - 1
            preds = fitted.predict(start=start, end=end)
            return pd.Series(np.asarray(preds))

        # older/other result objects may expose .data.endog
        if hasattr(fitted, 'data') and hasattr(fitted.data, 'endog'):
            start = len(fitted.data.endog)
            end = start + steps - 1
            preds = fitted.predict(start=start, end=end)
            return pd.Series(np.asarray(preds))
    except Exception:
        pass

    # 4) Last resort: try predict with positional steps again
    try:
        if hasattr(fitted, 'predict'):
            preds = fitted.predict(steps)
            return pd.Series(np.asarray(preds))
    except Exception:
        pass

    raise RuntimeError("Unable to produce forecast with given fitted model — unsupported model object or interface.")


# ---------------------------
# Evaluation metrics
# ---------------------------

def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(true, pred)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_absolute_error(true, pred))


def mape(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.array(true), np.array(pred)
    # exclude zeros to avoid divide-by-zero; if all zeros, return nan
    mask = true != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))) * 100


def directional_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    # percent of times sign(pred) == sign(true)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    return float(np.mean(true_sign == pred_sign))


# ---------------------------
# Simple backtest
# ---------------------------

def generate_signals(forecast_returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Simple rule: go long (1) if forecasted return > threshold, else 0 (flat).
    Returns position series aligned to forecast index.
    """
    pos = (forecast_returns > threshold).astype(int)
    return pos


def run_simple_backtest(price_series: pd.Series, signals: pd.Series, transaction_cost: float = 0.0) -> Dict[str, Any]:
    """Run a vectorized backtest given price series and discrete signals (0/1), where signals are the position for the day.
    Returns PnL series and summary stats. Handles empty data robustly.
    """
    # compute daily returns from prices
    returns = price_series.pct_change().fillna(0)

    # align signals to returns: assume position at open captured by previous signal (simple assumption)
    signals = signals.reindex(returns.index).fillna(0)

    # strategy returns: position * returns - transaction costs when position changes
    pos_change = signals.diff().abs().fillna(0)
    strat_returns = signals * returns - pos_change * transaction_cost

    # Ensure strat_returns is a 1-D Series (if it's a DataFrame, reduce to a single series)
    if hasattr(strat_returns, "squeeze"):
        try:
            strat_returns = strat_returns.squeeze()
        except Exception:
            pass

    # cumulative returns (vectorized)
    cum_returns = (1 + strat_returns).cumprod() - 1

    # performance metrics — coerce to plain Python floats safely
    try:
        avg_ret = float(np.nanmean(strat_returns)) * 252
    except Exception:
        avg_ret = float('nan')
    try:
        # Use ddof=1 if more than one observation to match sample std; fall back otherwise
        vol = float(np.nanstd(strat_returns, ddof=1)) * np.sqrt(252)
    except Exception:
        vol = float('nan')

    # compute sharpe robustly
    if np.isnan(vol) or vol == 0:
        sharpe = np.nan
    else:
        sharpe = (avg_ret / vol)

    # max drawdown — robust to Series/DataFrame shapes
    if getattr(cum_returns, "size", 0) == 0:
        max_dd = float('nan')
    else:
        # convert values to numpy array and compute min drawdown
        running_max = np.maximum.accumulate(np.asarray(cum_returns))
        drawdown = np.asarray(cum_returns) - running_max
        try:
            max_dd_val = np.nanmin(drawdown)
            max_dd = float(max_dd_val) if np.isfinite(max_dd_val) else float('nan')
        except Exception:
            max_dd = float('nan')

    summary = {
        'strategy_returns': strat_returns,
        'cumulative_returns': cum_returns,
        'annualized_return': avg_ret,
        'annualized_volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }

    return summary


# ---------------------------
# Walk-forward evaluation
# ---------------------------

def walk_forward_evaluation(series: pd.Series, cfg: Config) -> Dict[str, Any]:
    """Perform walk-forward forecasting and evaluation on log returns.
    Steps:
    - Start with initial train window of cfg.train_initial_days
    - For i in range(cfg.walk_forward_steps): fit model on current train, forecast cfg.forecast_horizon days
    - Store forecasts and move window forward by cfg.forecast_horizon (expanding window)
    Returns metrics and forecasts aligned to dates.
    """
    results = []
    n = len(series)
    start_train = cfg.train_initial_days
    if start_train >= n:
        raise ValueError("train_initial_days must be less than series length")

    # We'll use an expanding-window approach
    train_end_idx = start_train
    forecasts = []
    forecast_index = []

    for step in range(cfg.walk_forward_steps):
        if train_end_idx >= n - 1:
            break
        train_series = series.iloc[:train_end_idx].copy()
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + cfg.forecast_horizon, n)
        actual_series = series.iloc[test_start_idx:test_end_idx]

        # Fit model
        fitted = fit_arima_auto(train_series, cfg)
        preds = forecast_model(fitted, steps=len(actual_series))

        # preds correspond to next len(actual_series) periods; align index
        preds.index = actual_series.index

        # record
        forecasts.append(preds)
        forecast_index.append((actual_series.index[0], actual_series.index[-1]))

        # metrics for this step
        m = {
            'step': step,
            'train_end': train_series.index[-1],
            'test_start': actual_series.index[0],
            'test_end': actual_series.index[-1],
            'rmse': rmse(actual_series.values, preds.values),
            'mae': mae(actual_series.values, preds.values),
            'mape': mape(actual_series.values, preds.values),
            'directional_accuracy': directional_accuracy(actual_series.values, preds.values)
        }
        results.append(m)

        # expand train window
        train_end_idx = test_end_idx

    # combine forecasts into a single series
    if forecasts:
        all_forecasts = pd.concat(forecasts).sort_index()
    else:
        all_forecasts = pd.Series([], dtype=float)

    metrics_df = pd.DataFrame(results)

    return {
        'forecasts': all_forecasts,
        'metrics': metrics_df
    }


# ---------------------------
# Main pipeline
# ---------------------------

def main(cfg: Config):
    from datetime import date, timedelta
    end = date.today()
    start = end - pd.Timedelta(days=cfg.lookback_days)

    df_raw = download_data(cfg.ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    df = prepare_features(df_raw)

    # series to model: log returns
    series = df['return']

    # basic stationarity check
    adf_res = check_stationarity(series.dropna())
    logger.info(f"ADF p-value: {adf_res['pvalue']:.4f} — stationary? {adf_res['is_stationary']}")

    # Walk-forward evaluation
    wf = walk_forward_evaluation(series, cfg)
    forecasts = wf['forecasts']
    metrics_df = wf['metrics']

    logger.info("Walk-forward metrics (per-step):")
    logger.info('\n' + metrics_df.to_string(index=False))

    # Baseline naive forecast: yesterday's return as forecast for next day
    # Align baseline to forecast index
    baseline_forecasts = series.shift(1).reindex(forecasts.index)

    # Compare metrics aggregated across all steps
    common_index = forecasts.index.intersection(series.index)
    y_true = series.reindex(common_index)
    y_pred = forecasts.reindex(common_index).fillna(0)
    y_base = baseline_forecasts.reindex(common_index).fillna(0)

    metrics = {
        'model_rmse': rmse(y_true.values, y_pred.values),
        'model_mae': mae(y_true.values, y_pred.values),
        'model_mape': mape(y_true.values, y_pred.values),
        'model_dir_acc': directional_accuracy(y_true.values, y_pred.values),
        'baseline_rmse': rmse(y_true.values, y_base.values),
        'baseline_dir_acc': directional_accuracy(y_true.values, y_base.values)
    }

    logger.info("Aggregate metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # Build trading signals from forecasted returns
    # We'll use the forecasts aligned to price series' index (use Close price from df_raw)
    price = df_raw['Close'].reindex(forecasts.index.union(df_raw.index)).ffill()
    # signal threshold could be tuned; using 0 = go long if forecast positive
    signals = generate_signals(forecasts, threshold=0.0)

    backtest_res = run_simple_backtest(price_series=price.reindex(signals.index), signals=signals, transaction_cost=cfg.transaction_cost)

    logger.info("Backtest summary:")
    logger.info(f"Annualized return: {backtest_res['annualized_return']:.4f}")
    logger.info(f"Annualized volatility: {backtest_res['annualized_volatility']:.4f}")
    logger.info(f"Sharpe: {backtest_res['sharpe']:.4f}")
    logger.info(f"Max drawdown: {backtest_res['max_drawdown']:.4f}")

    # Plotting summary charts
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(series.cumsum(), label='Cumulative True Log-Returns')
        plt.plot(forecasts.cumsum(), label='Cumulative Forecast Log-Returns')
        plt.legend()
        plt.title('Cumulative log-returns: True vs Forecast')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(backtest_res['cumulative_returns'], label='Strategy Cumulative Returns')
        plt.title('Strategy Cumulative Returns')
        plt.show()
    except Exception:
        pass

    # save artifacts for resume: metrics_df, forecasts
    metrics_df.to_csv('wf_metrics.csv', index=False)
    forecasts.to_csv('forecasts.csv')
    pd.DataFrame([metrics]).to_csv('aggregate_metrics.csv', index=False)

    logger.info('Saved metrics to wf_metrics.csv, forecasts.csv, aggregate_metrics.csv')


if __name__ == '__main__':
    main(cfg)