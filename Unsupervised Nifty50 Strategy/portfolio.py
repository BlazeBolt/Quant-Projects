import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from config import OPT_FREQ_DAYS, MAX_WEIGHT_PER_STOCK


def optimize_weights(prices: pd.DataFrame,
                     lower_bound: float = 0.0) -> dict[str, float]:
    """
    Markowitz max-Sharpe weights with PyPortfolioOpt.
    prices: df with index=date, columns=tickers (price level).
    """
    if len(prices) < OPT_FREQ_DAYS:  # roughly 1 year
        raise ValueError("Not enough data to estimate covariance")

    mu = expected_returns.mean_historical_return(prices=prices, frequency=OPT_FREQ_DAYS)
    cov = risk_models.sample_cov(prices=prices, frequency=OPT_FREQ_DAYS)

    ef = EfficientFrontier(
        expected_returns=mu,
        cov_matrix=cov,
        weight_bounds=(lower_bound, MAX_WEIGHT_PER_STOCK),
        solver="SCS",
    )
    ef.max_sharpe()
    return ef.clean_weights()
