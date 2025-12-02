import pandas as pd
import numpy as np
from portfolio import optimize_weights
from config import OPT_FREQ_DAYS


def build_strategy_returns(
    daily_close: pd.DataFrame,
    fixed_dates: dict[str, list[str]],
) -> pd.DataFrame:
    """
    For each trade date (keys in fixed_dates):
    - look back 12 months, optimize weights
    - hold until month-end, compute daily log returns
    """
    returns_df = np.log(daily_close).diff()
    portfolio_returns: list[pd.Series] = []

    for start_str, cols in fixed_dates.items():
        try:
            start_date = pd.to_datetime(start_str)
            end_date = start_date + pd.offsets.MonthEnd(0)

            opt_start = start_date - pd.DateOffset(months=12)
            opt_end = start_date - pd.DateOffset(days=1)

            optimization_df = (
                daily_close
                .loc[opt_start:opt_end, cols]
                .dropna(axis=1, how="any")
            )

            if optimization_df.shape[1] == 0 or len(optimization_df) < OPT_FREQ_DAYS:
                # skip if no data or too short
                continue

            lb = round(1 / (2 * optimization_df.shape[1]), 3)

            try:
                weight_dict = optimize_weights(optimization_df, lower_bound=lb)
                weights = (
                    pd.Series(weight_dict)
                    .reindex(optimization_df.columns)
                    .fillna(0.0)
                )
            except Exception:
                # fallback: equal-weight
                n = optimization_df.shape[1]
                weights = pd.Series(1 / n, index=optimization_df.columns)

            forward_rets = (
                returns_df
                .loc[start_date:end_date, weights.index]
                .dropna(how="all")
            )
            if forward_rets.empty:
                continue

            strat_ret = (forward_rets * weights).sum(axis=1)
            strat_ret.name = "Strategy Return"
            portfolio_returns.append(strat_ret)

        except Exception:
            # in production you'd log this, not swallow
            continue

    if not portfolio_returns:
        raise ValueError("No portfolio returns generated")

    portfolio_df = pd.concat(portfolio_returns).to_frame()
    portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep="first")]
    return portfolio_df
