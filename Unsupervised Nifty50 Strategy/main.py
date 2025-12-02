import warnings
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE, BENCH_TICKER
from universe import get_nifty50_universe
from data_loader import download_panel, download_daily_close
from features import add_technical_features, build_monthly_panel, apply_liquidity_filter, add_lagged_returns
from factors import load_fama_french_5, compute_portfolio_betas, attach_betas_to_panel
from clustering import assign_clusters, build_cluster_signals
from backtest import build_strategy_returns
from metrics import annualized_return, annualized_vol, sharpe_ratio, max_drawdown
from plots import plot_strategy_vs_benchmark

warnings.filterwarnings("ignore")


def main():
    # 1. Universe
    nifty = get_nifty50_universe()
    symbols = nifty["YahooSymbol"].unique().tolist()

    # 2. Daily panel
    panel = download_panel(symbols, start=START_DATE, end=END_DATE)

    # 3. Technical features
    panel = add_technical_features(panel, price_col="close")

    # 4. Monthly panel
    data = build_monthly_panel(panel, price_col="close")
    data = apply_liquidity_filter(data)
    data = add_lagged_returns(data)

    # 5. Factor betas
    ff = load_fama_french_5(start=str(START_DATE))
    betas = compute_portfolio_betas(data, ff)
    data = attach_betas_to_panel(data, betas)

    # 6. Clustering & signals
    data = assign_clusters(data)
    fixed_dates = build_cluster_signals(data, selected_cluster=3)

    # 7. Backtest
    start_date = data.index.get_level_values("date").min() - pd.DateOffset(months=12)
    end_date = data.index.get_level_values("date").max()
    daily_close = download_daily_close(symbols, start=start_date, end=end_date)

    portfolio_df = build_strategy_returns(daily_close, fixed_dates)

    # 8. Benchmark (NIFTY 50 close)
    nifty_idx = yf.download(
        tickers=BENCH_TICKER,
        start=portfolio_df.index.min().date(),
        end=dt.date.today(),
    )
    nifty_close = nifty_idx["Close"].to_frame(BENCH_TICKER)
    nifty_ret = np.log(nifty_close).diff().dropna().rename(
        columns={BENCH_TICKER: "NIFTY50 Buy&Hold"}
    )

    portfolio_df = portfolio_df.merge(nifty_ret, left_index=True, right_index=True)

    # 9. Metrics
    strat = portfolio_df["Strategy Return"]
    bench = portfolio_df["NIFTY50 Buy&Hold"]

    print("=== Strategy Performance ===")
    print("Annualized return:", f"{annualized_return(strat):.2%}")
    print("Annualized vol   :", f"{annualized_vol(strat):.2%}")
    print("Sharpe (rf=0)    :", f"{sharpe_ratio(strat):.2f}")
    print("Max drawdown     :", f"{max_drawdown(strat):.2%}")

    print("\n=== Benchmark Performance (NIFTY50 Buy&Hold) ===")
    print("Annualized return:", f"{annualized_return(bench):.2%}")
    print("Annualized vol   :", f"{annualized_vol(bench):.2%}")
    print("Sharpe (rf=0)    :", f"{sharpe_ratio(bench):.2f}")
    print("Max drawdown     :", f"{max_drawdown(bench):.2%}")

    # 10. Plot
    plot_strategy_vs_benchmark(portfolio_df, end="2023-09-29")


if __name__ == "__main__":
    main()
