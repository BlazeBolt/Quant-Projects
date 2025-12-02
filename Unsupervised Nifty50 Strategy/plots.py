import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np


def plot_strategy_vs_benchmark(portfolio_df: pd.DataFrame,
                               strat_col: str = "Strategy Return",
                               bench_col: str = "NIFTY50 Buy&Hold",
                               end: str | None = None) -> None:
    if end:
        pf = portfolio_df.loc[:end].copy()
    else:
        pf = portfolio_df.copy()

    cum = np.exp(pf[[strat_col, bench_col]].cumsum()) - 1

    plt.style.use("ggplot")
    ax = cum.plot(figsize=(14, 6), linewidth=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.title("Unsupervised NIFTY Strategy vs Benchmark")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
