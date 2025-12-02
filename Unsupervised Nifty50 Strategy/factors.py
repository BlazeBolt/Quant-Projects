import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pandas_datareader import data as web
from config import FACTOR_COLUMNS


def load_fama_french_5(start: str = "2010-01-01") -> pd.DataFrame:
    """Load Fama–French 5 factors (US) and convert to monthly decimal returns."""
    ff = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[
        0
    ].drop("RF", axis=1)
    ff.index = ff.index.to_timestamp()
    ff = ff.resample("M").last().div(100.0)
    ff.index.name = "date"
    return ff


def compute_portfolio_betas(data: pd.DataFrame,
                            ff: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight portfolio over NIFTY names.
    Compute rolling factor betas of portfolio returns.
    """
    portfolio_ret = (
        data["return_1m"]
        .groupby("date")
        .mean()
        .to_frame("return_1m")
    )

    factor_data = ff.join(portfolio_ret).dropna().sort_index()

    y = factor_data["return_1m"]
    X = sm.add_constant(factor_data.drop(columns=["return_1m"]))

    window = min(24, X.shape[0])
    rols = RollingOLS(
        endog=y,
        exog=X,
        window=window,
        min_nobs=X.shape[1] + 1,
    ).fit(params_only=True)

    betas = rols.params.drop(columns=["const"])
    return betas  # index=date, columns=factors


def attach_betas_to_panel(data: pd.DataFrame,
                          betas: pd.DataFrame) -> pd.DataFrame:
    """Attach portfolio-level betas to each stock at that date."""
    data = data.join(betas, on="date")
    data[FACTOR_COLUMNS] = data.groupby(level="ticker")[FACTOR_COLUMNS].transform(
        lambda x: x.fillna(x.mean())
    )
    data = data.dropna()
    return data