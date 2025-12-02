import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import KMEANS_CLUSTERS, FACTOR_COLUMNS


def assign_clusters(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional KMeans clustering for each month.
    Uses technical features + factor betas.
    """
    data = data.drop(columns=["cluster"], errors="ignore").copy()
    tech_cols = [
        "garman_klass_vol",
        "rsi",
        "bb_low",
        "bb_mid",
        "bb_high",
        "atr",
        "macd",
    ]

    cluster_features = tech_cols + FACTOR_COLUMNS

    def get_clusters(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < KMEANS_CLUSTERS:
            df["cluster"] = np.nan
            return df
        X = df[cluster_features]
        km = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=0, n_init=10)
        df["cluster"] = km.fit_predict(X)
        return df

    data = (
        data.dropna()
        .groupby("date", group_keys=False)
        .apply(get_clusters)
    )
    return data


def build_cluster_signals(data: pd.DataFrame,
                          selected_cluster: int = 3) -> dict[str, list[str]]:
    """
    Build trade dates → tickers dict for selected cluster, shifted by +1 day.
    """
    df = data[data["cluster"] == selected_cluster].copy()

    df = df.reset_index(level="ticker")
    df.index = df.index + pd.DateOffset(1)  # signal → trade date
    df = df.reset_index().set_index(["date", "ticker"])

    fixed_dates: dict[str, list[str]] = {}
    for d in df.index.get_level_values("date").unique():
        tickers = df.xs(d, level="date").index.tolist()
        fixed_dates[d.strftime("%Y-%m-%d")] = tickers
    return fixed_dates
