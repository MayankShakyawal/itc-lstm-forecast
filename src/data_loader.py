"""Download / load ITC prices and turn them into 50-day multivariate windows."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import Config


def load_prices(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Load OHLCV data from the local CSV, downloading it via yfinance if needed."""
    path = Path(cfg.csv_path)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end,
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"Could not download {cfg.ticker}. Put an OHLCV CSV at {path} "
                "(columns: Date,Open,High,Low,Close,Volume) or run with --synthetic."
            )
        if isinstance(df.columns, pd.MultiIndex):   # newer yfinance versions
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df.to_csv(path)
    df = df.sort_index()
    return df[cfg.features].dropna()


def synthetic_prices(n: int = 2500, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk with OHLCV columns; only for smoke tests / offline demos."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.014, n)
    close = 200 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(2_000_000, 12_000_000, n).astype(float)
    idx = pd.bdate_range("2014-01-01", periods=n, name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def make_windows(values: np.ndarray, target_idx: int, window: int):
    """values: (T, F) scaled array -> X (N, window, F), y (N,) next-day target."""
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i, target_idx])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def prepare_data(df: pd.DataFrame, cfg: Config):
    """Chronological split, scaler fit on train only (no look-ahead leakage)."""
    n = len(df)
    split = int(n * (1 - cfg.test_ratio))
    train_df, test_df = df.iloc[:split], df.iloc[split - cfg.window:]  # keep context for first test window

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    t_idx = list(df.columns).index(cfg.target)
    X_train, y_train = make_windows(train_scaled, t_idx, cfg.window)
    X_test, y_test = make_windows(test_scaled, t_idx, cfg.window)
    test_dates = test_df.index[cfg.window:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "target_idx": t_idx,
        "test_dates": test_dates, "n_features": df.shape[1],
    }


def inverse_target(scaler: MinMaxScaler, scaled: np.ndarray, target_idx: int) -> np.ndarray:
    """Undo MinMax scaling for the target column only."""
    lo, hi = scaler.data_min_[target_idx], scaler.data_max_[target_idx]
    return np.asarray(scaled).reshape(-1) * (hi - lo) + lo
