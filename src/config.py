"""Central configuration for the ITC stacked LSTM forecaster."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # Data
    ticker: str = "ITC.NS"          # ITC Ltd on NSE
    start: str = "2012-01-01"
    end: str = "2026-10-01"
    csv_path: str = str(DATA_DIR / "itc.csv")
    target: str = "Close"
    # Multivariate inputs; the target column must be included
    features: list = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])

    # Windowing
    window: int = 50                # 50-day look-back windows
    test_ratio: float = 0.2         # held-out chronological split
    val_ratio: float = 0.1          # fraction of the train part used for validation

    # Model
    lstm_units: tuple = (128, 64, 32)   # stacked LSTM layers
    dropout: float = 0.2
    dense_units: int = 16
    learning_rate: float = 1e-3     # Adam
    batch_size: int = 32
    epochs: int = 100
    patience: int = 12

    seed: int = 42
