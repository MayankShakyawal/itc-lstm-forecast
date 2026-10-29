"""Forecast the next trading day's ITC close from the latest 50 days of data."""
import pickle

import numpy as np
from tensorflow import keras

from config import Config, MODEL_DIR
from data_loader import inverse_target, load_prices


def main():
    cfg = Config()
    df = load_prices(cfg, refresh=True)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)
    model = keras.models.load_model(MODEL_DIR / "lstm_best.keras")

    last = df[meta["features"]].tail(meta["window"])
    x = meta["scaler"].transform(last.values)[None, ...].astype("float32")
    pred = inverse_target(meta["scaler"], model.predict(x, verbose=0), meta["target_idx"])[0]
    print(f"Last close ({last.index[-1].date()}): {last['Close'].iloc[-1]:.2f}")
    print(f"Forecast for next trading day: {pred:.2f}")


if __name__ == "__main__":
    main()
