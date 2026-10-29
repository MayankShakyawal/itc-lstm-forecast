"""Iterative hyperparameter search that minimises validation MSE.

Random search over units / dropout / learning rate / batch size. The best
configuration is written to models/best_params.json and picked up by train.py.
"""
import argparse
import json
import random

import numpy as np
from tensorflow import keras

from config import Config, MODEL_DIR
from data_loader import load_prices, prepare_data, synthetic_prices
from model import build_model

SEARCH_SPACE = {
    "units": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-2, 5e-3, 1e-3, 5e-4],
    "batch_size": [16, 32, 64],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    random.seed(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)

    df = synthetic_prices() if args.synthetic else load_prices(cfg)
    data = prepare_data(df, cfg)

    best = {"val_loss": float("inf")}
    for t in range(1, args.trials + 1):
        params = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        model = build_model(cfg, data["n_features"], units=params["units"],
                            dropout=params["dropout"], learning_rate=params["learning_rate"])
        hist = model.fit(
            data["X_train"], data["y_train"],
            validation_split=cfg.val_ratio, shuffle=False,
            epochs=args.epochs, batch_size=params["batch_size"], verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )
        val = float(min(hist.history["val_loss"]))
        print(f"[trial {t}/{args.trials}] val_mse={val:.6f} params={params}")
        if val < best["val_loss"]:
            best = {"val_loss": val, **{k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in params.items()}}
        keras.backend.clear_session()

    out = MODEL_DIR / "best_params.json"
    out.write_text(json.dumps(best, indent=2))
    print(f"\nBest configuration saved to {out}\n{json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
