"""Train the stacked LSTM, evaluate on the held-out period, save plots and metrics."""
import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from config import Config, MODEL_DIR, OUTPUT_DIR
from data_loader import inverse_target, load_prices, prepare_data, synthetic_prices
from metrics import mae, mape, rmse
from model import build_model


def load_best_params(cfg: Config):
    p = MODEL_DIR / "best_params.json"
    if p.exists():
        best = json.loads(p.read_text())
        print(f"Using tuned hyperparameters from {p}")
        return best
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--synthetic", action="store_true", help="offline demo on random-walk data")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    keras.utils.set_random_seed(cfg.seed)

    df = synthetic_prices() if args.synthetic else load_prices(cfg, refresh=args.refresh)
    print(f"Loaded {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}")
    data = prepare_data(df, cfg)
    print(f"Train windows: {data['X_train'].shape}  Test windows: {data['X_test'].shape}")

    best = load_best_params(cfg)
    model = build_model(
        cfg, data["n_features"],
        units=tuple(best["units"]) if "units" in best else None,
        dropout=best.get("dropout"),
        learning_rate=best.get("learning_rate"),
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "lstm_best.keras"),
                                        save_best_only=True),
    ]
    hist = model.fit(
        data["X_train"], data["y_train"],
        validation_split=cfg.val_ratio, shuffle=False,
        epochs=cfg.epochs, batch_size=best.get("batch_size", cfg.batch_size),
        callbacks=callbacks, verbose=2,
    )

    # --- evaluation on held-out period, in rupee terms ---
    pred_scaled = model.predict(data["X_test"], verbose=0)
    y_pred = inverse_target(data["scaler"], pred_scaled, data["target_idx"])
    y_true = inverse_target(data["scaler"], data["y_test"], data["target_idx"])

    results = {"MAPE_pct": mape(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
               "MAE": mae(y_true, y_pred), "test_samples": int(len(y_true))}
    print("\nHeld-out results:", json.dumps(results, indent=2))
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler": data["scaler"], "target_idx": data["target_idx"],
                     "features": list(df.columns), "window": cfg.window}, f)

    pd.DataFrame({"Date": data["test_dates"], "Actual": y_true, "Predicted": y_pred}
                 ).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # --- plots ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(data["test_dates"], y_true, label="Actual", lw=1.4)
    ax.plot(data["test_dates"], y_pred, label="Predicted", lw=1.2)
    ax.set_title(f"ITC closing price: held-out forecast (MAPE {results['MAPE_pct']:.2f}%)")
    ax.set_ylabel("Price (INR)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "forecast.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.history["loss"], label="train MSE")
    ax.plot(hist.history["val_loss"], label="val MSE")
    ax.set_xlabel("Epoch"); ax.set_title("Training curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150); plt.close(fig)
    print(f"Artifacts saved in {OUTPUT_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
