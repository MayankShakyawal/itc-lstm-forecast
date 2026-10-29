# Stock Price Forecasting using Stacked LSTM (ITC)

Multivariate time-series forecasting of ITC closing prices with a stacked LSTM.

- Inputs: Open, High, Low, Close, Volume over a **50-day** window
- Model: 3 stacked LSTM layers + dropout + dense head, **Adam** optimizer, **MSE** loss
- Tuning: random search over units, dropout, learning rate and batch size (`src/tune.py`)
- Evaluation: **MAPE** on a chronological held-out split (last 20% of days), reported in INR terms

## Structure

```
itc-lstm-forecast/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py        # all settings in one place
│   ├── data_loader.py   # yfinance download, scaling, 50-day windows
│   ├── model.py         # stacked LSTM
│   ├── tune.py          # hyperparameter search
│   ├── train.py         # train + evaluate + plots
│   ├── predict.py       # next-day forecast
│   └── metrics.py       # MAPE / RMSE / MAE
├── data/                # itc.csv cache
├── models/              # lstm_best.keras, scaler.pkl, best_params.json
└── outputs/             # metrics.json, predictions.csv, forecast.png, loss_curve.png
```

## Run

```bash
pip install -r requirements.txt
cd src
python tune.py --trials 10      # optional, writes models/best_params.json
python train.py                 # trains, evaluates, saves plots
python predict.py               # next-day close forecast
```

No internet? Put an OHLCV CSV at `data/itc.csv` (columns `Date,Open,High,Low,Close,Volume`),
or smoke-test the pipeline with `python train.py --synthetic --epochs 3`.

## Notes

- The scaler is fit on the training segment only, so the held-out period does not leak.
- The split is chronological and never shuffled.
- The 1.92% MAPE on your resume is the figure from your own run. Re-running on newer
  data will give a different number, so update the bullet from `outputs/metrics.json`.
