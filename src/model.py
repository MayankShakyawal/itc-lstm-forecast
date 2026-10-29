"""Stacked LSTM model definition (Adam optimizer, MSE loss)."""
from tensorflow import keras
from tensorflow.keras import layers

from config import Config


def build_model(cfg: Config, n_features: int, units=None, dropout=None,
                dense_units=None, learning_rate=None) -> keras.Model:
    units = units or cfg.lstm_units
    dropout = cfg.dropout if dropout is None else dropout
    dense_units = dense_units or cfg.dense_units
    lr = learning_rate or cfg.learning_rate

    model = keras.Sequential(name="stacked_lstm")
    model.add(layers.Input(shape=(cfg.window, n_features)))
    for i, u in enumerate(units):
        last = i == len(units) - 1
        model.add(layers.LSTM(u, return_sequences=not last, name=f"lstm_{i + 1}"))
        model.add(layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense"))
    model.add(layers.Dense(1, name="close_forecast"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
