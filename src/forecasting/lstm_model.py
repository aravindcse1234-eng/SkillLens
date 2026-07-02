import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from src.utils.helpers import get_device
from src.utils.metrics import compute_forecast_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMForecaster:
    def __init__(self, seq_length: int = 12, n_features: int = 1,
                 lstm_units: List[int] = None, dropout: float = 0.2,
                 learning_rate: float = 0.001, epochs: int = 100):
        self.seq_length = seq_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.scaler = None
        self.device = get_device()

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential()
        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=(self.seq_length, self.n_features)))
            model.add(Dropout(self.dropout))

        model.add(Dense(self.n_features))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss="mse", metrics=["mae"])
        self.model = model
        logger.info(f"LSTM model built with {self.lstm_units} units")

    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)

    def fit(self, data: pd.Series, validation_split: float = 0.1,
            batch_size: int = 32, early_stopping_patience: int = 10) -> Dict:
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = self._prepare_sequences(scaled_data)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if self.model is None:
            self._build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        logger.info("LSTM model trained")
        return history.history

    def predict(self, data: pd.Series, steps: int = 12) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted yet")

        scaled_data = self.scaler.transform(data.values[-self.seq_length:].reshape(-1, 1))
        current_seq = scaled_data.reshape(1, self.seq_length, self.n_features)

        predictions = []
        for _ in range(steps):
            pred = self.model.predict(current_seq, verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred[0, 0]

        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        return predictions

    def forecast_skill(self, skill_data: pd.Series, steps: int = 12) -> Tuple[np.ndarray, Dict]:
        history = self.fit(skill_data)
        predictions = self.predict(skill_data, steps)

        train_preds = self.model.predict(
            self._prepare_sequences(
                self.scaler.transform(skill_data.values.reshape(-1, 1))
            )[0], verbose=0
        )
        train_preds_inv = self.scaler.inverse_transform(train_preds).flatten()
        actual = skill_data.values[self.seq_length:]
        metrics = compute_forecast_metrics(actual[:len(train_preds_inv)], train_preds_inv)
        return predictions, metrics

    def save(self, path: str):
        if self.model:
            self.model.save(path)
            import joblib
            joblib.dump(self.scaler, path.replace(".h5", "_scaler.pkl"))
            logger.info(f"LSTM model saved to {path}")

    def load(self, path: str):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        import joblib
        self.scaler = joblib.load(path.replace(".h5", "_scaler.pkl"))
        logger.info(f"LSTM model loaded from {path}")
        return self
