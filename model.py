"""Causal features and a time-ordered, out-of-sample model evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret1", "ret3", "ret12", "ma6", "ma24", "vol12", "range", "body", "volume_ratio", "rsi14"]


def feature_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(frame) < 180:
        raise ValueError("Zu wenige abgeschlossene Kerzen (mindestens 180 benötigt)")
    frame = frame.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    if len(frame) < 180 or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Unvollständige oder ungültige Marktdaten")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Ungültige Preise oder Volumina")
    close = frame.close.astype(float)
    ret = close.pct_change()
    frame["ret1"] = ret
    frame["ret3"] = close.pct_change(3)
    frame["ret12"] = close.pct_change(12)
    frame["ma6"] = close / close.rolling(6).mean() - 1
    frame["ma24"] = close / close.rolling(24).mean() - 1
    frame["vol12"] = ret.rolling(12).std()
    frame["range"] = (frame.high - frame.low) / close
    frame["body"] = (close - frame.open) / frame.open
    avg_volume = frame.volume.rolling(24).mean()
    frame["volume_ratio"] = (frame.volume / avg_volume.replace(0, np.nan) - 1).fillna(0)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    frame["rsi14"] = 100 - 100 / (1 + up / down.replace(0, 1e-12))
    frame.loc[(up == 0) & (down == 0), "rsi14"] = 50
    # At close(t) the next executable price is open(t+1). Label is its
    # return to open(t+2); last two feature rows have no known label.
    frame["target"] = (frame.open.shift(-2) > frame.open.shift(-1)).astype(int)
    frame.loc[frame.index[-2:], "target"] = np.nan
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def fit_model(training: pd.DataFrame):
    training = training.dropna(subset=FEATURES + ["target"])
    if len(training) < 100:
        raise ValueError("Zu wenige Trainingsdaten")
    y = training.target.astype(int)
    if y.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=int(y.iloc[0]))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))
    model.fit(training[FEATURES], y)
    return model


def probability(model, row: pd.DataFrame) -> float:
    if row[FEATURES].isna().any().any():
        raise ValueError("Ungültige Merkmale")
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    probs = model.predict_proba(row[FEATURES])[0]
    return float(probs[list(classes).index(1)]) if 1 in classes else 0.0


def evaluate(frame: pd.DataFrame, fee: float, slip: float, buy: float, sell: float,
             start: float = 1000.0, trade_eur: float = 25.0,
             max_position: float = 50.0, stop: float = .04, take: float = .08) -> dict:
    """Chronological holdout; same position sizing and exit rules as paper mode."""
    cut = int(len(frame) * 0.7)
    cash, units, entry, trades, peak, max_dd = start, 0.0, 0.0, 0, start, 0.0
    benchmark = float(frame.open.iloc[-1] / frame.open.iloc[cut])
    for i in range(cut, len(frame)):
        # Match the running bot: refit after each completed bar, excluding
        # the two newest training rows whose forward labels are not yet known.
        model = fit_model(frame.iloc[:i - 2])
        signal_row = frame.iloc[[i - 1]]
        p = probability(model, signal_row)
        price = float(frame.open.iloc[i])
        signal_close = float(frame.close.iloc[i - 1])
        if units > 0 and (p <= sell or signal_close <= entry * (1 - stop)
                          or signal_close >= entry * (1 + take)):
            cash += units * price * (1 - slip) * (1 - fee)
            units = 0.0
            entry = 0.0
            trades += 1
        elif units == 0 and p >= buy:
            spend = min(trade_eur, max_position, cash * .95)
            if spend > 0:
                entry = price * (1 + slip)
                units = spend / (entry * (1 + fee))
                cash -= spend
                trades += 1
        equity = cash + units * price
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    end = cash + units * float(frame.open.iloc[-1])
    return {"return_pct": round((end / start - 1) * 100, 2),
            "buy_hold_pct": round((benchmark - 1) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2), "trades": trades,
            "test_bars": len(frame) - cut}
