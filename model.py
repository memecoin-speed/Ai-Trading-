"""Causal features and strict time-ordered strategy validation."""
from __future__ import annotations

import math
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
    # At close(t), next executable price is open(t+1). The label is open(t+1)->open(t+2).
    frame["target"] = (frame.open.shift(-2) > frame.open.shift(-1)).astype(int)
    frame.loc[frame.index[-2:], "target"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


def fit_model(training: pd.DataFrame):
    training = training.dropna(subset=FEATURES + ["target"])
    if len(training) < 100:
        raise ValueError("Zu wenige Trainingsdaten")
    y = training.target.astype(int)
    if y.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=int(y.iloc[0]))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1, class_weight="balanced"))
    model.fit(training[FEATURES], y)
    return model


def probability(model, row: pd.DataFrame) -> float:
    if row[FEATURES].isna().any().any():
        raise ValueError("Ungültige Merkmale")
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    probs = model.predict_proba(row[FEATURES])[0]
    return float(probs[list(classes).index(1)]) if 1 in classes else 0.0


def _simulate(frame, start_idx, end_idx, fee, slip, buy, sell, start, trade_eur,
              max_position, stop, take, trailing=0.0):
    cash, units, entry, peak_price = start, 0.0, 0.0, 0.0
    peak_equity, max_dd = start, 0.0
    entries = exits = 0
    round_pnls, probs, outcomes = [], [], []
    open_cost = 0.0
    for i in range(start_idx, end_idx):
        # i-2 is the newest label that is knowable before open(i).
        model = fit_model(frame.iloc[:i - 2])
        p = probability(model, frame.iloc[[i - 1]])
        price = float(frame.open.iloc[i])
        signal_close = float(frame.close.iloc[i - 1])
        target = frame.target.iloc[i - 1]
        if not pd.isna(target):
            probs.append(p); outcomes.append(int(target))
        if units > 0:
            peak_price = max(peak_price, signal_close)
            trailing_hit = trailing > 0 and signal_close <= peak_price * (1 - trailing)
            if p <= sell or signal_close <= entry * (1 - stop) or signal_close >= entry * (1 + take) or trailing_hit:
                proceeds = units * price * (1 - slip) * (1 - fee)
                cash += proceeds; round_pnls.append(proceeds - open_cost)
                units = entry = peak_price = open_cost = 0.0; exits += 1
        elif p >= buy:
            # Confidence sizing stays bounded; weak qualifying signals use less capital.
            strength = min(1.5, max(0.5, 0.5 + (p - buy) / max(1e-6, 1 - buy)))
            spend = min(trade_eur * strength, max_position, cash * .95)
            if spend > 0:
                entry = price * (1 + slip); peak_price = entry
                units = spend / (entry * (1 + fee)); cash -= spend; open_cost = spend; entries += 1
        equity = cash + units * price
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, 1 - equity / peak_equity)
    last = float(frame.open.iloc[end_idx - 1])
    end = cash + units * last
    wins = sum(x > 0 for x in round_pnls); gross_profit = sum(x for x in round_pnls if x > 0)
    gross_loss = -sum(x for x in round_pnls if x < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    closed = len(round_pnls)
    brier = float(np.mean([(p-y)**2 for p, y in zip(probs, outcomes)])) if probs else float("nan")
    accuracy = float(np.mean([(p >= .5) == bool(y) for p, y in zip(probs, outcomes)])) * 100 if probs else 0.0
    benchmark = float(frame.open.iloc[end_idx - 1] / frame.open.iloc[start_idx])
    return {
        "return_pct": (end/start-1)*100, "buy_hold_pct": (benchmark-1)*100,
        "excess_pct": ((end/start-1)-(benchmark-1))*100, "max_drawdown_pct": max_dd*100,
        "entries": entries, "exits": exits, "closed_trades": closed,
        "win_rate_pct": wins/closed*100 if closed else 0.0, "profit_factor": pf,
        "test_bars": end_idx-start_idx, "open_position": units > 0,
        "brier": brier, "direction_accuracy_pct": accuracy,
    }


def evaluate(frame: pd.DataFrame, fee: float, slip: float, buy: float, sell: float,
             start: float = 1000.0, trade_eur: float = 25.0, max_position: float = 50.0,
             stop: float = .04, take: float = .08, trailing: float = .03) -> dict:
    cut = int(len(frame) * .7)
    r = _simulate(frame, cut, len(frame), fee, slip, buy, sell, start, trade_eur,
                  max_position, stop, take, trailing)
    for k in ("return_pct", "buy_hold_pct", "excess_pct", "max_drawdown_pct", "win_rate_pct", "direction_accuracy_pct"):
        r[k] = round(r[k], 2)
    if math.isfinite(r["brier"]): r["brier"] = round(r["brier"], 4)
    return r


def nested_validate(frame: pd.DataFrame, fee: float, slip: float, start: float = 1000.0,
                    trade_eur: float = 25.0, max_position: float = 50.0,
                    stop: float = .04, take: float = .08, trailing: float = .03) -> dict:
    """Tune only on middle 15%; report final untouched last 30%."""
    n = len(frame); tune_start, test_start = int(n*.55), int(n*.70)
    candidates = [(b, s) for b in (.56, .58, .60, .62) for s in (.42, .45, .48) if s < b]
    scored = []
    for buy, sell in candidates:
        r = _simulate(frame, tune_start, test_start, fee, slip, buy, sell, start, trade_eur,
                      max_position, stop, take, trailing)
        # Conservative tuning objective: excess return penalized for drawdown and tiny samples.
        sample_penalty = 2.0 if r["closed_trades"] < 2 else 0.0
        score = r["excess_pct"] - .5*r["max_drawdown_pct"] - sample_penalty
        scored.append((score, buy, sell))
    _, buy, sell = max(scored, key=lambda x: x[0])
    result = _simulate(frame, test_start, n, fee, slip, buy, sell, start, trade_eur,
                       max_position, stop, take, trailing)
    result.update({"selected_buy": buy, "selected_sell": sell, "tuning_bars": test_start-tune_start})
    for k in ("return_pct", "buy_hold_pct", "excess_pct", "max_drawdown_pct", "win_rate_pct", "direction_accuracy_pct"):
        result[k] = round(result[k], 2)
    if math.isfinite(result["brier"]): result["brier"] = round(result["brier"], 4)
    return result
