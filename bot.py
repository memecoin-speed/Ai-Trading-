"""Kraken spot AI bot: paper by default, explicit live opt-in."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from dotenv import load_dotenv

from model import evaluate, feature_frame, fit_model, nested_validate, probability

LOG = logging.getLogger("trader")


def safe_error(error: Exception) -> str:
    message = str(error)
    for secret in (os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("KRAKEN_API_KEY", ""),
                   os.getenv("KRAKEN_API_SECRET", "")):
        if secret:
            message = message.replace(secret, "[geschützt]")
    return message


def setting_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    mode: str
    symbol: str
    timeframe: str
    paper_start: float
    trade_eur: float
    max_position: float
    buy: float
    sell: float
    stop: float
    take: float
    fee: float
    slip: float
    data_dir: Path
    telegram_token: str
    chat_id: str
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    trailing_stop_pct: float = 0.03

    @classmethod
    def load(cls, allow_unconfigured_live=False):
        load_dotenv()
        s = cls(os.getenv("MODE", "paper").lower(), os.getenv("SYMBOL", "BTC/EUR"),
                os.getenv("TIMEFRAME", "1h"), setting_float("PAPER_START_EUR", "1000"),
                setting_float("TRADE_EUR", "25"), setting_float("MAX_POSITION_EUR", "50"),
                setting_float("BUY_PROBABILITY", "0.58"), setting_float("SELL_PROBABILITY", "0.45"),
                setting_float("STOP_LOSS_PCT", "0.04"), setting_float("TAKE_PROFIT_PCT", "0.08"),
                setting_float("FEE_RATE", "0.004"), setting_float("SLIPPAGE_RATE", "0.001"),
                Path(os.getenv("DATA_DIR", "./data")), os.getenv("TELEGRAM_BOT_TOKEN", ""),
                os.getenv("TELEGRAM_CHAT_ID", ""), setting_float("MAX_DAILY_LOSS_PCT", "0.03"),
                setting_float("MAX_DRAWDOWN_PCT", "0.10"), setting_float("TRAILING_STOP_PCT", "0.03"))
        if s.mode not in {"paper", "live"} or s.symbol not in {"BTC/EUR", "ETH/EUR"}:
            raise ValueError("MODE muss paper/live sein; SYMBOL muss BTC/EUR oder ETH/EUR sein")
        if s.timeframe not in {"1h", "4h"} or s.paper_start <= 0 or s.trade_eur <= 0:
            raise ValueError("Ungültiger Zeitrahmen oder Einsatz")
        if not all(math.isfinite(x) for x in (s.paper_start, s.trade_eur, s.max_position,
                                               s.buy, s.sell, s.stop, s.take, s.fee, s.slip, s.max_daily_loss_pct, s.max_drawdown_pct, s.trailing_stop_pct)):
            raise ValueError("Alle numerischen Einstellungen müssen endlich sein")
        if not (0.5 <= s.buy <= 0.9 and 0.1 <= s.sell < s.buy and
                0 < s.stop <= 0.25 and 0 < s.take <= 0.5 and
                0 <= s.fee <= 0.03 and 0 <= s.slip <= 0.03 and s.max_position >= s.trade_eur and
                0.001 <= s.max_daily_loss_pct <= 0.25 and 0.01 <= s.max_drawdown_pct <= 0.5 and
                0 <= s.trailing_stop_pct <= 0.25):
            raise ValueError("Ungültige Grenzwerte oder Positionsgröße")
        if s.mode == "live" and not allow_unconfigured_live and (os.getenv("ENABLE_LIVE_TRADING") != "YES" or
                                  not os.getenv("KRAKEN_API_KEY") or not os.getenv("KRAKEN_API_SECRET")):
            raise ValueError("Live benötigt ENABLE_LIVE_TRADING=YES und Kraken-API-Schlüssel")
        return s


class Store:
    def __init__(self, path: Path, paper_start: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, time TEXT, side TEXT, qty REAL, price REAL, fee REAL, mode TEXT, order_id TEXT)")
        self.db.commit()
        if self.get("cash") is None:
            self.put(cash=paper_start, units=0.0, entry=0.0, last_candle=0, paused=False, pending=False)

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, **fields):
        with self.db:
            for key, value in fields.items():
                self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def record_fill(self, fields: dict, side, qty, price, fee, mode, order_id):
        """Position, candle cursor, and trade log commit together."""
        with self.db:
            for key, value in fields.items():
                self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))
            self.db.execute("INSERT INTO trades(time,side,qty,price,fee,mode,order_id) VALUES (?,?,?,?,?,?,?)",
                            (datetime.now(timezone.utc).isoformat(), side, qty, price, fee, mode, order_id))


class Telegram:
    def __init__(self, token: str, chat_id: str, store: Store | None = None):
        self.token, self.chat_id = token, chat_id
        self.store = store
        self.offset = store.get("telegram_offset") if store else None

    def request(self, method, params):
        data = urllib.parse.urlencode(params).encode()
        request = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/{method}", data=data)
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Telegram returns useful JSON such as 401 Unauthorized, 400 chat not found,
            # or 409 Conflict. Log only the status/description, never the bot token/URL.
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                description = str(payload.get("description") or "HTTP error")[:300]
                error_code = payload.get("error_code", exc.code)
            except Exception:
                error_code = exc.code
                description = str(exc.reason or "HTTP error")[:300]
            raise RuntimeError(f"Telegram API HTTP {error_code}: {description}") from None
        if not result.get("ok"):
            code = result.get("error_code", "?")
            description = str(result.get("description") or "API error")[:300]
            raise RuntimeError(f"Telegram API {code}: {description}")
        return result["result"]

    def send(self, message):
        if self.token and self.chat_id:
            try:
                self.request("sendMessage", {"chat_id": self.chat_id, "text": message[:3900]})
            except Exception as exc:  # noqa: BLE001 - notification failure must not stop trading loop
                LOG.warning("Telegram-Nachricht fehlgeschlagen: %s", exc)
        print(message, flush=True)

    def updates(self):
        if not self.token or not self.chat_id:
            return []
        if self.offset is None:
            # Discard old queued commands on first run, then persist the cursor.
            recent = self.request("getUpdates", {"offset": -1, "timeout": 1, "limit": 1})
            self.offset = (recent[-1]["update_id"] + 1) if recent else 0
            if self.store:
                self.store.put(telegram_offset=self.offset)
            return []
        result = self.request("getUpdates", {"offset": self.offset or 0, "timeout": 1, "limit": 20})
        updates = []
        for event in result:
            msg = event.get("message") or {}
            chat = msg.get("chat") or {}
            if chat.get("type") == "private" and str(chat.get("id")) == str(self.chat_id):
                command = (msg.get("text") or "").split("@")[0].strip().split(" ")[0]
            else:
                command = None
            updates.append((event["update_id"], command))
        return updates

    def ack(self, update_id: int):
        # Commands such as /pause are idempotent. Record the cursor only after
        # handling each command so a crash cannot silently discard it.
        self.offset = update_id + 1
        if self.store:
            self.store.put(telegram_offset=self.offset)


class Trader:
    def __init__(self, settings: Settings):
        self.s = settings
        state_path = settings.data_dir / f"{settings.mode}_{settings.symbol.replace('/', '_')}.sqlite3"
        if settings.mode == "live" and not state_path.is_file():
            raise RuntimeError("Live-Zustand fehlt: erst Kontostand/Orders prüfen, dann 'python bot.py init-live' ausführen")
        self.exchange = ccxt.kraken({"enableRateLimit": True, "timeout": 20000,
                                      "apiKey": os.getenv("KRAKEN_API_KEY", "") if settings.mode == "live" else "",
                                      "secret": os.getenv("KRAKEN_API_SECRET", "") if settings.mode == "live" else ""})
        self.exchange.load_markets()
        market = self.exchange.market(settings.symbol)
        if not market.get("spot") or market.get("active") is False or market["quote"] != "EUR":
            raise ValueError("Der Kraken-Spotmarkt ist nicht verfügbar")
        self.market = market
        self.store = Store(state_path, settings.paper_start)
        if settings.mode == "live" and self.store.get("live_initialized") is not True:
            raise RuntimeError("Live-Zustand nicht initialisiert; Echtgeldhandel gesperrt")
        self.telegram = Telegram(settings.telegram_token, settings.chat_id, self.store)

    def candles(self):
        raw = self.exchange.fetch_ohlcv(self.s.symbol, self.s.timeframe, limit=720)
        # Kraken includes the forming candle. The current bar is never used.
        frame = [[int(r[0]), *map(float, r[1:6])] for r in raw[:-1]]
        if len(frame) < 180:
            raise ValueError("Zu wenige Marktdaten")
        age_ms = time.time() * 1000 - frame[-1][0]
        if age_ms < 0 or age_ms > (7200 if self.s.timeframe == "1h" else 28800) * 1000:
            raise ValueError("Marktdaten sind veraltet")
        return feature_frame(frame)

    def analysis(self):
        frame = self.candles()
        model = fit_model(frame.iloc[:-2])
        p = probability(model, frame.iloc[[-1]])
        return frame, p

    def report(self, frame, p):
        return (f"{self.s.symbol} | {self.s.timeframe} | {self.s.mode.upper()}\n"
                f"Letzter Schluss: {frame.close.iloc[-1]:.2f} EUR\n"
                f"Modellscore: {p:.1%} für positive nächste Kerze (nicht kalibriert)\n"
                f"Kaufen ab {self.s.buy:.0%}; verkaufen bis {self.s.sell:.0%}\n"
                f"Offene Bot-Position: {self.store.get('units', 0):.8f} {self.market['base']}\n"
                f"Pause: {self.store.get('paused', False)} | Ungeklärte Order: {self.store.get('pending', False)}")

    def backtest(self, frame=None):
        frame = frame if frame is not None else self.candles()
        result = evaluate(frame, self.s.fee, self.s.slip, self.s.buy, self.s.sell,
                          self.s.paper_start, self.s.trade_eur, self.s.max_position,
                          self.s.stop, self.s.take, self.s.trailing_stop_pct)
        pf = result["profit_factor"]
        pf_text = "∞" if pf == float("inf") else f"{pf:.2f}"
        return (f"WALK-FORWARD BACKTEST | {self.s.symbol} | {self.s.timeframe}\n"
                f"Out-of-sample: {result['test_bars']} Kerzen (letzte 30 %)\n"
                f"Strategie nach Kosten: {result['return_pct']:+.2f}% | Buy & Hold: {result['buy_hold_pct']:+.2f}%\n"
                f"Differenz zu Buy & Hold: {result['excess_pct']:+.2f} Prozentpunkte\n"
                f"Max Drawdown: {result['max_drawdown_pct']:.2f}%\n"
                f"Geschlossene Trades: {result['closed_trades']} | Trefferquote: {result['win_rate_pct']:.1f}% | Profit-Factor: {pf_text}\n"
                f"Entries/Exits: {result['entries']}/{result['exits']} | Position am Testende: {'offen' if result['open_position'] else 'keine'}\n"
                "Walk-forward: pro Kerze nur zu diesem Zeitpunkt bekannte Daten; Gebühren und Slippage berücksichtigt. Keine Gewinnprognose.")

    def validation_report(self, symbol=None, timeframe=None):
        symbol = symbol or self.s.symbol
        timeframe = timeframe or self.s.timeframe
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=720)
        frame = feature_frame([[int(r[0]), *map(float, r[1:6])] for r in raw[:-1]])
        r = nested_validate(frame, self.s.fee, self.s.slip, self.s.paper_start,
                            self.s.trade_eur, self.s.max_position, self.s.stop,
                            self.s.take, self.s.trailing_stop_pct)
        pf = r["profit_factor"]
        pf_text = "∞" if pf == float("inf") else f"{pf:.2f}"
        return (f"NESTED OOS | {symbol} | {timeframe}\n"
                f"Tuning: {r['tuning_bars']} Kerzen | final unangetastet: {r['test_bars']} Kerzen\n"
                f"Gewählte Schwellen nur aus Tuning: BUY {r['selected_buy']:.0%} / SELL {r['selected_sell']:.0%}\n"
                f"Strategie: {r['return_pct']:+.2f}% | Buy&Hold: {r['buy_hold_pct']:+.2f}% | Differenz: {r['excess_pct']:+.2f}pp\n"
                f"Drawdown: {r['max_drawdown_pct']:.2f}% | Trades: {r['closed_trades']} | Treffer: {r['win_rate_pct']:.1f}% | PF: {pf_text}\n"
                f"Richtungsgenauigkeit: {r['direction_accuracy_pct']:.1f}% | Brier: {r['brier']}")

    def robustness_report(self):
        lines = ["ROBUSTHEIT | Nested Out-of-Sample | keine Gewinnprognose"]
        for symbol, timeframe in (("BTC/EUR","1h"),("BTC/EUR","4h"),("ETH/EUR","1h"),("ETH/EUR","4h")):
            try:
                raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=720)
                frame = feature_frame([[int(r[0]), *map(float, r[1:6])] for r in raw[:-1]])
                r = nested_validate(frame, self.s.fee, self.s.slip, self.s.paper_start,
                                    self.s.trade_eur, self.s.max_position, self.s.stop,
                                    self.s.take, self.s.trailing_stop_pct)
                pf = r["profit_factor"]; pft = "∞" if pf == float("inf") else f"{pf:.2f}"
                lines.append(f"{symbol} {timeframe}: {r['return_pct']:+.2f}% | vs B&H {r['excess_pct']:+.2f}pp | DD {r['max_drawdown_pct']:.1f}% | PF {pft} | {r['closed_trades']} Trades")
            except Exception as exc:
                lines.append(f"{symbol} {timeframe}: Fehler {safe_error(exc)[:100]}")
        return "\n".join(lines)

    def equity(self, price=None):
        if self.s.mode != "paper":
            return None
        cash = float(self.store.get("cash", 0))
        units = float(self.store.get("units", 0))
        if price is None:
            try:
                price = float(self.exchange.fetch_ticker(self.s.symbol).get("last") or 0)
            except Exception:
                price = 0.0
        return cash + units * float(price or 0)

    def enforce_daily_loss_limit(self, price):
        if self.s.mode != "paper":
            return False
        today = datetime.now(timezone.utc).date().isoformat()
        equity = self.equity(price)
        day = self.store.get("risk_day")
        start = float(self.store.get("day_start_equity", equity) or equity)
        if day != today or start <= 0:
            self.store.put(risk_day=today, day_start_equity=equity)
            return False
        loss = max(0.0, 1 - equity / start)
        total_dd = max(0.0, 1 - equity / self.s.paper_start)
        if loss >= self.s.max_daily_loss_pct:
            self.store.put(paused=True)
            self.telegram.send(f"RISIKO-STOP: Tagesverlust {loss:.2%} erreicht (Limit {self.s.max_daily_loss_pct:.2%}). Bot pausiert.")
            return True
        if total_dd >= self.s.max_drawdown_pct:
            self.store.put(paused=True)
            self.telegram.send(f"DRAWDOWN-STOP: Equity {equity:.2f} EUR, Verlust seit Start {total_dd:.2%}. Bot pausiert.")
            return True
        return False

    def trade_stats(self):
        rows = self.store.db.execute("SELECT side,qty,price,fee FROM trades ORDER BY id").fetchall()
        buys = sells = 0
        realized = 0.0
        basis_qty = basis_cost = 0.0
        fees = 0.0
        for side, qty, price, fee in rows:
            qty, price, fee = float(qty), float(price), float(fee)
            fees += fee
            if side == "buy":
                buys += 1; basis_qty += qty; basis_cost += qty * price + fee
            elif side == "sell":
                sells += 1
                avg = basis_cost / basis_qty if basis_qty > 0 else 0
                sold_cost = avg * min(qty, basis_qty)
                realized += qty * price - fee - sold_cost
                if basis_qty > 0:
                    frac = min(1.0, qty / basis_qty); basis_qty *= (1-frac); basis_cost *= (1-frac)
        return buys, sells, realized, fees

    def watchlist_report(self):
        lines = [f"WATCHLIST | {self.s.timeframe} | {self.s.mode.upper()}"]
        for symbol in ("BTC/EUR", "ETH/EUR"):
            raw = self.exchange.fetch_ohlcv(symbol, self.s.timeframe, limit=720)
            frame = feature_frame([[int(r[0]), *map(float, r[1:6])] for r in raw[:-1]])
            model = fit_model(frame.iloc[:-2]); p = probability(model, frame.iloc[[-1]])
            signal = "BUY-Zone" if p >= self.s.buy else ("SELL-Zone" if p <= self.s.sell else "HOLD-Zone")
            lines.append(f"{symbol}: {frame.close.iloc[-1]:.2f} EUR | Score {p:.1%} | {signal}")
        lines.append(f"Automatische Ausführung nur für {self.s.symbol}; Watchlist ist Analyse.")
        return "\n".join(lines)

    def live_free(self, asset):
        try:
            bal = self.exchange.fetch_balance()
        except (ccxt.AuthenticationError, ccxt.PermissionDenied):
            self.store.put(paused=True)
            raise
        amount = float((bal.get("free") or {}).get(asset, 0) or 0)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Ungültiges Kraken-Guthaben; kein Trade ausgeführt")
        return amount

    def live_order(self, side, qty, quote_cost=None):
        # Persist BEFORE submission. Any uncertain outcome stops all new trades.
        self.store.put(pending=True)
        try:
            if side == "buy":
                order = self.exchange.create_market_buy_order_with_cost(
                    self.s.symbol, quote_cost, {"oflags": "fciq"})
            else:
                order = self.exchange.create_order(
                    self.s.symbol, "market", side, qty, None, {"oflags": "fciq"})
        except (ccxt.InvalidOrder, ccxt.InsufficientFunds, ccxt.AuthenticationError,
                ccxt.PermissionDenied):
            # An explicit rejection did not create an order. Pause for correction.
            self.store.put(pending=False, paused=True)
            raise
        order_id = order.get("id")
        if not order_id:
            raise RuntimeError("Keine Order-ID; Kraken prüfen, Bot bleibt gesperrt")
        self.store.put(pending=order_id)
        for _ in range(5):
            time.sleep(2)
            try:
                order = self.exchange.fetch_order(order_id, self.s.symbol)
            except (ccxt.OrderNotFound, ccxt.NetworkError):
                # The submission is already recorded. Query again, never resubmit.
                continue
            if order.get("status") == "closed":
                filled = float(order.get("filled") or 0)
                average = float(order.get("average") or 0)
                if not math.isfinite(filled) or not math.isfinite(average) or filled <= 0 or average <= 0:
                    raise RuntimeError("Füllung unklar; Bot bleibt gesperrt")
                fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
                base_fee = sum(float(f.get("cost") or 0) for f in fees if f.get("currency") == self.market["base"])
                # fciq explicitly requests fees in the quote currency. Some
                # parsed Kraken orders omit currency despite this flag.
                quote_fee = sum(float(f.get("cost") or 0) for f in fees
                                if f.get("currency") in (self.market["quote"], None))
                if not all(math.isfinite(x) and x >= 0 for x in (base_fee, quote_fee)):
                    raise RuntimeError("Gebühren unklar; Bot bleibt gesperrt")
                if base_fee > 0 or any(f.get("currency") not in (self.market["quote"], None)
                                       and float(f.get("cost") or 0) > 0 for f in fees):
                    raise RuntimeError("Gebührenwährung weicht von EUR ab; Bot bleibt gesperrt")
                return filled, average, str(order_id), base_fee, quote_fee
        raise RuntimeError("Order noch offen/unklar; manuell auf Kraken prüfen")

    def execute(self, frame, p):
        candle = int(frame.ts.iloc[-1])
        if candle <= int(self.store.get("last_candle", 0)):
            return None
        if self.store.get("pending"):
            raise RuntimeError("Ungeklärte Order: Handel angehalten, Kraken-Konto prüfen")
        if self.store.get("paused"):
            return None
        price = float(frame.close.iloc[-1])
        if self.enforce_daily_loss_limit(price):
            return None
        units = float(self.store.get("units", 0))
        entry = float(self.store.get("entry", 0))
        if not all(math.isfinite(x) and x >= 0 for x in (price, units, entry)) or price == 0:
            raise RuntimeError("Ungültige Positions- oder Marktdaten; kein Trade ausgeführt")
        side = None
        peak_price = float(self.store.get("peak_price", entry) or entry)
        if units > 0:
            peak_price = max(peak_price, price)
            self.store.put(peak_price=peak_price)
        trailing_hit = units > 0 and self.s.trailing_stop_pct > 0 and price <= peak_price * (1 - self.s.trailing_stop_pct)
        if units > 0 and (p <= self.s.sell or price <= entry * (1 - self.s.stop) or
                          price >= entry * (1 + self.s.take) or trailing_hit):
            side = "sell"
        elif units == 0 and p >= self.s.buy:
            side = "buy"
        if not side:
            self.store.put(last_candle=candle)
            return None
        if side == "buy":
            cash = self.live_free(self.market["quote"]) if self.s.mode == "live" else float(self.store.get("cash"))
            strength = min(1.5, max(0.5, 0.5 + (p - self.s.buy) / max(1e-6, 1 - self.s.buy)))
            spend = min(self.s.trade_eur * strength, self.s.max_position, cash * 0.95)
            # Reserve estimated fees inside the configured EUR spend.
            quote_cost = spend / (1 + self.s.fee)
            qty = float(self.exchange.amount_to_precision(self.s.symbol, quote_cost / (price * (1 + self.s.slip))))
            if qty <= 0 or not self.enough(qty, price):
                self.store.put(last_candle=candle)
                return f"Signal BUY, aber Guthaben oder Mindestorder reicht nicht ({spend:.2f} EUR)."
            if self.s.mode == "paper":
                fill = price * (1 + self.s.slip)
                fee = qty * fill * self.s.fee
                fields = {"cash": cash - qty * fill - fee, "units": qty, "entry": fill,
                          "peak_price": fill, "last_candle": candle}
                order_id = "paper"
            else:
                qty, fill, order_id, base_fee, quote_fee = self.live_order("buy", qty, quote_cost)
                held = qty - base_fee
                if held <= 0:
                    raise RuntimeError("Netto-Füllung unklar; Bot bleibt gesperrt")
                fee = quote_fee + base_fee * fill
                fields = {"units": held, "entry": fill, "last_candle": candle, "pending": False}
        else:
            if self.s.mode == "live":
                qty = float(self.exchange.amount_to_precision(self.s.symbol,
                                  min(units, self.live_free(self.market["base"]))))
                if qty <= 0 or not self.enough(qty, price):
                    self.store.put(paused=True)
                    raise RuntimeError("Bot-Position kleiner als Mindestorder; manuell prüfen")
                qty, fill, order_id, base_fee, quote_fee = self.live_order("sell", qty)
                fee = quote_fee + base_fee * fill
                remaining = max(0.0, units - qty - base_fee)
                fields = {"units": remaining, "entry": entry if remaining else 0,
                          "last_candle": candle, "pending": False}
            else:
                qty = units
                fill = price * (1 - self.s.slip)
                fee = qty * fill * self.s.fee
                cash = float(self.store.get("cash")) + qty * fill - fee
                fields = {"cash": cash, "units": 0.0, "entry": 0.0, "peak_price": 0.0, "last_candle": candle}
                order_id = "paper"
        self.store.record_fill(fields, side, qty, fill, fee, self.s.mode, order_id)
        return f"{self.s.mode.upper()} {side.upper()}: {qty:.8f} {self.market['base']} zu ca. {fill:.2f} EUR | Modell {p:.1%}"

    def enough(self, qty, price):
        limits = self.market.get("limits") or {}
        amount_min = (limits.get("amount") or {}).get("min") or 0
        cost_min = (limits.get("cost") or {}).get("min") or 0
        return qty >= amount_min and qty * price >= cost_min

    def handle(self, command):
        if command == "/help" or command == "/start":
            return "/scan /watch /status /stats /backtest /validate /robust /pause /resume /help"
        if command == "/watch":
            return self.watchlist_report()
        if command == "/stats":
            buys, sells, realized, fees = self.trade_stats()
            eq = self.equity() if self.s.mode == "paper" else None
            extra = f" | Equity: {eq:.2f} EUR" if eq is not None else ""
            return f"Trades: {buys} Käufe / {sells} Verkäufe | Realisiert: {realized:+.2f} EUR | Gebühren: {fees:.2f} EUR{extra}"
        if command == "/status":
            return (f"{self.s.mode.upper()} | Pause: {self.store.get('paused')} | "
                    f"Ungeklärte Order: {self.store.get('pending')} | "
                    f"Bot-Position: {self.store.get('units'):.8f} {self.market['base']} | "
                    f"Paper-EUR: {self.store.get('cash'):.2f}" if self.s.mode == "paper" else
                    f"LIVE | Pause: {self.store.get('paused')} | Ungeklärte Order: {self.store.get('pending')} | Bot-Position: {self.store.get('units'):.8f} {self.market['base']}")
        if command == "/pause":
            self.store.put(paused=True)
            return "Automatische Trades pausiert. Offene Position bleibt bestehen."
        if command == "/resume":
            if self.store.get("pending"):
                return "Ungeklärte Live-Order; erst manuell auf Kraken prüfen."
            self.store.put(paused=False)
            return "Automatische Trades fortgesetzt."
        if command == "/scan":
            frame, p = self.analysis()
            return self.report(frame, p)
        if command == "/backtest":
            return self.backtest()
        if command == "/validate":
            return self.validation_report()
        if command == "/robust":
            return self.robustness_report()
        return "Unbekannt. /help zeigt die Befehle."


def run_once(bot: Trader, next_scan: float) -> float:
    """Keep scanning even when Telegram command polling is unavailable."""
    try:
        updates = bot.telegram.updates()
    except Exception as exc:  # noqa: BLE001 - Telegram is optional for autonomous scans
        LOG.warning("Telegram-Abruf fehlgeschlagen: %s", exc)
        updates = []
    for update_id, command in updates:
        try:
            if command:
                bot.telegram.send(bot.handle(command))
        except Exception as exc:  # noqa: BLE001 - one command must not block the scan
            LOG.warning("Telegram-Befehl fehlgeschlagen: %s", exc)
            bot.telegram.send("Befehl fehlgeschlagen. Bitte später erneut versuchen.")
        bot.telegram.ack(update_id)
    if time.monotonic() >= next_scan:
        next_scan = time.monotonic() + 60
        frame, p = bot.analysis()
        result = bot.execute(frame, p)
        if result:
            bot.telegram.send(result)
    return next_scan


def offline_status(settings: Settings) -> str:
    state_path = settings.data_dir / f"{settings.mode}_{settings.symbol.replace('/', '_')}.sqlite3"
    if settings.mode == "live" and not state_path.is_file():
        return "LIVE | Zustandsdatei fehlt; Handel gesperrt. Vor init-live Kraken-Konto prüfen."
    store = Store(state_path, settings.paper_start)
    label = f"{settings.mode.upper()} | Pause: {store.get('paused')} | "
    label += f"Ungeklärte Order: {store.get('pending')} | "
    label += f"Bot-Position: {store.get('units'):.8f} {settings.symbol.split('/')[0]}"
    if settings.mode == "paper":
        label += f" | Paper-EUR: {store.get('cash'):.2f}"
    elif store.get("live_initialized") is not True:
        label += " | Live-Zustand nicht initialisiert; Handel gesperrt"
    return label


def lock_path(settings: Settings) -> Path:
    if settings.telegram_token:
        fingerprint = hashlib.sha256(settings.telegram_token.encode()).hexdigest()[:20]
        return settings.data_dir / f"telegram_{fingerprint}.lock"
    return settings.data_dir / f"{settings.mode}_{settings.symbol.replace('/', '_')}.lock"



def start_health_server():
    """Bind Render's PORT so the long-running bot can use a Web Service plan."""
    port = int(os.getenv("PORT", "10000"))

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path in {"/", "/health", "/healthz"}:
                body = b"Kraken AI Trading Bot: OK\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, name="render-health", daemon=True)
    thread.start()
    LOG.info("Render health server listening on port %s", port)
    return server

def main():
    parser = argparse.ArgumentParser(description="Kraken AI trading bot")
    parser.add_argument("command", choices=["run", "scan", "backtest", "status", "chat-id", "init-live"], nargs="?", default="run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "chat-id":
        load_dotenv()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN fehlt")
        updates = Telegram(token, "").request("getUpdates", {"timeout": 1, "limit": 20})
        ids = sorted({str(item.get("message", {}).get("chat", {}).get("id"))
                      for item in updates if item.get("message", {}).get("chat", {}).get("type") == "private"
                      and item.get("message", {}).get("chat", {}).get("id")})
        print("Chat-IDs der letzten Nachrichten:", ", ".join(ids) if ids else "keine; zuerst /start an den Bot senden")
        return
    s = Settings.load(allow_unconfigured_live=args.command in {"status", "init-live"})
    if args.command == "init-live":
        if s.mode != "live":
            raise ValueError("init-live erfordert MODE=live")
        state_path = s.data_dir / f"live_{s.symbol.replace('/', '_')}.sqlite3"
        if state_path.exists():
            raise RuntimeError("Live-Zustand existiert bereits; vor Änderungen Konto und Datei manuell abgleichen")
        state = Store(state_path, s.paper_start)
        state.put(live_initialized=True)
        print("Live-Zustand initialisiert. Erst Kontostand und offene Orders prüfen, dann Echtgeldzugang aktivieren.")
        return
    if args.command == "status":
        print(offline_status(s))
        return
    if args.command == "run":
        start_health_server()
    bot = Trader(s)
    if args.command != "run":
        print(bot.handle("/" + args.command))
        return
    # Single process per database. The operating system releases the lock on exit.
    lock = open(lock_path(s), "a+b")  # noqa: SIM115 - held for process lifetime
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            lock.write(b"1")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        raise RuntimeError("Ein Bot läuft bereits für dieses Konto und Handelspaar") from exc
    bot.telegram.send(f"Bot gestartet: {s.symbol} {s.timeframe}, {s.mode.upper()}. /help für Befehle")
    next_scan = 0
    while True:
        try:
            next_scan = run_once(bot, next_scan)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate network/model failures between scans
            message = safe_error(exc)
            LOG.error("Durchlauf fehlgeschlagen (%s): %s", type(exc).__name__, message)
            bot.telegram.send(f"Bot-Fehler: {message}. Neuer Versuch in 60 Sekunden; ungeklärte Orders bleiben gesperrt.")
            next_scan = time.monotonic() + 60
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot gestoppt")
    except Exception as error:  # noqa: BLE001 - user-facing command-line error boundary
        print(f"Fehler: {safe_error(error)}", file=sys.stderr)
        sys.exit(1)
