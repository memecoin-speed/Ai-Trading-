import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import ccxt

from bot import (
    Settings,
    Store,
    Telegram,
    Trader,
    lock_path,
    offline_status,
    run_once,
    safe_error,
)
from model import evaluate, feature_frame, fit_model, probability


def sample_rows(n=400):
    rows = []
    price = 25000.0
    for i in range(n):
        old = price
        price = old * (1 + 0.0012 * math.sin(i * 0.4) + 0.0005 * math.cos(i * 0.13))
        rows.append([1700000000000 + i * 3600000, old, max(old, price) * 1.001,
                     min(old, price) * 0.999, price, 20 + i % 13])
    return rows


class FakeExchange:
    def amount_to_precision(self, symbol, amount):
        return f"{amount:.8f}"


class BotTests(unittest.TestCase):
    def test_no_future_label_in_last_two_rows(self):
        frame = feature_frame(sample_rows())
        self.assertTrue(frame.target.iloc[-2:].isna().all())
        self.assertEqual(int(frame.target.iloc[10]), int(frame.open.iloc[12] > frame.open.iloc[11]))
        model = fit_model(frame.iloc[:-2])
        self.assertTrue(0 <= probability(model, frame.iloc[[-1]]) <= 1)
        report = evaluate(frame, 0.004, 0.001, 0.58, 0.45)
        self.assertEqual(report["test_bars"], 120)

    def test_public_market_pipeline_ignores_forming_bar(self):
        rows = sample_rows()
        shift = int(time.time() * 1000) - rows[-1][0] - 30 * 60 * 1000
        for row in rows:
            row[0] += shift

        class PublicExchange(FakeExchange):
            def load_markets(self):
                pass

            def market(self, symbol):
                return {"base": "BTC", "quote": "EUR", "spot": True, "active": True,
                        "limits": {"amount": {"min": .0001}, "cost": {"min": 1}}}

            def fetch_ohlcv(self, symbol, timeframe, limit):
                return rows

        # The newest row is forming. Its extreme value must not influence the score.
        rows[-1][4] = 1e9
        with tempfile.TemporaryDirectory() as directory, patch("bot.ccxt.kraken", return_value=PublicExchange()):
            s = Settings("paper", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                         .04, .08, .004, .001, Path(directory), "", "")
            bot = Trader(s)
            frame, score = bot.analysis()
            self.assertEqual(len(frame), 399)
            self.assertLess(float(frame.close.iloc[-1]), 1e6)
            self.assertTrue(0 <= score <= 1)
            self.assertEqual(bot.store.get("units"), 0)

    def test_backtest_uses_configured_position_size(self):
        frame = feature_frame(sample_rows())
        small = evaluate(frame, .004, .001, .5, .1, 1000, 25, 50, .04, .08)
        big = evaluate(frame, .004, .001, .5, .1, 1000, 1000, 1000, .04, .08)
        self.assertGreater(small["trades"], 0)
        self.assertLess(abs(small["return_pct"]), abs(big["return_pct"]))

    def test_backtest_refits_only_with_past_labels(self):
        frame = feature_frame(sample_rows(200))
        with patch("model.fit_model", wraps=fit_model) as fit:
            result = evaluate(frame, .004, .001, .58, .45)
        self.assertEqual(fit.call_count, result["test_bars"])
        first_training = fit.call_args_list[0].args[0]
        self.assertEqual(int(first_training.index[-1]), int(len(frame) * .7) - 3)

    def test_flat_market_has_neutral_rsi_and_invalid_prices_fail_closed(self):
        rows = sample_rows(200)
        for row in rows:
            row[1:5] = [25000, 25000, 25000, 25000]
            row[5] = 0
        frame = feature_frame(rows)
        self.assertEqual(frame.rsi14.iloc[-1], 50)
        self.assertEqual(frame.volume_ratio.iloc[-1], 0)
        rows[-1][4] = 0
        with self.assertRaises(ValueError):
            feature_frame(rows)

    def test_paper_buy_sell_and_duplicate_bar(self):
        frame = feature_frame(sample_rows())
        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("paper", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = FakeExchange()
            bot.market = {"base": "BTC", "quote": "EUR", "limits": {"amount": {"min": .0001}, "cost": {"min": 1}}}
            bot.store = Store(Path(directory) / "paper.sqlite3", 1000)
            msg = bot.execute(frame, .8)
            self.assertIn("BUY", msg)
            self.assertIsNone(bot.execute(frame, .8))
            self.assertGreater(bot.store.get("units"), 0)
            self.assertEqual(bot.store.db.execute("SELECT count(*) FROM trades").fetchone()[0], 1)
            frame.loc[frame.index[-1], "ts"] += 3600000
            msg = bot.execute(frame, .2)
            self.assertIn("SELL", msg)
            self.assertEqual(bot.store.get("units"), 0)
            self.assertEqual(bot.store.db.execute("SELECT count(*) FROM trades").fetchone()[0], 2)

    def test_pending_and_pause_block_orders(self):
        frame = feature_frame(sample_rows())
        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("paper", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = FakeExchange()
            bot.market = {"base": "BTC", "quote": "EUR", "limits": {}}
            bot.store = Store(Path(directory) / "paper.sqlite3", 1000)
            bot.store.put(paused=True)
            self.assertIsNone(bot.execute(frame, .9))
            bot.store.put(paused=False, pending=True)
            with self.assertRaises(RuntimeError):
                bot.execute(frame, .9)

    def test_live_order_stays_locked_until_state_is_recorded(self):
        class FilledExchange(FakeExchange):
            def create_market_buy_order_with_cost(self, *args):
                return {"id": "ORDER-123"}

            def fetch_order(self, *args):
                return {"status": "closed", "filled": .001, "average": 25000,
                        "fee": {"currency": "EUR", "cost": .1}}

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = FilledExchange()
            bot.market = {"base": "BTC", "quote": "EUR"}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with patch("bot.time.sleep"):
                filled, price, order_id, base_fee, quote_fee = bot.live_order("buy", .001, 24.9)
            self.assertEqual((filled, price, order_id, base_fee, quote_fee),
                             (.001, 25000, "ORDER-123", 0, .1))
            self.assertEqual(bot.store.get("pending"), "ORDER-123")

    def test_installed_ccxt_kraken_uses_eur_cost_and_eur_fees(self):
        exchange = ccxt.kraken()
        market = {"id": "XXBTZEUR", "altname": "XBTEUR", "symbol": "BTC/EUR", "base": "BTC",
                  "quote": "EUR", "spot": True, "active": True, "type": "spot",
                  "precision": {"amount": 1e-8, "price": .1, "cost": .01},
                  "limits": {"amount": {"min": .0001}, "cost": {"min": .5}}}
        exchange.set_markets([market])
        sent = []
        exchange.privatePostAddOrder = lambda params: (
            sent.append(params) or {"error": [], "result": {"txid": ["ORDER-1"], "descr": {"order": "buy 24.90 XBTEUR @ market"}}}
        )
        exchange.create_market_buy_order_with_cost("BTC/EUR", 24.9, {"oflags": "fciq"})
        self.assertEqual(sent[0]["volume"], "24.9")
        self.assertEqual(set(sent[0]["oflags"].split(",")), {"fciq", "viqc"})
        exchange.create_order("BTC/EUR", "market", "sell", .001, None, {"oflags": "fciq"})
        self.assertEqual(sent[1]["oflags"], "fciq")
        exchange.privatePostQueryOrders = lambda params: {"error": [], "result": {params["txid"]: {
            "status": "closed", "descr": {"pair": "XXBTZEUR", "type": "buy", "ordertype": "market"},
            "vol": ".001", "vol_exec": ".001", "cost": "25", "fee": ".1",
            "price": "25000", "oflags": "fciq,viqc", "trades": []}}}
        parsed = exchange.fetch_order("ORDER-1", "BTC/EUR")
        self.assertEqual(parsed["fee"]["currency"], "EUR")
        self.assertEqual(parsed["average"], 25000)

    def test_live_buy_cost_cap_and_net_units(self):
        class LiveExchange(FakeExchange):
            spent = None

            def fetch_balance(self):
                return {"free": {"EUR": 1000, "BTC": .000998}}

            def create_market_buy_order_with_cost(self, symbol, cost, params):
                self.spent = cost
                self.assert_flags = params
                return {"id": "ORDER-456"}

            def fetch_order(self, *args):
                return {"status": "closed", "filled": .001, "average": 25000,
                        "fee": {"currency": "EUR", "cost": .1}}

        frame = feature_frame(sample_rows())
        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = LiveExchange()
            bot.market = {"base": "BTC", "quote": "EUR", "limits": {"amount": {"min": .0001}, "cost": {"min": 1}}}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with patch("bot.time.sleep"):
                self.assertIn("BUY", bot.execute(frame, .9))
            self.assertLessEqual(bot.exchange.spent, 25)
            self.assertEqual(bot.exchange.assert_flags, {"oflags": "fciq"})
            self.assertAlmostEqual(bot.store.get("units"), .001)
            self.assertFalse(bot.store.get("pending"))
            self.assertEqual(bot.store.db.execute("SELECT count(*) FROM trades").fetchone()[0], 1)
            self.assertIsNone(bot.execute(frame, .9))

    def test_unknown_live_submission_blocks_retry(self):
        class UncertainExchange(FakeExchange):
            def create_market_buy_order_with_cost(self, *args):
                raise TimeoutError("connection closed after submit")

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = UncertainExchange()
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with self.assertRaises(TimeoutError):
                bot.live_order("buy", .001, 25)
            self.assertTrue(bot.store.get("pending"))

    def test_explicit_live_rejection_pauses_without_pending_order(self):
        class RejectedExchange(FakeExchange):
            def create_market_buy_order_with_cost(self, *args):
                raise ccxt.InsufficientFunds("insufficient funds")

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = RejectedExchange()
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with self.assertRaises(ccxt.InsufficientFunds):
                bot.live_order("buy", .001, 25)
            self.assertFalse(bot.store.get("pending"))
            self.assertTrue(bot.store.get("paused"))

    def test_invalid_live_fill_remains_locked(self):
        class InvalidFill(FakeExchange):
            def create_market_buy_order_with_cost(self, *args):
                return {"id": "ORDER-BAD"}

            def fetch_order(self, *args):
                return {"status": "closed", "filled": float("nan"), "average": 25000}

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = InvalidFill()
            bot.market = {"base": "BTC", "quote": "EUR"}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with patch("bot.time.sleep"), self.assertRaises(RuntimeError):
                bot.live_order("buy", .001, 25)
            self.assertEqual(bot.store.get("pending"), "ORDER-BAD")

    def test_unexpected_base_currency_fee_remains_locked(self):
        class WrongFee(FakeExchange):
            def create_market_buy_order_with_cost(self, *args):
                return {"id": "ORDER-FEE"}

            def fetch_order(self, *args):
                return {"status": "closed", "filled": .001, "average": 25000,
                        "fee": {"currency": "BTC", "cost": .000002}}

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = WrongFee()
            bot.market = {"base": "BTC", "quote": "EUR"}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with patch("bot.time.sleep"), self.assertRaises(RuntimeError):
                bot.live_order("buy", .001, 25)
            self.assertEqual(bot.store.get("pending"), "ORDER-FEE")

    def test_transient_order_lookup_does_not_resubmit(self):
        class SlowOrder(FakeExchange):
            submissions = 0
            lookups = 0

            def create_market_buy_order_with_cost(self, *args):
                self.submissions += 1
                return {"id": "ORDER-SLOW"}

            def fetch_order(self, *args):
                self.lookups += 1
                if self.lookups == 1:
                    raise ccxt.OrderNotFound("not indexed yet")
                return {"status": "closed", "filled": .001, "average": 25000,
                        "fee": {"currency": "EUR", "cost": .1}}

        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = SlowOrder()
            bot.market = {"base": "BTC", "quote": "EUR"}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            with patch("bot.time.sleep"):
                bot.live_order("buy", .001, 25)
            self.assertEqual(bot.exchange.submissions, 1)
            self.assertEqual(bot.exchange.lookups, 2)

    def test_live_sell_with_eur_fee_closes_only_bot_position(self):
        class SellExchange(FakeExchange):
            sell_amount = None
            sell_flags = None

            def fetch_balance(self):
                return {"free": {"BTC": .001}}

            def create_order(self, symbol, order_type, side, amount, price, params):
                self.sell_amount, self.sell_flags = amount, params
                return {"id": "ORDER-SELL"}

            def fetch_order(self, *args):
                return {"status": "closed", "filled": .001, "average": 26000,
                        "fee": {"currency": "EUR", "cost": .1}}

        frame = feature_frame(sample_rows())
        with tempfile.TemporaryDirectory() as directory:
            bot = Trader.__new__(Trader)
            bot.s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                             .04, .08, .004, .001, Path(directory), "", "")
            bot.exchange = SellExchange()
            bot.market = {"base": "BTC", "quote": "EUR", "limits": {"amount": {"min": .0001}, "cost": {"min": 1}}}
            bot.store = Store(Path(directory) / "live.sqlite3", 1000)
            bot.store.put(units=.001, entry=25000)
            with patch("bot.time.sleep"):
                self.assertIn("SELL", bot.execute(frame, .2))
            self.assertEqual(bot.exchange.sell_amount, .001)
            self.assertEqual(bot.exchange.sell_flags, {"oflags": "fciq"})
            self.assertEqual(bot.store.get("units"), 0)
            self.assertFalse(bot.store.get("pending"))

    def test_trade_and_state_commit_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.sqlite3"
            store = Store(path, 1000)
            store.record_fill({"cash": 974.9, "units": .001, "last_candle": 123},
                              "buy", .001, 25000, .1, "paper", "paper")
            store.db.close()
            reopened = Store(path, 1000)
            self.assertEqual(reopened.get("last_candle"), 123)
            self.assertEqual(reopened.get("units"), .001)
            self.assertEqual(reopened.db.execute("SELECT count(*) FROM trades").fetchone()[0], 1)

    def test_status_needs_no_exchange_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            s = Settings("paper", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                         .04, .08, .004, .001, Path(directory), "", "")
            self.assertIn("Paper-EUR: 1000.00", offline_status(s))

    def test_telegram_does_not_run_other_chat_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "telegram.sqlite3", 1000)
            store.put(telegram_offset=100)
            t = Telegram("token", "42", store)
            t.request = lambda method, params: [
                {"update_id": 100, "message": {"chat": {"id": 99, "type": "private"}, "text": "/resume"}},
                {"update_id": 101, "message": {"chat": {"id": 42, "type": "private"}, "text": "/status"}},
            ]
            self.assertEqual(t.updates(), [(100, None), (101, "/status")])
            self.assertEqual(store.get("telegram_offset"), 100)
            t.ack(100)
            t.ack(101)
            self.assertEqual(store.get("telegram_offset"), 102)

    def test_telegram_discards_old_commands_on_first_start(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "telegram.sqlite3", 1000)
            t = Telegram("token", "42", store)
            calls = []

            def request(method, params):
                calls.append(params)
                if len(calls) == 1:
                    return [{"update_id": 250, "message": {"chat": {"id": 42, "type": "private"}, "text": "/resume"}}]
                return [{"update_id": 251, "message": {"chat": {"id": 42, "type": "private"}, "text": "/status"}}]

            t.request = request
            self.assertEqual(t.updates(), [])
            self.assertEqual(t.updates(), [(251, "/status")])
            self.assertEqual(calls[0]["offset"], -1)
            t.ack(251)
            self.assertEqual(store.get("telegram_offset"), 252)

    def test_group_commands_are_ignored_even_with_matching_chat_id(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "telegram.sqlite3", 1000)
            store.put(telegram_offset=1)
            t = Telegram("token", "42", store)
            t.request = lambda method, params: [
                {"update_id": 1, "message": {"chat": {"id": 42, "type": "group"}, "text": "/resume"}}
            ]
            self.assertEqual(t.updates(), [(1, None)])

    def test_live_mode_requires_explicit_switch_and_keys(self):
        keys = {"MODE": "live", "ENABLE_LIVE_TRADING": "NO", "KRAKEN_API_KEY": "key",
                "KRAKEN_API_SECRET": "secret"}
        with patch.dict(os.environ, keys), self.assertRaises(ValueError):
            Settings.load()

    def test_nonfinite_capital_is_rejected(self):
        with patch.dict(os.environ, {"MODE": "paper", "PAPER_START_EUR": "NaN"}), self.assertRaises(ValueError):
            Settings.load()

    def test_errors_do_not_print_telegram_token(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "ABCSECRET"}):
            message = safe_error(RuntimeError("https://api.telegram.org/botABCSECRET/getUpdates"))
        self.assertNotIn("ABCSECRET", message)

    def test_live_status_can_be_read_without_trading_keys(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MODE": "live", "DATA_DIR": directory, "ENABLE_LIVE_TRADING": "NO",
                         "KRAKEN_API_KEY": "", "KRAKEN_API_SECRET": ""}
        ):
            s = Settings.load(allow_unconfigured_live=True)
            self.assertIn("LIVE |", offline_status(s))
            with self.assertRaises(ValueError):
                Settings.load()

    def test_missing_live_state_fails_closed_and_init_is_one_time(self):
        with tempfile.TemporaryDirectory() as directory:
            s = Settings("live", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                         .04, .08, .004, .001, Path(directory), "", "")
            self.assertIn("Zustandsdatei fehlt", offline_status(s))
            with self.assertRaises(RuntimeError):
                Trader(s)
            env = dict(os.environ, MODE="live", DATA_DIR=directory, ENABLE_LIVE_TRADING="NO",
                       KRAKEN_API_KEY="", KRAKEN_API_SECRET="")
            first = subprocess.run([sys.executable, "bot.py", "init-live"], env=env,
                                   capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            state = Store(Path(directory) / "live_BTC_EUR.sqlite3", 1000)
            self.assertTrue(state.get("live_initialized"))
            second = subprocess.run([sys.executable, "bot.py", "init-live"], env=env,
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(second.returncode, 0)

    def test_one_telegram_token_uses_one_lock_across_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            s1 = Settings("paper", "BTC/EUR", "1h", 1000, 25, 50, .58, .45,
                          .04, .08, .004, .001, Path(directory), "secret-token", "42")
            s2 = Settings("live", "ETH/EUR", "1h", 1000, 25, 50, .58, .45,
                          .04, .08, .004, .001, Path(directory), "secret-token", "42")
            self.assertEqual(lock_path(s1), lock_path(s2))
            self.assertNotIn("secret-token", str(lock_path(s1)))

    def test_telegram_outage_does_not_block_automatic_scan(self):
        class BrokenTelegram:
            def updates(self):
                raise TimeoutError("Telegram unavailable")

            def send(self, message):
                pass

        class FakeBot:
            telegram = BrokenTelegram()
            scanned = False

            def analysis(self):
                self.scanned = True
                return "frame", .6

            def execute(self, frame, score):
                return None

        bot = FakeBot()
        self.assertGreater(run_once(bot, 0), 0)
        self.assertTrue(bot.scanned)

    def test_pause_is_persisted_before_telegram_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "paper.sqlite3", 1000)
            store.put(telegram_offset=10)
            telegram = Telegram("token", "42", store)
            telegram.request = lambda method, params: [
                {"update_id": 10, "message": {"chat": {"id": 42, "type": "private"}, "text": "/pause"}}
            ]
            pending = telegram.updates()
            self.assertEqual(store.get("telegram_offset"), 10)
            self.assertEqual(pending, [(10, "/pause")])
            # Simulated crash before /pause handling: a restarted bot sees it again.
            restarted = Telegram("token", "42", store)
            restarted.request = telegram.request
            self.assertEqual(restarted.updates(), [(10, "/pause")])


if __name__ == "__main__":
    unittest.main()
