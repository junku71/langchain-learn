import tempfile
import unittest
import csv
from pathlib import Path

from broker.models import OrderResult
from broker.paper import PaperBroker
from paper.daily_loop import DailyPaperTradingLoop
from paper.trade_logger import TradeLogger


class DailyPortfolioLoopTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.broker = PaperBroker(
            initial_cash=100_000,
            commission_rate=0,
        )
        self.logger = TradeLogger(
            filename=str(
                Path(self.temp_dir.name) / "trades.csv"
            )
        )
        self.loop = DailyPaperTradingLoop(
            broker=self.broker,
            logger=self.logger,
        )

        self.broker.buy(
            ticker="STOP",
            price=100,
            quantity=10,
            sector="A",
            stop_loss=95,
            take_profit=120,
            trailing_stop_pct=0.10,
        )
        self.broker.buy(
            ticker="TARGET",
            price=200,
            quantity=10,
            sector="B",
            stop_loss=180,
            take_profit=220,
            trailing_stop_pct=0.10,
        )
        self.broker.buy(
            ticker="TRAIL",
            price=50,
            quantity=10,
            sector="C",
            stop_loss=40,
            take_profit=80,
            trailing_stop_pct=0.10,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_daily_exits_and_portfolio_recalculation(self):
        first_day = self.loop.run_day(
            date="2026-09-01",
            bars={
                "STOP": {
                    "Open": 100,
                    "High": 105,
                    "Low": 94,
                    "Close": 96,
                },
                "TARGET": {
                    "Open": 200,
                    "High": 225,
                    "Low": 195,
                    "Close": 218,
                },
                "TRAIL": {
                    "Open": 50,
                    "High": 60,
                    "Low": 49,
                    "Close": 58,
                },
            },
        )

        self.assertEqual(
            first_day["actions"]["STOP"]["reason"],
            "STOP_LOSS",
        )
        self.assertEqual(
            first_day["actions"]["TARGET"]["reason"],
            "TAKE_PROFIT",
        )
        self.assertEqual(
            first_day["actions"]["TRAIL"]["action"],
            "HOLD",
        )
        self.assertEqual(
            first_day["actions"]["TRAIL"]["trailing_stop"],
            54,
        )


        self.assertEqual(
            set(first_day["portfolio"]["positions"]),
            {"TRAIL"},
        )
        self.assertEqual(
            first_day["portfolio"]["total_pnl"],
            230,
        )

        second_day = self.loop.run_day(
            date="2026-09-02",
            bars={
                "TRAIL": {
                    "Open": 53,
                    "High": 55,
                    "Low": 52,
                    "Close": 54,
                },
            },
        )

        self.assertEqual(
            second_day["actions"]["TRAIL"]["reason"],
            "TRAILING_STOP_GAP_DOWN",
        )
        self.assertEqual(
            second_day["portfolio"]["positions"],
            {},
        )
        self.assertEqual(
            second_day["portfolio"]["total_equity"],
            100_180,
        )


def test_trade_logger_upgrades_legacy_csv_and_logs_stock_name(tmp_path):
    path = tmp_path / "trades.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerows([
            [
                "timestamp", "ticker", "side", "status", "price", "quantity",
                "commission", "realized_pnl", "reason",
            ],
            ["2026-09-01", "005930.KS", "BUY", "FILLED", "70000", "2", "0", "0", ""],
        ])

    logger = TradeLogger(str(path))
    logger.log(OrderResult(
        status="FILLED", ticker="005930.KS", side="BUY", price=71000, quantity=1
    ))

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    assert rows[0][:4] == ["timestamp", "ticker", "name", "side"]
    assert rows[1][1] == "005930.KS"
    assert rows[1][2]
    assert rows[2][1] == "005930.KS"
    assert rows[2][2]


if __name__ == "__main__":
    unittest.main()
