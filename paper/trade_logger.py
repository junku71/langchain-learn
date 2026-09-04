import csv
from pathlib import Path
from datetime import datetime

from broker.models import (
    OrderResult,
)


def _stock_name(ticker: str) -> str:
    # Lazy import avoids paper.trade_logger -> trading package -> graphs ->
    # paper.trade_logger circular initialization.
    from trading.display import stock_name
    return stock_name(ticker)


class TradeLogger:

    def __init__(
        self,
        filename: str = "logs/trades.csv",
    ):

        self.filename = Path(
            filename
        )
        self._include_name = False

        self.filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.filename.exists():

            self._create_file()
            self._include_name = True
        else:
            self._upgrade_schema()

    def _create_file(self):

        with self.filename.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "ticker",
                "name",
                "side",
                "status",
                "price",
                "quantity",
                "commission",
                "realized_pnl",
                "reason",
            ])

    def _upgrade_schema(self):
        """Insert the name column into legacy trade logs without losing rows."""
        with self.filename.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if rows and "name" in rows[0]:
            self._include_name = True
            return
        if not rows or "ticker" not in rows[0]:
            return
        ticker_index = rows[0].index("ticker")
        rows[0].insert(ticker_index + 1, "name")
        for row in rows[1:]:
            ticker = row[ticker_index] if len(row) > ticker_index else ""
            row.insert(ticker_index + 1, _stock_name(ticker) if ticker else "")
        temporary = self.filename.with_suffix(self.filename.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        try:
            temporary.replace(self.filename)
        except PermissionError:
            # A separately running scheduler may temporarily lock the CSV on
            # Windows. Keep the legacy row shape until the next process start.
            temporary.unlink(missing_ok=True)
            return
        self._include_name = True

    def log(
        self,
        result: OrderResult,
    ):

        with self.filename.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(f)

            row = [
                datetime.now().isoformat(),

                result.ticker,

                result.side,

                result.status,

                result.price,

                result.quantity,

                result.commission,

                result.realized_pnl,

                result.reason,
            ]
            if self._include_name:
                row.insert(2, _stock_name(result.ticker))
            writer.writerow(row)
