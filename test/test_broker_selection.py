import unittest

from broker.kis import KISBroker, KISConfig
from broker.paper import PaperBroker
from broker.kis_account import get_kis_account_report
from broker.trading_context import create_broker


class FakeResponse:
    def __init__(self, data):
        self.data = data
        self.ok = True
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse({
            "access_token": "test-token",
            "expires_in": 3600,
        })

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))

        if url.endswith("/inquire-price"):
            return FakeResponse({
                "rt_cd": "0",
                "output": {"stck_prpr": "70000"},
            })

        if url.endswith("/inquire-balance"):
            return FakeResponse({
                "rt_cd": "0",
                "output1": [{
                    "pdno": "005930",
                    "hldg_qty": "10",
                    "pchs_avg_pric": "65000",
                }],
                "output2": [{
                    "dnca_tot_amt": "1000000",
                    "prvs_rcdl_excc_amt": "950000",
                    "scts_evlu_amt": "700000",
                    "pchs_amt_smtl_amt": "650000",
                    "tot_evlu_amt": "1700000",
                    "evlu_pfls_smtl_amt": "50000",
                }],
            })

        if url.endswith("/order-cash"):
            return FakeResponse({
                "rt_cd": "0",
                "output": {"ODNO": "12345"},
            })

        raise AssertionError(f"Unexpected URL: {url}")


def make_config(enable_trading=False):
    return KISConfig(
        app_key="app-key",
        app_secret="app-secret",
        cano="12345678",
        account_type="VIRTUAL",
        enable_trading=enable_trading,
    )


class BrokerSelectionTest(unittest.TestCase):
    def test_factory_defaults_to_paper(self):
        broker = create_broker("paper")
        self.assertIsInstance(broker, PaperBroker)

    def test_kis_disabled_order_does_not_call_api(self):
        session = FakeSession()
        broker = KISBroker(
            make_config(enable_trading=False),
            session=session,
        )

        result = broker.buy(
            ticker="005930.KS",
            price=70000,
            quantity=1,
        )

        self.assertEqual(result.status, "REJECTED")
        self.assertEqual(session.calls, [])

    def test_kis_price_balance_positions_and_virtual_order(self):
        session = FakeSession()
        broker = KISBroker(
            make_config(enable_trading=True),
            session=session,
        )

        self.assertEqual(
            broker.get_current_price("005930.KS"),
            70000,
        )
        self.assertEqual(broker.get_balance()["cash"], 1000000)
        self.assertEqual(broker.get_balance()["d2_cash"], 950000)
        self.assertEqual(broker.get_balance()["stock_market_value"], 700000)
        self.assertEqual(broker.get_balance()["stock_purchase_amount"], 650000)
        self.assertIn("005930.KS", broker.get_positions())

        balance_calls = [
            call
            for call in session.calls
            if call[1].endswith("/inquire-balance")
        ]
        self.assertEqual(len(balance_calls), 1)

        result = broker.buy(
            ticker="005930.KS",
            price=70000,
            quantity=1,
            sector="SEMICONDUCTOR",
        )

        self.assertEqual(result.status, "SUBMITTED")
        self.assertEqual(result.order_id, "12345")

        order_call = next(
            call
            for call in session.calls
            if call[1].endswith("/order-cash")
        )
        self.assertEqual(
            order_call[2]["headers"]["tr_id"],
            "VTTC0802U",
        )

    def test_kis_account_report(self):
        broker = KISBroker(
            make_config(enable_trading=False),
            session=FakeSession(),
        )

        report = get_kis_account_report(broker)
        position = report["positions"]["005930.KS"]

        self.assertEqual(report["balance"]["cash"], 1000000)
        self.assertEqual(position["current_price"], 70000)
        self.assertEqual(position["market_value"], 700000)
        self.assertEqual(position["unrealized_pnl"], 50000)

    def test_kis_sell_maps_supported_order_types(self):
        session = FakeSession()
        broker = KISBroker(make_config(enable_trading=True), session=session)
        expected = {
            "LIMIT": ("00", "70000"),
            "MARKET": ("01", "0"),
            "BEST_LIMIT": ("03", "0"),
            "PRIORITY_LIMIT": ("04", "0"),
        }

        for order_type, (code, price) in expected.items():
            result = broker.sell(
                "005930.KS", 70000, 1, order_type=order_type
            )
            order_call = [
                call for call in session.calls
                if call[1].endswith("/order-cash")
            ][-1]
            body = order_call[2]["json"]
            self.assertEqual(result.status, "SUBMITTED")
            self.assertEqual(body["ORD_DVSN"], code)
            self.assertEqual(body["ORD_UNPR"], price)

    def test_kis_buy_maps_supported_order_types(self):
        session = FakeSession()
        broker = KISBroker(make_config(enable_trading=True), session=session)
        expected = {
            "LIMIT": ("00", "70000"),
            "MARKET": ("01", "0"),
            "BEST_LIMIT": ("03", "0"),
            "PRIORITY_LIMIT": ("04", "0"),
        }

        for order_type, (code, price) in expected.items():
            result = broker.buy(
                "005930.KS", 70000, 1, order_type=order_type
            )
            order_call = [
                call for call in session.calls
                if call[1].endswith("/order-cash")
            ][-1]
            body = order_call[2]["json"]
            self.assertEqual(result.status, "SUBMITTED")
            self.assertEqual(body["ORD_DVSN"], code)
            self.assertEqual(body["ORD_UNPR"], price)


if __name__ == "__main__":
    unittest.main()
