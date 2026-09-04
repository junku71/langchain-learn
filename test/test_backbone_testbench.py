import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("backbone_testbench.py")
SPEC = importlib.util.spec_from_file_location("backbone_testbench", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

create_initial_state = MODULE.create_initial_state
resolve_domestic_stock = MODULE.resolve_domestic_stock


def test_resolves_korean_stock_name():
    ticker, name = resolve_domestic_stock("삼성전자")

    assert ticker == "005930.KS"
    assert name == "삼성전자"


def test_rejects_non_domestic_ticker():
    with pytest.raises(KeyError):
        resolve_domestic_stock("NVDA")


def test_initial_state_contains_backbone_contract():
    state = create_initial_state("005930.KS", sector="반도체")

    assert state["ticker"] == "005930.KS"
    assert state["sector"] == "반도체"
    assert state["risk_per_trade"] == 0.01
    assert state["portfolio_guard_result"] is None
