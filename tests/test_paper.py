from datetime import timedelta

from wliq.core.config import PaperFillConfig
from wliq.core.models import OrderRequest, OrderStatus, Side
from wliq.execution.paper import PaperBroker
from tests.conftest import T0, q, t


def order(side: Side, qty: float, limit: float | None, coid: str = "c1") -> OrderRequest:
    return OrderRequest("MU", side, qty, "LIMIT" if limit else "MARKET",
                        limit, coid, "trace1")


def test_passive_limit_does_not_fill_on_touch_alone():
    b = PaperBroker(PaperFillConfig(latency_ms=0))
    quote = q("MU", 0, 100.00, 100.02, bs=300)
    b.submit(order(Side.BUY, 100, 100.00), quote, T0)
    # price trades at our limit but volume < queue ahead -> no fill
    fills = b.on_trade(t("MU", 1, 100.00, 200), T0 + timedelta(seconds=1))
    assert fills == []
    assert b.statuses["c1"] is OrderStatus.SUBMITTED


def test_passive_fill_on_strict_trade_through():
    b = PaperBroker(PaperFillConfig(latency_ms=0))
    quote = q("MU", 0, 100.00, 100.02, bs=300)
    b.submit(order(Side.BUY, 100, 100.00), quote, T0)
    b.on_trade(t("MU", 1, 100.00, 300), T0 + timedelta(seconds=1))  # queue consumed, no fill
    b.on_trade(t("MU", 2, 100.00, 50), T0 + timedelta(seconds=2))   # spill without through
    assert b.statuses["c1"] is OrderStatus.SUBMITTED
    fills = b.on_trade(t("MU", 3, 99.99, 100), T0 + timedelta(seconds=3))  # strict through
    assert len(fills) == 1 and fills[0].quantity == 100 and fills[0].price == 100.00
    assert b.statuses["c1"] is OrderStatus.FILLED


def test_queue_spill_fills_after_through_flag_via_require_flag_off():
    b = PaperBroker(PaperFillConfig(latency_ms=0, require_trade_through=False))
    quote = q("MU", 0, 100.00, 100.02, bs=200)
    b.submit(order(Side.BUY, 100, 100.00), quote, T0)
    fills = b.on_trade(t("MU", 1, 100.00, 350), T0 + timedelta(seconds=1))
    assert len(fills) == 1 and fills[0].quantity == 100


def test_marketable_partial_fill_capped_by_displayed_size():
    b = PaperBroker(PaperFillConfig(latency_ms=0, marketable_max_take_fraction=0.5))
    quote = q("MU", 0, 100.00, 100.02, asz=100)
    status = b.submit(order(Side.BUY, 200, 100.02), quote, T0)
    assert status is OrderStatus.PARTIAL
    assert b.fills[0].quantity == 50


def test_duplicate_submit_is_idempotent():
    b = PaperBroker(PaperFillConfig(latency_ms=0))
    quote = q("MU", 0, 100.00, 100.02)
    b.submit(order(Side.SELL, 100, 100.02), quote, T0)
    n = len(b.orders)
    b.submit(order(Side.SELL, 100, 100.02), quote, T0)
    assert len(b.orders) == n


def test_cancel_latency_respected():
    b = PaperBroker(PaperFillConfig(latency_ms=0, cancel_latency_ms=500))
    quote = q("MU", 0, 100.00, 100.02, bs=100)
    b.submit(order(Side.BUY, 100, 100.00), quote, T0)
    b.cancel("c1", T0)
    # before cancel latency elapses, order can still work
    b.on_trade(t("MU", 0.1, 99.99, 100), T0 + timedelta(milliseconds=100))
    b.on_trade(t("MU", 0.2, 100.00, 500), T0 + timedelta(milliseconds=200))
    assert b.statuses["c1"] in (OrderStatus.FILLED, OrderStatus.PARTIAL)
