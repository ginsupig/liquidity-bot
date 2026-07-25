from datetime import datetime, timedelta, timezone

from wliq.core.clock import SimClock
from wliq.core.config import PaperFillConfig
from wliq.core.models import OrderRequest, Quote, Side, Trade
from wliq.execution.venue_paper import PaperVenue

T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


def mk(clock=None, **kw):
    cfg = PaperFillConfig(latency_ms=200, cancel_latency_ms=100,
                          queue_position_factor=1.0, require_trade_through=True,
                          partial_fill_min_shares=1,
                          marketable_max_take_fraction=0.5)
    clock = clock or SimClock(T0)
    return clock, PaperVenue(cfg, clock, ack_latency_ms=50, **kw)


def q(ts, bid=99.98, ask=100.02, bs=500, asz=500):
    return Quote("MU", ts, bid, ask, bs, asz)


def test_market_fills_only_after_activation_from_next_quote():
    clock, v = mk()
    v.on_quote(q(T0))
    req = OrderRequest("MU", Side.BUY, 100, "MARKET", None, "m1", "t", "exit")
    v.submit(req)
    # quote arriving BEFORE activation must not fill
    clock.advance_ms(100)
    evs = v.on_quote(q(clock.now() - timedelta(milliseconds=150)))
    assert not [e for e in evs if e.kind == "fill"]
    clock.advance_ms(200)   # past 200ms latency
    evs = v.on_quote(q(clock.now(), ask=100.05, asz=400))
    fills = [e for e in evs if e.kind == "fill"]
    assert fills and fills[0].fill.price == 100.05
    assert fills[0].fill.quantity == 100  # min(400*0.5, 100)


def test_ioc_partial_then_cancel_remainder():
    clock, v = mk()
    v.on_quote(q(T0))
    req = OrderRequest("MU", Side.BUY, 400, "LIMIT", 100.02, "i1", "t", "entry")
    v.submit(req, tif="IOC")
    clock.advance_ms(250)
    evs = v.on_quote(q(clock.now(), asz=200))   # only 100 takeable (0.5*200)
    kinds = [e.kind for e in evs]
    assert "fill" in kinds and "cancel_ack" in kinds
    f = next(e for e in evs if e.kind == "fill")
    assert f.fill.quantity == 100


def test_fok_all_or_cancel():
    clock, v = mk()
    v.on_quote(q(T0))
    req = OrderRequest("MU", Side.BUY, 1000, "LIMIT", 100.02, "f1", "t", "entry")
    v.submit(req, tif="FOK")
    clock.advance_ms(250)
    evs = v.on_quote(q(clock.now(), asz=400))   # 200 available < 1000
    assert [e.kind for e in evs] == ["cancel_ack"]


def test_ack_latency_and_cancel_via_poll():
    clock, v = mk()
    v.on_quote(q(T0))
    req = OrderRequest("MU", Side.SELL, 100, "LIMIT", 100.10, "p1", "t", "entry")
    v.submit(req)
    assert v.poll() == []           # before ack latency
    clock.advance_ms(60)
    assert [e.kind for e in v.poll()] == ["ack"]
    v.cancel("p1")
    clock.advance_ms(50)
    assert v.poll() == []           # cancel latency not elapsed
    clock.advance_ms(60)
    assert [e.kind for e in v.poll()] == ["cancel_ack"]


def test_stressed_fill_probability_blocks_fills():
    clock, v = mk(fill_probability=0.0)
    v.on_quote(q(T0))
    req = OrderRequest("MU", Side.BUY, 100, "LIMIT", 99.98, "s1", "t", "entry")
    v.submit(req)
    clock.advance_ms(300)
    evs = v.on_trade(Trade("MU", clock.now(), 99.90, 500, None))  # trade-through
    assert not [e for e in evs if e.kind == "fill"]
