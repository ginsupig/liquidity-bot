from datetime import datetime, timezone

import pytest

from wliq.core.clock import SimClock
from wliq.core.journal import OrderJournal
from wliq.core.models import Fill, OrderRequest, Side
from wliq.execution.order_manager import (IllegalTransition, ManagedOrder,
                                          OrderManager, OState)

T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


def om(tmp_path, cap=float("inf")):
    clock = SimClock(T0)
    return clock, OrderManager(clock, OrderJournal(tmp_path / "j.jsonl"),
                               max_gross_notional_usd=cap)


def req(coid="c1", intent="entry", qty=10, sym="MU"):
    return OrderRequest(sym, Side.SELL, qty, "LIMIT", 100.0, coid, "t", intent)


def test_lifecycle_and_reservation_release(tmp_path):
    clock, m = om(tmp_path)
    o = m.create(req(), ref_price=100.0, risk_usd=25.0, ttl_ms=5000)
    assert m.reserved_notional == 1000.0 and m.reserved_risk == 25.0
    m.mark_submitting("c1"); m.on_ack("c1")
    assert o.state is OState.ACKNOWLEDGED
    m.on_fill(Fill("MU", Side.SELL, 4, 100.0, clock.now(), "c1", "t", True))
    assert o.state is OState.PARTIAL and 0 < m.reserved_notional < 1000.0
    m.on_fill(Fill("MU", Side.SELL, 6, 99.9, clock.now(), "c1", "t", False))
    assert o.state is OState.FILLED
    assert o.filled_qty == 10 and abs(o.avg_fill_price - 99.94) < 1e-9
    assert m.reserved_notional == 0.0 and m.reserved_risk == 0.0


def test_illegal_transition_raises(tmp_path):
    clock, m = om(tmp_path)
    m.create(req(), 100.0, 10.0, None)
    with pytest.raises(IllegalTransition):
        m.on_cancel_ack("c1")   # NEW -> CANCELLED illegal
        m._transition(m.orders["c1"], OState.CANCELLED, "bad")


def test_one_active_entry_per_symbol(tmp_path):
    clock, m = om(tmp_path)
    m.create(req("c1"), 100.0, 10.0, None)
    with pytest.raises(IllegalTransition):
        m.create(req("c2"), 100.0, 10.0, None)
    m.create(req("c3", intent="exit"), 100.0, 0.0, None)  # exits are fine


def test_gross_notional_cap(tmp_path):
    clock, m = om(tmp_path, cap=1500.0)
    m.create(req("c1"), 100.0, 10.0, None)
    with pytest.raises(IllegalTransition):
        m.create(req("c2", sym="WDC"), 100.0, 10.0, None)  # 2000 > 1500


def test_timer_ttl_cancel_and_no_ack_expiry(tmp_path):
    clock, m = om(tmp_path)
    m.create(req("c1"), 100.0, 10.0, ttl_ms=1000)
    m.mark_submitting("c1"); m.on_ack("c1")
    m.create(req("c2", sym="WDC"), 100.0, 10.0, ttl_ms=1000)
    m.mark_submitting("c2")                       # never acked
    clock.advance_ms(1500)
    to_cancel = m.poll()
    assert to_cancel == ["c1"]
    assert m.orders["c1"].state is OState.CANCEL_PENDING
    assert m.orders["c2"].state is OState.EXPIRED
    m.on_cancel_ack("c1")
    assert m.orders["c1"].state is OState.CANCELLED
    assert m.reserved_notional == 0.0


def test_duplicate_coid_rejected(tmp_path):
    clock, m = om(tmp_path)
    m.create(req("c1"), 100.0, 10.0, None)
    with pytest.raises(IllegalTransition):
        m.create(req("c1"), 100.0, 10.0, None)
