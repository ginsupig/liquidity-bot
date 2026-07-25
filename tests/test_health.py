from datetime import timedelta

from wliq.broker.health import DataHealthMonitor
from tests.conftest import T0


def test_unknown_symbol_unhealthy():
    h = DataHealthMonitor()
    h.on_connect()
    ok, why = h.is_healthy("MU", T0)
    assert not ok and why == "no_data"


def test_stale_detection():
    h = DataHealthMonitor(stale_ms=1000)
    h.on_connect()
    h.on_event("MU", T0)
    ok, why = h.is_healthy("MU", T0 + timedelta(seconds=2))
    assert not ok and why.startswith("stale_data")


def test_sequence_gap_detection():
    h = DataHealthMonitor(gap_tolerance=2)
    h.on_connect()
    h.on_event("MU", T0, seq=1)
    h.on_event("MU", T0, seq=10)  # 8 dropped
    ok, why = h.is_healthy("MU", T0)
    assert not ok and "sequence_gaps" in why
    assert h.dropped_messages == 8
