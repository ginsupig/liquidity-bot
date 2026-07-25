from datetime import datetime, timedelta, timezone

from wliq.core.models import DepthLevel, DepthSnapshot, Quote, Side, Trade
from wliq.data.quality import DataQualityMonitor

T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


def q(i, bid=99.98, ask=100.02):
    return Quote("MU", T0 + timedelta(seconds=i), bid, ask, 500, 500)


def test_duplicates_dropped_and_counted():
    m = DataQualityMonitor()
    assert m.on_quote(q(1), arrival=T0 + timedelta(seconds=1))
    assert not m.on_quote(q(1), arrival=T0 + timedelta(seconds=1))
    assert m.stats["MU"].duplicates == 1


def test_sequence_gap_detection():
    m = DataQualityMonitor()
    m.on_trade(Trade("MU", T0, 100.0, 100, Side.BUY), seq=10, arrival=T0)
    m.on_trade(Trade("MU", T0 + timedelta(seconds=1), 100.1, 100, Side.BUY),
               seq=14, arrival=T0 + timedelta(seconds=1))
    assert m.stats["MU"].seq_gaps == 3


def test_crossed_l1_and_misaligned_trade():
    m = DataQualityMonitor()
    m.on_quote(Quote("MU", T0, 100.05, 100.00, 500, 500), arrival=T0)  # crossed
    assert m.stats["MU"].l1_violations == 1
    m.on_quote(q(1), arrival=T0 + timedelta(seconds=1))
    m.on_trade(Trade("MU", T0 + timedelta(seconds=2), 101.5, 100, Side.BUY),
               arrival=T0 + timedelta(seconds=2))
    assert m.stats["MU"].misaligned_trades == 1


def test_confidence_fails_closed_then_recovers():
    m = DataQualityMonitor()
    assert m.confidence("MU") == 0.0          # zero evidence -> zero confidence
    for i in range(30):
        m.on_quote(q(i), arrival=T0 + timedelta(seconds=i))
    assert m.confidence("MU") > 0.9


def test_l2_sorted_violation():
    m = DataQualityMonitor()
    snap = DepthSnapshot("MU", T0, [DepthLevel(99.90, 100), DepthLevel(99.95, 100)],
                         [DepthLevel(100.05, 100)])   # bids not descending
    m.on_depth(snap)
    assert m.stats["MU"].l2_violations >= 1
