from pathlib import Path

from wliq.core.journal import OrderJournal
from wliq.core.models import OrderRequest, OrderStatus, Side
from tests.conftest import T0


def test_deterministic_ids_and_duplicate_detection(tmp_path: Path):
    j = OrderJournal(tmp_path / "orders.jsonl")
    coid = j.next_client_order_id("MU", "entry", T0)
    assert coid == f"wliq-{T0.date().isoformat()}-MU-entry-0001"
    order = OrderRequest("MU", Side.SELL, 100, "LIMIT", 100.0, coid, "trace1")
    assert not j.is_duplicate(coid)
    j.record("submit", order, OrderStatus.PENDING)
    assert j.is_duplicate(coid)


def test_sequence_survives_restart(tmp_path: Path):
    path = tmp_path / "orders.jsonl"
    j1 = OrderJournal(path)
    c1 = j1.next_client_order_id("MU", "entry", T0)
    j1.record("submit", OrderRequest("MU", Side.SELL, 100, "LIMIT", 100.0, c1, "t"),
              OrderStatus.PENDING)
    j2 = OrderJournal(path)  # simulated restart
    c2 = j2.next_client_order_id("MU", "entry", T0)
    assert c2.endswith("-0002")
    assert j2.is_duplicate(c1)
