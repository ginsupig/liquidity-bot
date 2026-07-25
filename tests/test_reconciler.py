from wliq.core.models import GovernorAction, Position, Side
from wliq.execution.reconciler import reconcile
from tests.conftest import T0


class FakeBroker:
    def __init__(self, positions: dict[str, float], ok: bool = True):
        self._p, self._ok = positions, ok
    def read_positions(self): return self._p
    def read_open_orders(self): return []
    def read_account_ok(self): return self._ok


def local(symbol: str, qty: float, side: Side) -> dict[str, Position]:
    return {symbol: Position(symbol, side, qty, 100.0, T0, 101.0, 99.0, 3.0, "t",
                             GovernorAction.REVERSAL)}


def test_match_ok():
    r = reconcile(local("MU", 100, Side.SELL), FakeBroker({"MU": -100.0}))
    assert r.ok


def test_mismatch_flagged():
    r = reconcile(local("MU", 100, Side.SELL), FakeBroker({"MU": -50.0}))
    assert not r.ok and any("position_mismatch:MU" in i for i in r.issues)


def test_broker_read_failure_fails_closed():
    class Boom(FakeBroker):
        def read_positions(self): raise ConnectionError("down")
    r = reconcile({}, Boom({}))
    assert not r.ok and "broker_read_failed" in r.issues[0]


def test_unknown_account_flagged():
    r = reconcile({}, FakeBroker({}, ok=False))
    assert not r.ok and "account_state_unknown" in r.issues
