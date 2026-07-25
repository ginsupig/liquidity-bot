from datetime import timedelta

from wliq.core.models import Fill, GovernorAction, Side, Signal
from wliq.execution.positions import PositionManager
from tests.conftest import T0, q
from tests.test_strategy import fv


def entry(pm: PositionManager) -> None:
    sig = Signal("MU", T0, Side.SELL, GovernorAction.REVERSAL, 4.0,
                 100.0, 100.22, 99.60, "trace1")
    pm.on_entry_fill(Fill("MU", Side.SELL, 100, 100.0, T0, "c1", "trace1"), sig)


def test_hard_stop_short():
    pm = PositionManager()
    entry(pm)
    instr = pm.check_exits(q("MU", 10, 100.24, 100.26), None, T0 + timedelta(seconds=10), 300)
    assert instr is not None and instr.reason == "hard_stop"
    assert instr.side is Side.BUY


def test_target_short():
    pm = PositionManager()
    entry(pm)
    instr = pm.check_exits(q("MU", 10, 99.55, 99.57), None, T0 + timedelta(seconds=10), 300)
    assert instr is not None and instr.reason == "target"


def test_time_stop():
    pm = PositionManager()
    entry(pm)
    instr = pm.check_exits(q("MU", 400, 100.0, 100.02), None, T0 + timedelta(seconds=400), 300)
    assert instr is not None and instr.reason == "time_stop"


def test_impact_resumed_exit_for_reversal():
    pm = PositionManager()
    entry(pm)  # short vs up-move; move_dir = +1
    f = fv(impact_decay=-0.5, flow_imbalance=0.5, microprice_bps=0.0)
    instr = pm.check_exits(q("MU", 10, 100.0, 100.02), f, T0 + timedelta(seconds=10), 300)
    assert instr is not None and instr.reason == "impact_resumed"


def test_closed_trade_pnl():
    pm = PositionManager()
    entry(pm)
    closed = pm.on_exit_fill(Fill("MU", Side.BUY, 100, 99.60, T0 + timedelta(seconds=60),
                                  "c2", "trace1"), "target")
    assert closed is not None
    assert abs(closed.pnl - 40.0) < 1e-9
    assert pm.positions == {}
