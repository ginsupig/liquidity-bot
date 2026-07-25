"""v2 success-criteria tests: cumulative partial-exit PnL, deterministic
replay, restart recovery through the real SessionEngine."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wliq.core.models import Fill, GovernorAction, Quote, Side, Signal
from wliq.execution.positions import PositionManager

T0 = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)


def test_partial_exit_pnl_is_cumulative():
    pm = PositionManager(max_spread_bps_exit=50)
    sig = Signal("MU", T0, Side.SELL, GovernorAction.REVERSAL, 4.0,
                 100.0, 100.5, 99.0, "tr")
    pm.on_entry_fill(Fill("MU", Side.SELL, 100, 100.0, T0, "e1", "tr"), sig)
    # scale out in three partials at improving prices
    assert pm.on_exit_fill(Fill("MU", Side.BUY, 40, 99.80, T0, "x1", "tr"),
                           "target") is None
    assert pm.on_exit_fill(Fill("MU", Side.BUY, 30, 99.60, T0, "x2", "tr"),
                           "target") is None
    trade = pm.on_exit_fill(Fill("MU", Side.BUY, 30, 99.40, T0, "x3", "tr"),
                            "target")
    assert trade is not None
    # total pnl = 40*.20 + 30*.40 + 30*.60 = 8 + 12 + 18 = 38
    assert abs(trade.pnl - 38.0) < 1e-9
    assert trade.quantity == 100
    exit_vwap = (40 * 99.80 + 30 * 99.60 + 30 * 99.40) / 100
    assert abs(trade.exit_price - exit_vwap) < 1e-9


def _make_day(root: Path):
    from tests.conftest import T0 as TQ, q, t
    from wliq.core.config import Settings
    from wliq.data.recorder import ParquetRecorder
    rec = ParquetRecorder(root, flush_rows=100_000)
    for i in range(900):
        sec = float(i)
        mu = 100.0 + min(i, 500) * 0.004 if i < 600 else 102.0 - (i - 600) * 0.002
        rec.append_quote(q("MU", sec, mu, mu + 0.02))
        rec.append_quote(q("QQQ", sec, 500.00, 500.02))
        rec.append_quote(q("SMH", sec, 250.00, 250.02))
        rec.append_quote(q("WDC", sec, 80.00, 80.02))
        side = Side.BUY if i < 600 else Side.SELL
        rec.append_trade(t("MU", sec + 0.5,
                           mu + (0.02 if side is Side.BUY else 0.0), 300, side))
    rec.close()
    real = next(root.glob("date=*")).name
    (root / real).rename(root / f"date={TQ.date().isoformat()}")
    s = Settings(symbols=["MU"], benchmarks={"MU": ["WDC"]},
                 record_root=root, mode="replay")
    s.strategy.no_trade_first_seconds = 0
    s.strategy.min_volume_z = -10
    return s, TQ.date().isoformat()


def _replay(tmp_path, tag):
    from wliq.research.replay import run_replay
    root = tmp_path / f"rec-{tag}"
    s, date = _make_day(root)
    return run_replay(s, date, logs_dir=tmp_path / f"logs-{tag}")


def test_replay_is_deterministic(tmp_path):
    r1 = _replay(tmp_path, "a")
    r2 = _replay(tmp_path, "b")
    t1 = [(t.symbol, t.quantity, round(t.entry_price, 6),
           round(t.exit_price, 6), round(t.pnl, 6), t.exit_reason)
          for t in r1.trades]
    t2 = [(t.symbol, t.quantity, round(t.entry_price, 6),
           round(t.exit_price, 6), round(t.pnl, 6), t.exit_reason)
          for t in r2.trades]
    assert t1 == t2
    assert r1.governor_actions == r2.governor_actions
    assert r1.signals == r2.signals


def test_session_recovery_roundtrip(tmp_path):
    """Write position events + ledger, boot a fresh SessionEngine, verify it
    rebuilds the book and risk state."""
    from wliq.core.clock import SimClock
    from wliq.execution.ledger import PositionEventLog, RiskLedger
    from wliq.runtime import SessionEngine

    root = tmp_path / "rec"
    s, _date = _make_day(root)
    logs = tmp_path / "logs"
    PositionEventLog(logs / "position_events.jsonl").append_fill(
        "entry", Fill("MU", Side.SELL, 25, 100.0, T0, "e1", "tr"),
        {"stop": 100.6, "target": 99.1, "action": "REVERSAL"})
    RiskLedger(logs / "risk_ledger.jsonl").append("trade_closed", pnl=-11.5)
    eng = SessionEngine(s, SimClock(T0), live_adapter=None, logs_dir=logs)
    assert eng.recover()
    assert "MU" in eng.pm.positions and eng.pm.positions["MU"].quantity == 25
    assert abs(eng.risk.realized_pnl - (-11.5)) < 1e-9
    assert abs(eng.risk2.realized - (-11.5)) < 1e-9
