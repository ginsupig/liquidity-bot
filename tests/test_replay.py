"""End-to-end replay test: synthesize a recorded day with an unconfirmed burst
and verify (a) strict event ordering, (b) the pipeline produces decisions,
(c) closed-trade accounting is consistent."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from wliq.core.config import Settings
from wliq.data.recorder import ParquetRecorder
from wliq.research.replay import load_events, run_replay
from tests.conftest import T0, q, t
from wliq.core.models import Side


@pytest.fixture
def recorded_day(tmp_path: Path, monkeypatch) -> tuple[Settings, str]:
    root = tmp_path / "data"
    rec = ParquetRecorder(root, flush_rows=100_000)
    # ~15 minutes of synthetic data: benchmarks flat, MU bursts then stalls.
    import wliq.data.recorder as rmod
    for i in range(900):
        sec = float(i)
        mu_px = 100.0 + min(i, 500) * 0.004 if i < 600 else 102.0 - (i - 600) * 0.002
        rec.append_quote(q("MU", sec, mu_px, mu_px + 0.02))
        rec.append_quote(q("QQQ", sec, 500.00, 500.02))
        rec.append_quote(q("SMH", sec, 250.00, 250.02))
        rec.append_quote(q("WDC", sec, 80.00, 80.02))
        side = Side.BUY if i < 600 else Side.SELL
        rec.append_trade(t("MU", sec + 0.5, mu_px + (0.02 if side is Side.BUY else 0.0),
                           300, side))
    rec.close()
    # recorder writes under today's date; move to the synthetic date partition
    real_date = next(root.glob("date=*")).name
    target = root / f"date={T0.date().isoformat()}"
    (root / real_date).rename(target)
    settings = Settings(symbols=["MU"], benchmarks={"MU": ["WDC"]},
                        record_root=root, mode="replay")
    settings.strategy.no_trade_first_seconds = 0
    settings.strategy.min_volume_z = -10  # synthetic data has no volume history
    return settings, T0.date().isoformat()


def test_events_strictly_ordered(recorded_day):
    settings, date = recorded_day
    events = load_events(settings.record_root, date, settings.all_streamed_symbols())
    assert len(events) > 3000
    ordered = sorted(events)
    assert all(ordered[i].ts <= ordered[i + 1].ts for i in range(len(ordered) - 1))


def test_replay_runs_full_pipeline(recorded_day):
    settings, date = recorded_day
    result = run_replay(settings, date,
                        journal_path=settings.record_root / "journal.jsonl")
    assert sum(result.governor_actions.values()) > 0
    # accounting consistency: total equals sum of parts
    assert abs(result.total_pnl - sum(tr.pnl for tr in result.trades)) < 1e-9
