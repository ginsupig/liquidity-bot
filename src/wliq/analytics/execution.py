"""Sprint 4 — execution quality analytics.

Per-fill markouts against subsequent mids give:
  * realized spread  : 2*side*(fill_px - mid_{t+Δ})   (captured spread minus
                       adverse move; positive = we earned the spread)
  * effective spread : 2*side*(fill_px - mid_at_fill)
  * adverse selection: effective - realized = 2*side*(mid_{t+Δ} - mid_at_fill)
  * implementation shortfall: side*(fill_px - decision_px) + unfilled penalty
All in bps of the reference mid, side-signed so positive = cost to us for
IS/adverse, positive = earned for realized spread.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wliq.core.models import Fill, Side


@dataclass
class FillRecord:
    fill: Fill
    mid_at_fill: float
    decision_price: float
    decision_ts: datetime


class ExecutionAnalytics:
    HORIZONS_S = (1.0, 5.0, 30.0)

    def __init__(self) -> None:
        self.records: list[FillRecord] = []
        self._mids: dict[str, list[tuple[datetime, float]]] = {}

    def on_mid(self, symbol: str, ts: datetime, mid: float) -> None:
        self._mids.setdefault(symbol, []).append((ts, mid))

    def on_fill(self, fill: Fill, mid_at_fill: float, decision_price: float,
                decision_ts: datetime) -> None:
        self.records.append(FillRecord(fill, mid_at_fill, decision_price,
                                       decision_ts))

    def _mid_after(self, symbol: str, ts: datetime, horizon_s: float) -> float | None:
        arr = self._mids.get(symbol, [])
        target = ts + timedelta(seconds=horizon_s)
        i = bisect_right([a[0] for a in arr], target)
        if i >= len(arr):
            return arr[-1][1] if arr else None
        return arr[i][1]

    def summary(self) -> dict:
        if not self.records:
            return {"n_fills": 0}
        out: dict = {"n_fills": len(self.records)}
        eff, shortfall = [], []
        realized: dict[float, list[float]] = {h: [] for h in self.HORIZONS_S}
        adverse: dict[float, list[float]] = {h: [] for h in self.HORIZONS_S}
        for r in self.records:
            sgn = 1.0 if r.fill.side is Side.BUY else -1.0
            ref = max(r.mid_at_fill, 1e-9)
            eff.append(2 * sgn * (r.fill.price - r.mid_at_fill) / ref * 1e4)
            shortfall.append(sgn * (r.fill.price - r.decision_price)
                             / max(r.decision_price, 1e-9) * 1e4)
            for h in self.HORIZONS_S:
                m = self._mid_after(r.fill.symbol, r.fill.ts, h)
                if m is None:
                    continue
                realized[h].append(2 * sgn * (r.fill.price - m) / ref * 1e4)
                adverse[h].append(2 * sgn * (m - r.mid_at_fill) / ref * 1e4)
        _avg = lambda xs: sum(xs) / len(xs) if xs else None
        out["effective_spread_bps"] = _avg(eff)
        out["implementation_shortfall_bps"] = _avg(shortfall)
        for h in self.HORIZONS_S:
            out[f"realized_spread_{int(h)}s_bps"] = _avg(realized[h])
            out[f"adverse_selection_{int(h)}s_bps"] = _avg(adverse[h])
        return out
