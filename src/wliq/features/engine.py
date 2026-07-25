"""Online, bounded-memory microstructure feature engine.

Consumes normalized Quote / Trade / DepthSnapshot events, maintains per-symbol
rolling state, and emits FeatureVector snapshots. All state is windowed;
nothing grows without bound. No look-ahead: features at time t use only events
with ts <= t.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from wliq.core.models import DepthSnapshot, FeatureVector, Quote, Side, Trade
from wliq.features.residual import ResidualModel
from wliq.features.dynamics import (ConfirmationTimer, ResidualDynamics,
                                    replenishment_confidence, tod_bucket,
                                    vol_regime)
from wliq.features.volume import VolumeTracker


class FeatureEngine:
    def __init__(
        self,
        window_seconds: int,
        benchmarks: dict[str, list[str]],
        market_etf: str = "QQQ",
        sector_etf: str = "SMH",
        residual_halflife_seconds: int = 300,
        volume_tracker: VolumeTracker | None = None,
        session_open: str = "09:30",
        session_tz: str = "America/New_York",
    ) -> None:
        self.window = timedelta(seconds=window_seconds)
        self.benchmarks = benchmarks
        self.market_etf = market_etf
        self.sector_etf = sector_etf
        self.residual = ResidualModel(halflife_seconds=residual_halflife_seconds)
        self.volume = volume_tracker or VolumeTracker(None)
        self.tz = ZoneInfo(session_tz)
        h, m = session_open.split(":")
        self.open_time = dtime(int(h), int(m))

        self.quotes: dict[str, deque[Quote]] = defaultdict(deque)
        self.trades: dict[str, deque[Trade]] = defaultdict(deque)
        self.depth: dict[str, DepthSnapshot] = {}
        self._prev_depth: dict[str, DepthSnapshot] = {}
        self._spread_hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1800))
        self._sec_samples: dict[str, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=1200))
        self._last_sec: dict[str, int] = {}
        # windowed liquidity flow accumulators (timestamped for correct trimming)
        self._consumed: dict[str, deque[tuple[datetime, float, float]]] = defaultdict(deque)  # ts, bid, ask
        self._restored: dict[str, deque[tuple[datetime, float, float]]] = defaultdict(deque)
        self._dyn: dict[str, ResidualDynamics] = defaultdict(ResidualDynamics)
        self._confirm: dict[str, ConfirmationTimer] = defaultdict(ConfirmationTimer)
        self._impulse_active: dict[str, bool] = defaultdict(bool)
        self.vol_regime_low_bps: float = 8.0
        self.vol_regime_high_bps: float = 45.0
        self.data_confidence_fn = None   # optional hook -> float per symbol
        self.noii_pressure: dict[str, float] = defaultdict(float)
        self._vwap_num: dict[str, float] = defaultdict(float)
        self._vwap_den: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ events
    def on_quote(self, quote: Quote) -> None:
        q = self.quotes[quote.symbol]
        if q:
            prev = q[-1]
            rb = max(0.0, quote.bid_size - prev.bid_size) if quote.bid >= prev.bid else quote.bid_size
            ra = max(0.0, quote.ask_size - prev.ask_size) if quote.ask <= prev.ask else quote.ask_size
            if rb or ra:
                self._restored[quote.symbol].append((quote.ts, rb, ra))
        q.append(quote)
        self._spread_hist[quote.symbol].append(quote.spread_bps)
        sec = int(quote.ts.timestamp())
        if self._last_sec.get(quote.symbol) != sec:
            self._sec_samples[quote.symbol].append((sec, quote.mid))
            self._last_sec[quote.symbol] = sec
            self._maybe_update_residual(quote.symbol)
        self._trim(quote.symbol, quote.ts)

    def on_trade(self, trade: Trade) -> None:
        t = self._classify(trade)
        self.trades[t.symbol].append(t)
        if t.aggressor is Side.SELL:
            self._consumed[t.symbol].append((t.ts, t.size, 0.0))
        elif t.aggressor is Side.BUY:
            self._consumed[t.symbol].append((t.ts, 0.0, t.size))
        self._vwap_num[t.symbol] += t.price * t.size
        self._vwap_den[t.symbol] += t.size
        self.volume.on_trade(t.symbol, t.ts, t.size)
        self._trim(t.symbol, t.ts)

    def on_depth(self, snap: DepthSnapshot) -> None:
        self._prev_depth[snap.symbol] = self.depth.get(snap.symbol, snap)
        self.depth[snap.symbol] = snap

    def _classify(self, trade: Trade) -> Trade:
        """Quote-relative aggressor classification when the feed omits direction."""
        if trade.aggressor is not None:
            return trade
        q = self.quotes[trade.symbol]
        if not q:
            return trade
        ref = q[-1]
        if trade.price >= ref.ask - 1e-9:
            side = Side.BUY
        elif trade.price <= ref.bid + 1e-9:
            side = Side.SELL
        elif trade.price > ref.mid:
            side = Side.BUY
        elif trade.price < ref.mid:
            side = Side.SELL
        else:
            return trade
        return Trade(trade.symbol, trade.ts, trade.price, trade.size, side,
                     trade.session, trade.seq, trade.recv_ts)

    def _trim(self, symbol: str, now: datetime) -> None:
        cutoff = now - self.window
        for dq in (self.quotes[symbol], self.trades[symbol]):
            while dq and dq[0].ts < cutoff:
                dq.popleft()
        for dq in (self._consumed[symbol], self._restored[symbol]):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    # -------------------------------------------------------------- residual
    def _sec_return(self, symbol: str, lookback: int = 1) -> float | None:
        lookback = max(1, lookback)
        s = self._sec_samples[symbol]
        if len(s) < lookback + 1:
            return None
        newer, older = s[-1][1], s[-1 - lookback][1]
        if older <= 0:
            return None
        return math.log(newer / older)

    def _peer_return(self, symbol: str, lookback: int = 1) -> float:
        rets = [r for p in self.benchmarks.get(symbol, [])
                if (r := self._sec_return(p, lookback)) is not None]
        return float(np.mean(rets)) if rets else 0.0

    def _maybe_update_residual(self, symbol: str) -> None:
        if symbol in (self.market_etf, self.sector_etf):
            return
        sr = self._sec_return(symbol)
        mr = self._sec_return(self.market_etf)
        er = self._sec_return(self.sector_etf)
        if sr is None or mr is None:
            return
        self.residual.update(symbol, sr, mr, er if er is not None else mr,
                             self._peer_return(symbol))

    # --------------------------------------------------------------- compute
    def minutes_since_open(self, now: datetime) -> float:
        local = now.astimezone(self.tz)
        open_dt = local.replace(hour=self.open_time.hour, minute=self.open_time.minute,
                                second=0, microsecond=0)
        return (local - open_dt).total_seconds() / 60.0

    def _window_return(self, symbol: str) -> float:
        q = self.quotes[symbol]
        if len(q) < 2 or q[0].mid <= 0:
            return 0.0
        return math.log(q[-1].mid / q[0].mid)

    def compute(self, symbol: str, now: datetime) -> FeatureVector | None:
        q = self.quotes[symbol]
        if not q:
            return None
        latest = q[-1]
        mid = latest.mid
        if mid <= 0:
            return None

        n_window = max(1, int(self.window.total_seconds()))
        own = self._window_return(symbol)
        mkt = self._sec_return(self.market_etf, min(n_window, len(self._sec_samples[self.market_etf]) - 1) or 1) or 0.0
        sec = self._sec_return(self.sector_etf, min(n_window, len(self._sec_samples[self.sector_etf]) - 1) or 1) or 0.0
        peer = self._peer_return(symbol, min(n_window, 60))
        residual = self.residual.residual(symbol, own, mkt, sec, peer)

        # z-scale from realized 1s vol aggregated to window horizon
        s = self._sec_samples[symbol]
        realized_vol_bps = 0.0
        residual_z = 0.0
        if len(s) >= 30:
            mids = np.array([x[1] for x in s], dtype=float)
            rets = np.diff(np.log(mids))
            sd1 = float(np.std(rets))
            realized_vol_bps = 10_000 * sd1 * math.sqrt(len(rets))
            scale = sd1 * math.sqrt(n_window)
            if scale > 1e-9:
                residual_z = residual / scale

        # flow / impact
        ts = self.trades[symbol]
        buy = sum(t.size for t in ts if t.aggressor is Side.BUY)
        sell = sum(t.size for t in ts if t.aggressor is Side.SELL)
        total = buy + sell
        flow = (buy - sell) / total if total else 0.0

        impact_decay = 0.0
        if len(ts) >= 8 and len(q) >= 4:
            half = len(ts) // 2
            early, late = list(ts)[:half], list(ts)[half:]
            def _impact(trades: list[Trade], p0: float, p1: float) -> float:
                sv = abs(sum(t.size * (1 if t.aggressor is Side.BUY else -1)
                             for t in trades if t.aggressor))
                return abs(math.log(p1 / p0)) / max(sv, 1.0) if p0 > 0 else 0.0
            qmid = len(q) // 2
            e = _impact(early, q[0].mid, q[qmid].mid)
            l = _impact(late, q[qmid].mid, q[-1].mid)
            if e > 1e-15:
                impact_decay = max(-1.0, min(1.0, (e - l) / e))

        # replenishment skew (windowed, ratio of restored to consumed per side)
        cb = sum(x[1] for x in self._consumed[symbol])
        ca = sum(x[2] for x in self._consumed[symbol])
        rb = sum(x[1] for x in self._restored[symbol])
        ra = sum(x[2] for x in self._restored[symbol])
        replen = max(-3.0, min(3.0, rb / max(cb, 1.0) - ra / max(ca, 1.0)))

        micro_bps = 10_000 * (latest.microprice - mid) / mid
        peer_conf = float(np.sign(own) * peer / max(abs(own), 1e-9)) if own else 0.0
        peer_conf = max(-2.0, min(2.0, peer_conf))

        # L2 features (zero-neutral when depth unavailable)
        depth_imb = near_touch = queue_dep = 0.0
        d = self.depth.get(symbol)
        if d and d.bids and d.asks:
            db, da = d.depth_shares(Side.BUY), d.depth_shares(Side.SELL)
            if db + da > 0:
                depth_imb = (db - da) / (db + da)
            touch = d.bids[0].size + d.asks[0].size
            near_touch = touch / max(db + da, 1.0)
            prev = self._prev_depth.get(symbol)
            if prev and prev.bids and prev.asks:
                dt = max((d.ts - prev.ts).total_seconds(), 1e-3)
                bid_dep = (prev.bids[0].size - d.bids[0].size) / dt if d.bids[0].price >= prev.bids[0].price else 0.0
                ask_dep = (prev.asks[0].size - d.asks[0].size) / dt if d.asks[0].price <= prev.asks[0].price else 0.0
                denom = max(prev.bids[0].size + prev.asks[0].size, 1.0)
                queue_dep = max(-5.0, min(5.0, (ask_dep - bid_dep) / denom))

        sh = self._spread_hist[symbol]
        spread_pctile = float(np.mean(np.asarray(sh) <= latest.spread_bps)) if len(sh) >= 30 else 0.5

        msu = self.minutes_since_open(now)
        vwap = self._vwap_num[symbol] / self._vwap_den[symbol] if self._vwap_den[symbol] > 0 else mid

        # ---- v2 dynamics ----
        vel, acc, rv_z = self._dyn[symbol].update(now, residual)
        ct = self._confirm[symbol]
        impulse_now = abs(residual_z) >= 2.0
        if impulse_now and not self._impulse_active[symbol]:
            ct.start(now)
        self._impulse_active[symbol] = impulse_now
        sec_conf = float(np.sign(own) * sec / max(abs(own), 1e-9)) if own else 0.0
        ct.update(now, peer_conf, max(-2.0, min(2.0, sec_conf)))
        repl_conf = replenishment_confidence(len(self._consumed[symbol]),
                                             len(self._restored[symbol]))
        dconf = (self.data_confidence_fn(symbol)
                 if self.data_confidence_fn else 1.0)

        return FeatureVector(
            symbol=symbol,
            ts=now,
            mid=mid,
            spread_bps=latest.spread_bps,
            spread_pctile=spread_pctile,
            residual_return=residual,
            residual_z=residual_z,
            flow_imbalance=flow,
            impact_decay=impact_decay,
            replenishment_skew=replen,
            microprice_bps=micro_bps,
            peer_confirmation=peer_conf,
            depth_imbalance=depth_imb,
            near_touch_ratio=near_touch,
            queue_depletion=queue_dep,
            volume_z=self.volume.volume_z(symbol, msu),
            vwap_distance_bps=10_000 * (mid - vwap) / vwap if vwap > 0 else 0.0,
            realized_vol_bps=realized_vol_bps,
            minutes_since_open=msu,
            event_count=len(q) + len(ts),
            residual_velocity=vel,
            residual_acceleration=acc,
            residual_vol_z=rv_z,
            peer_confirm_latency_s=ct.peer_latency,
            sector_confirm_latency_s=ct.sector_latency,
            replenishment_confidence=repl_conf,
            vol_regime=vol_regime(realized_vol_bps, self.vol_regime_low_bps,
                                  self.vol_regime_high_bps),
            tod_bucket=tod_bucket(msu),
            noii_pressure=self.noii_pressure[symbol],
            data_confidence=dconf,
        )
