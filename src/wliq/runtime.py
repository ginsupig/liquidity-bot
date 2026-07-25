"""SessionEngine v2 — ONE pipeline for replay, paper, and live.

Event flow per market-data event:
  quality gate -> recorder -> feature engine -> venue sim (paper) ->
  OM fill/ack processing -> position exits -> FSM step -> risk gates ->
  adaptive execution plan -> OM create/submit -> journals/OUFS

The ONLY differences between modes:
  * Clock: SimClock (replay) vs WallClock (paper/live)
  * Venue: PaperVenue vs live WebullAdapter
  * Live additionally requires the config triple-interlock, startup recovery +
    reconciliation, and a periodic REST reconciliation loop.
"""
from __future__ import annotations

import signal as _signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

from wliq.analytics.execution import ExecutionAnalytics
from wliq.broker.health import DataHealthMonitor
from wliq.core.clock import Clock, SimClock, WallClock
from wliq.core.config import Settings
from wliq.core.journal import OrderJournal
from wliq.core.logging import get_logger
from wliq.core.models import (DepthSnapshot, FeatureVector, Noii, OrderRequest,
                              Quote, Side, Signal, Trade)
from wliq.core.oufs import OufsWriter
from wliq.data.quality import DataQualityMonitor, RawPacketRecorder
from wliq.data.recorder import ParquetRecorder
from wliq.execution.adaptive import plan_entry
from wliq.execution.ledger import (PositionEventLog, RiskLedger,
                                   rebuild_open_orders)
from wliq.execution.order_manager import (ManagedOrder, OrderManager, OState,
                                          TERMINAL, IllegalTransition)
from wliq.execution.positions import PositionManager
from wliq.execution.reconciler import reconcile
from wliq.execution.risk import RiskManager
from wliq.execution.risk2 import PortfolioRisk, PortfolioRiskConfig
from wliq.execution.venue_paper import PaperVenue, VenueEvent
from wliq.features.engine import FeatureEngine
from wliq.features.volume import VolumeProfile, VolumeTracker
from wliq.ml.regime import AnomalyDetector, classify_regime
from wliq.ml.store import FeatureStore
from wliq.strategy.exhaustion import ResidualExhaustionStrategy
from wliq.strategy.fsm import FsmConfig, SignalStateMachine, SState
from wliq.strategy.governor import StrategyGovernor

log = get_logger("runtime")


class SessionEngine:
    def __init__(self, settings: Settings, clock: Clock,
                 live_adapter=None, logs_dir: Path = Path("logs")) -> None:
        s = settings
        self.s = s
        self.clock = clock
        self.live = live_adapter is not None
        self.adapter = live_adapter

        profile = None
        if s.volume_model.profile_path.exists():
            profile = VolumeProfile.load(s.volume_model.profile_path,
                                         s.volume_model.bucket_minutes)
        self.engine = FeatureEngine(
            s.feature_window_seconds, s.benchmarks, s.market_etf, s.sector_etf,
            s.residual_halflife_seconds, VolumeTracker(profile),
            s.session_open, s.session_tz)
        self.health = DataHealthMonitor(s.risk.stale_data_ms,
                                        s.governor.max_sequence_gap_tolerance)
        self.quality = DataQualityMonitor()
        self.engine.data_confidence_fn = self.quality.confidence
        self.raw = RawPacketRecorder(s.record_root, enabled=self.live)
        self.recorder = ParquetRecorder(s.record_root, s.recorder.flush_rows,
                                        s.recorder.flush_seconds,
                                        s.recorder.dedupe)
        self.governor = StrategyGovernor(s.governor, s.strategy, self.health)
        self.scorer = ResidualExhaustionStrategy(s.strategy)
        self.fsm = SignalStateMachine(FsmConfig(impulse_z=s.strategy.residual_z_entry),
                                      s.strategy)
        self.anomaly = AnomalyDetector()
        self.risk = RiskManager(s.risk)
        self.oufs = OufsWriter(logs_dir / "oufs")
        self.journal = OrderJournal(logs_dir / "orders.jsonl")
        self.om = OrderManager(self.clock, self.journal, self.oufs,
                               max_gross_notional_usd=s.risk.max_gross_exposure_usd)
        self.om.on_terminal = self._on_order_terminal
        self.venue = PaperVenue(s.execution.paper, self.clock)
        self.pm = PositionManager(max_spread_bps_exit=s.risk.max_spread_bps * 2)
        self.ledger = RiskLedger(logs_dir / "risk_ledger.jsonl")
        self.pos_log = PositionEventLog(logs_dir / "position_events.jsonl")
        self.risk2 = PortfolioRisk(PortfolioRiskConfig(
            max_gross_exposure_usd=s.risk.max_gross_exposure_usd,
            sector_map=getattr(s, "sector_map", {}) or {},
            clusters=[[sym] + peers for sym, peers in s.benchmarks.items()]),
            self.ledger)
        self.xq = ExecutionAnalytics()
        self.fstore = FeatureStore(Path(s.record_root) / "feature_store")
        self._pending_sig: dict[str, Signal] = {}
        self._pending_exit: dict[str, str] = {}
        self._decision_px: dict[str, tuple[float, datetime]] = {}
        self._trade_rate: dict[str, float] = {}
        self._last_recon: datetime | None = None
        self._last_beat: datetime | None = None
        self._last_eval: dict[str, datetime] = {}
        self.emergency = False

    # ---------------- recovery ----------------
    def recover(self) -> bool:
        """Rebuild positions/orders/risk from journals; reconcile if live.
        Returns False (blocking) on any live mismatch."""
        for pos in self.pos_log.rebuild_positions().values():
            self.pm.positions[pos.symbol] = pos
            self.risk.positions[pos.symbol] = pos
        for mo in rebuild_open_orders(self.journal.path):
            self.om.restore(mo)
        self.risk2.restore()
        day = self.ledger.load_day()
        self.risk.realized_pnl = day["realized_pnl"]
        self.risk.consecutive_losses = day["consecutive_losses"]
        log.info("recovered", positions=list(self.pm.positions),
                 open_orders=len(self.om.active_orders()),
                 realized=day["realized_pnl"])
        if self.live:
            result = reconcile(self.pm.positions, self.adapter)
            if not result.ok:
                log.error("recovery_reconcile_failed", issues=result.issues)
                return False
        return True

    # ---------------- venue event processing ----------------
    def _apply_venue_events(self, events: list[VenueEvent]) -> None:
        for ev in events:
            if ev.kind == "ack":
                self.om.on_ack(ev.coid)
            elif ev.kind == "cancel_ack":
                mo = self.om.orders.get(ev.coid)
                if mo and mo.state is not OState.CANCEL_PENDING and mo.state not in TERMINAL:
                    if mo.state in (OState.ACKNOWLEDGED, OState.PARTIAL):
                        self.om.request_cancel(ev.coid)
                self.om.on_cancel_ack(ev.coid)
            elif ev.kind == "reject":
                self.om.on_reject(ev.coid, ev.reason)
            elif ev.kind == "fill" and ev.fill is not None:
                self._on_fill(ev.fill)

    def _on_fill(self, fill) -> None:
        mo = self.om.on_fill(fill)
        mid, dts = self._decision_px.get(fill.client_order_id, (fill.price,
                                                                fill.ts))
        self.xq.on_fill(fill, mid_at_fill=mid, decision_price=mid,
                        decision_ts=dts)
        if mo.request.intent == "entry":
            sig = self._pending_sig.get(fill.client_order_id)
            if sig is not None:
                self.pm.on_entry_fill(fill, sig)
                self.risk.positions[fill.symbol] = self.pm.positions[fill.symbol]
                self.pos_log.append_fill("entry", fill, {
                    "stop": sig.stop_price, "target": sig.target_price,
                    "action": sig.action.value})
                self.fsm.on_entry_fill(fill.symbol, fill.ts)
        else:
            reason = self._pending_exit.get(fill.client_order_id, "exit")
            self.pos_log.append_fill("exit", fill)
            closed = self.pm.on_exit_fill(fill, reason)
            if closed is not None:
                self.risk.on_trade_closed(closed.symbol, closed.pnl,
                                          self.clock.now())
                self.risk2.on_trade_closed(closed.pnl)
                self.ledger.append("trade_closed", symbol=closed.symbol,
                                   pnl=closed.pnl, reason=closed.exit_reason,
                                   trace_id=closed.trace_id)
                self.oufs.outcome("trade_closed", symbol=closed.symbol,
                                  pnl=closed.pnl, reason=closed.exit_reason)
                self.fsm.on_position_closed(closed.symbol, self.clock.now())
                log.info("trade_closed", symbol=closed.symbol,
                         pnl=round(closed.pnl, 2), reason=closed.exit_reason)

    def _on_order_terminal(self, mo: ManagedOrder) -> None:
        if (mo.request.intent == "entry" and mo.filled_qty <= 1e-9
                and mo.state in (OState.CANCELLED, OState.EXPIRED,
                                 OState.REJECTED)):
            self.fsm.on_entry_failed(mo.request.symbol, self.clock.now())
        self._pending_sig.pop(mo.request.client_order_id, None)
        self._pending_exit.pop(mo.request.client_order_id, None)

    # ---------------- periodic (clock-driven, event-independent) ----------
    def on_tick(self) -> None:
        now = self.clock.now()
        for coid in self.om.poll():                    # timer cancels
            (self.adapter.cancel_order(coid) if self.live
             else self.venue.cancel(coid))
        self._apply_venue_events(self.venue.poll())    # acks / cancel acks
        if self.live and (self._last_recon is None
                          or (now - self._last_recon).total_seconds()
                          >= self.s.reconcile_interval_s):
            self._last_recon = now
            result = reconcile(self.pm.positions, self.adapter)
            if not result.ok:
                log.error("periodic_reconcile_failed", issues=result.issues)
                self.risk.cfg.kill_switch = True       # fail closed
                self.ledger.append("halt", reason="reconcile_mismatch")
        if (self._last_beat is None
                or (now - self._last_beat).total_seconds() >= 30):
            self._last_beat = now
            self.oufs.health(
                positions=len(self.pm.positions),
                open_orders=len(self.om.active_orders()),
                realized=self.risk.realized_pnl,
                halted=self.risk2.halted,
                reserved_notional=round(self.om.reserved_notional, 2),
                quality={k: round(v["confidence"], 2)
                         for k, v in self.quality.snapshot().items()})

    # ---------------- market data handlers ----------------
    def on_quote(self, q: Quote, raw: bytes | None = None) -> None:
        if raw:
            self.raw.record("quotes", raw)
        if not self.quality.on_quote(q, arrival=self.clock.now()):
            return                                     # duplicate — drop
        self.recorder.append_quote(q)
        self.engine.on_quote(q)
        self.health.on_event(q.symbol, q.ts)
        self.xq.on_mid(q.symbol, q.ts, q.mid)
        if not self.live:
            self._apply_venue_events(self.venue.on_quote(q))
        self.on_tick()
        if q.symbol in self.s.symbols:
            self._evaluate(q)

    def on_trade(self, t: Trade, raw: bytes | None = None) -> None:
        if raw:
            self.raw.record("trades", raw)
        if not self.quality.on_trade(t, arrival=self.clock.now()):
            return
        self.recorder.append_trade(t)
        self.engine.on_trade(t)
        r = self._trade_rate.get(t.symbol, t.size)
        self._trade_rate[t.symbol] = 0.95 * r + 0.05 * t.size
        if not self.live:
            self._apply_venue_events(self.venue.on_trade(t))
        self.on_tick()

    def on_depth(self, d: DepthSnapshot, raw: bytes | None = None) -> None:
        if raw:
            self.raw.record("depth", raw)
        if self.quality.on_depth(d):
            self.recorder.append_depth(d)
            self.engine.on_depth(d)

    def on_noii(self, n: Noii, raw: bytes | None = None) -> None:
        self.recorder.append_noii(n)
        imb = getattr(n, "imbalance_shares", 0.0) or 0.0
        side = getattr(n, "imbalance_side", "") or ""
        self.engine.noii_pressure[n.symbol] = (imb if side.upper().startswith("B")
                                               else -imb)

    # ---------------- decision core ----------------
    def _evaluate(self, q: Quote) -> None:
        now = self.clock.now()
        last = self._last_eval.get(q.symbol)
        if last and (now - last).total_seconds() * 1000 < self.s.signal_interval_ms:
            return
        self._last_eval[q.symbol] = now
        f = self.engine.compute(q.symbol, q.ts)
        if f is None:
            return
        # exits first, always
        instr = self.pm.check_exits(q, f, now, self.s.strategy.max_holding_seconds)
        if instr is not None or self.emergency:
            self._submit_exit(q, instr)
            return
        flags = self.anomaly.check(f)
        if flags:
            self.oufs.decision("anomaly_block", symbol=f.symbol, flags=flags)
            return
        if f.data_confidence < self.s.min_data_confidence:
            return
        sector_ret = self.engine._sec_return(self.s.sector_etf, 60) or 0.0
        self.governor.update_context(market_breadth=0.0,
                                     sector_momentum=sector_ret)
        decision = self.governor.decide(f, now)
        self.fstore.append(f, context=self.fsm.state(q.symbol).value)
        sig = self.fsm.step(f, decision.action)
        self.oufs.decision("evaluate", symbol=f.symbol,
                           governor=decision.action.value,
                           fsm=self.fsm.state(q.symbol).value,
                           residual_z=round(f.residual_z, 2))
        if sig is None:
            return
        self._submit_entry(sig, f, q)

    def _submit_entry(self, sig: Signal, f: FeatureVector, q: Quote) -> None:
        now = self.clock.now()
        ok, reason = self.risk.approve(sig, f, now=now)
        if ok:
            gross = sum(abs(p.quantity * p.entry_price)
                        for p in self.pm.positions.values())
            net = sum(p.side.sign * p.quantity * p.entry_price
                      for p in self.pm.positions.values())
            ok, reason = self.risk2.approve(sig, f, self.pm.positions, gross,
                                            net, self.om.reserved_notional)
        if ok and self.om.has_active_entry(sig.symbol):
            ok, reason = False, "active_entry_intent"
        self.oufs.decision("signal", symbol=sig.symbol, side=sig.side.value,
                           score=round(sig.score, 2), approved=ok,
                           reason=reason, trace_id=sig.trace_id)
        if not ok:
            self.fsm.on_entry_failed(sig.symbol, now)
            return
        qty = int(self.risk.size(sig) * self.risk2.size_multiplier(f))
        if qty <= 0:
            self.fsm.on_entry_failed(sig.symbol, now)
            return
        urgency = min(1.0, sig.score / 6.0)
        plan = plan_entry(f, q, sig.side, urgency,
                          self._trade_rate.get(sig.symbol, 100.0))
        coid = self.journal.next_client_order_id(sig.symbol, "entry", now)
        if self.journal.is_duplicate(coid):
            return
        req = OrderRequest(sig.symbol, sig.side, qty,
                           "LIMIT" if plan.limit_price else "MARKET",
                           plan.limit_price, coid, sig.trace_id, "entry",
                           {"score": sig.score, "style": plan.style})
        try:
            self.om.create(req, ref_price=q.mid,
                           risk_usd=self.s.risk.risk_per_trade_usd,
                           ttl_ms=plan.ttl_ms)
        except IllegalTransition as e:
            log.warning("om_reject", err=str(e))
            self.fsm.on_entry_failed(sig.symbol, now)
            return
        self._pending_sig[coid] = sig
        self._decision_px[coid] = (q.mid, now)
        self.om.mark_submitting(coid)
        if self.live:
            self.adapter.place_order(req)
        else:
            self._apply_venue_events(self.venue.submit(req, tif=plan.tif))
        log.info("entry_submitted", symbol=sig.symbol, side=sig.side.value,
                 qty=qty, style=plan.style, coid=coid, trace_id=sig.trace_id)

    def _submit_exit(self, q: Quote, instr) -> None:
        now = self.clock.now()
        symbol = q.symbol
        if instr is None:                             # emergency flatten path
            pos = self.pm.positions.get(symbol)
            if pos is None:
                return
            side = Side.BUY if pos.side is Side.SELL else Side.SELL
            qty, reason, trace = pos.quantity, "emergency_flatten", pos.trace_id
        else:
            side, qty, reason, trace = (instr.side, instr.quantity,
                                        instr.reason, instr.trace_id)
        for mo in self.om.active_orders():             # one exit at a time
            if mo.request.symbol == symbol and mo.request.intent == "exit":
                return
        coid = self.journal.next_client_order_id(symbol, "exit", now)
        req = OrderRequest(symbol, side, qty, "MARKET", None, coid, trace,
                           "exit", {"reason": reason})
        try:
            self.om.create(req, ref_price=q.mid, risk_usd=0.0, ttl_ms=None,
                           activation_latency_ms=self.s.execution.paper.latency_ms)
        except IllegalTransition as e:
            log.warning("om_exit_reject", err=str(e))
            return
        self._pending_exit[coid] = reason
        self._decision_px[coid] = (q.mid, now)
        self.om.mark_submitting(coid)
        if self.live:
            self.adapter.place_order(req)
        else:
            self._apply_venue_events(self.venue.submit(req))

    # ---------------- shutdown ----------------
    def shutdown(self, flatten: bool = False) -> None:
        self.emergency = flatten
        self.fstore.flush()
        self.recorder.close()
        self.oufs.health(event="shutdown", flatten=flatten)
        log.info("shutdown", flatten=flatten)


def run_session(settings: Settings, paper_only: bool = True) -> None:
    """Entry point for `wliq paper` / `wliq run` on the live stream."""
    from wliq.broker.webull_adapter import WebullAdapter
    clock = WallClock()
    health = DataHealthMonitor(settings.risk.stale_data_ms,
                               settings.governor.max_sequence_gap_tolerance)
    adapter = WebullAdapter(settings, health)
    live = (not paper_only) and settings.mode == "live"
    eng = SessionEngine(settings, clock,
                        live_adapter=adapter if live else None)
    eng.health = health
    if not eng.recover():
        raise SystemExit("Recovery/reconciliation failed; refusing to trade.")
    adapter.on_quote = lambda q: eng.on_quote(q)
    adapter.on_trade = lambda t: eng.on_trade(t)
    adapter.on_depth = lambda d: eng.on_depth(d)
    adapter.on_noii = lambda n: eng.on_noii(n)

    def _stop(*_a: object) -> None:
        eng.shutdown(flatten=live)
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _stop)
    _signal.signal(_signal.SIGTERM, _stop)
    log.info("session_started", mode=settings.mode, live=live,
             symbols=settings.symbols)
    try:
        adapter.run_stream()
    finally:
        eng.shutdown()
