"""Sprint 4 — adaptive execution policy.

Decides HOW to work an entry given urgency and microstructure:
  * dynamic passive offset (join / improve / cross) from spread percentile,
    queue depletion, and signal urgency
  * queue position estimate -> expected passive fill probability
  * smart retry ladder: passive -> repriced join -> marketable IOC, with a
    hard cap on total slippage vs the decision price
"""
from __future__ import annotations

from dataclasses import dataclass

from wliq.core.models import FeatureVector, Quote, Side


@dataclass
class ExecPlan:
    style: str                 # "passive" | "join" | "cross"
    limit_price: float | None
    tif: str                   # DAY | IOC
    ttl_ms: int
    expected_fill_prob: float
    max_chase_bps: float       # retry ladder must not exceed this vs decision px


def estimate_queue_fill_prob(queue_shares: float, trade_rate_shares_s: float,
                             ttl_s: float, through_prob_s: float = 0.02) -> float:
    """P(filled) ~ P(queue consumed by touch flow) + P(trade-through in TTL).
    Crude but monotone and calibratable from recorded counterfactuals."""
    if queue_shares <= 0:
        return 1.0
    t_consume = queue_shares / max(trade_rate_shares_s, 1e-6)
    p_consume = max(0.0, min(1.0, (ttl_s - t_consume) / max(ttl_s, 1e-6)))
    p_through = 1.0 - (1.0 - through_prob_s) ** max(ttl_s, 0.0)
    return max(0.0, min(1.0, p_consume * 0.6 + p_through * 0.8))


def plan_entry(f: FeatureVector, q: Quote, side: Side, urgency: float,
               trade_rate_shares_s: float, tick: float = 0.01,
               base_ttl_ms: int = 4000) -> ExecPlan:
    """urgency in [0,1]: FSM FAILED_RETEST entries are time-sensitive but not
    panic; scale from signal quality (peak z) upstream."""
    touch = q.bid if side is Side.BUY else q.ask
    queue = q.bid_size if side is Side.BUY else q.ask_size
    p_passive = estimate_queue_fill_prob(queue, trade_rate_shares_s,
                                         base_ttl_ms / 1000.0)
    wide = f.spread_pctile >= 0.75
    depleting_against = (f.queue_depletion < -0.5 if side is Side.BUY
                         else f.queue_depletion > 0.5)
    if urgency >= 0.85 or (depleting_against and urgency >= 0.5):
        px = q.ask if side is Side.BUY else q.bid
        return ExecPlan("cross", px, "IOC", 1500, 0.9,
                        max_chase_bps=f.spread_bps + 3.0)
    if wide and f.spread_bps >= 2 * tick / max(q.mid, 1e-9) * 1e4:
        px = touch + tick if side is Side.BUY else touch - tick
        return ExecPlan("join", px, "DAY", base_ttl_ms,
                        min(0.95, p_passive + 0.25), max_chase_bps=f.spread_bps)
    return ExecPlan("passive", touch, "DAY", base_ttl_ms, p_passive,
                    max_chase_bps=f.spread_bps)


def next_retry(prev: ExecPlan, f: FeatureVector, q: Quote, side: Side,
               decision_price: float, tick: float = 0.01) -> ExecPlan | None:
    """Retry ladder: passive -> join -> cross(IOC) -> give up. Never exceeds
    max_chase_bps total slippage vs the decision price."""
    if prev.style == "cross":
        return None
    nxt = "join" if prev.style == "passive" else "cross"
    if nxt == "join":
        px = (q.bid + tick) if side is Side.BUY else (q.ask - tick)
        tif, ttl = "DAY", 2500
    else:
        px = q.ask if side is Side.BUY else q.bid
        tif, ttl = "IOC", 1500
    slip_bps = (px - decision_price) / max(decision_price, 1e-9) * 1e4
    if side is Side.SELL:
        slip_bps = -slip_bps
    if slip_bps > prev.max_chase_bps:
        return None
    return ExecPlan(nxt, px, tif, ttl, 0.9 if nxt == "cross" else 0.6,
                    prev.max_chase_bps)
