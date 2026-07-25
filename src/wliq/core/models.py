"""Core domain models.

Every model is a slotted dataclass. Market-data models are frozen; stateful
execution models are mutable. Timestamps are timezone-aware UTC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class Session(str, Enum):
    PRE = "PRE"
    RTH = "RTH"
    POST = "POST"
    CLOSED = "CLOSED"


class GovernorAction(str, Enum):
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"
    NO_TRADE = "NO_TRADE"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    session: Session = Session.RTH
    seq: int | None = None
    recv_ts: datetime = field(default_factory=utcnow)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_bps(self) -> float:
        return 10_000 * self.spread / self.mid if self.mid > 0 else float("inf")

    @property
    def microprice(self) -> float:
        total = self.bid_size + self.ask_size
        if total <= 0:
            return self.mid
        return (self.ask * self.bid_size + self.bid * self.ask_size) / total


@dataclass(slots=True, frozen=True)
class Trade:
    symbol: str
    ts: datetime
    price: float
    size: float
    aggressor: Side | None = None
    session: Session = Session.RTH
    seq: int | None = None
    recv_ts: datetime = field(default_factory=utcnow)


@dataclass(slots=True, frozen=True)
class DepthLevel:
    price: float
    size: float


@dataclass(slots=True, frozen=True)
class DepthSnapshot:
    """Level 2 book snapshot (top N levels each side, best first)."""
    symbol: str
    ts: datetime
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    seq: int | None = None
    recv_ts: datetime = field(default_factory=utcnow)

    def depth_notional(self, side: Side, levels: int | None = None) -> float:
        book = self.bids if side is Side.BUY else self.asks
        sel = book[:levels] if levels else book
        return sum(l.price * l.size for l in sel)

    def depth_shares(self, side: Side, levels: int | None = None) -> float:
        book = self.bids if side is Side.BUY else self.asks
        sel = book[:levels] if levels else book
        return sum(l.size for l in sel)


@dataclass(slots=True, frozen=True)
class Bar:
    symbol: str
    ts: datetime           # bar open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    timespan: str = "M1"


@dataclass(slots=True, frozen=True)
class Noii:
    """Net Order Imbalance Indicator auction snapshot.

    UNVERIFIED: exact Webull OpenAPI NOII payload fields are not confirmed
    against live sandbox output. All parsing is isolated in the adapter.
    """
    symbol: str
    ts: datetime
    paired_qty: float
    imbalance_qty: float
    imbalance_side: Side | None
    indicative_price: float | None
    near_price: float | None
    far_price: float | None
    auction: str = "CLOSE"  # OPEN | CLOSE
    recv_ts: datetime = field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Features / signals
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class FeatureVector:
    symbol: str
    ts: datetime
    mid: float
    spread_bps: float
    spread_pctile: float          # rolling percentile of spread in [0,1]
    residual_return: float        # beta-adjusted log return over window
    residual_z: float
    flow_imbalance: float         # (aggr buy - aggr sell) / total in [-1,1]
    impact_decay: float           # [-1,1]; positive = impact fading (exhaustion)
    replenishment_skew: float     # bid replenishment minus ask replenishment
    microprice_bps: float         # microprice deviation from mid
    peer_confirmation: float      # sign(own) * peer_mean / |own|; <~0.3 = unconfirmed
    depth_imbalance: float        # (bid depth - ask depth)/(sum) in [-1,1]; 0 if no L2
    near_touch_ratio: float       # touch size / total depth per side avg; 0 if no L2
    queue_depletion: float        # signed touch-size depletion velocity; 0 if no L2
    volume_z: float               # standardized volume surprise
    vwap_distance_bps: float
    realized_vol_bps: float       # rolling 1s-return stdev annualized to window, bps
    minutes_since_open: float
    event_count: int
    # ---- v2 additions (defaults keep v0.2 call sites valid) ----
    residual_velocity: float = 0.0        # d(residual)/dt over short window, bps/s
    residual_acceleration: float = 0.0    # d(velocity)/dt, bps/s^2
    residual_vol_z: float = 0.0           # residual-return vol vs its own history
    peer_confirm_latency_s: float = -1.0  # s until peers confirmed; -1 = never
    sector_confirm_latency_s: float = -1.0
    replenishment_confidence: float = 0.0  # [0,1] sample-count-weighted confidence
    vol_regime: int = 1                    # 0 low / 1 normal / 2 high (per-symbol)
    tod_bucket: int = 0                    # 0 open / 1 morning / 2 midday / 3 afternoon / 4 close
    noii_pressure: float = 0.0             # signed auction imbalance; 0 if absent
    data_confidence: float = 1.0           # Sprint-1 quality score


@dataclass(slots=True, frozen=True)
class Signal:
    symbol: str
    ts: datetime
    side: Side
    action: GovernorAction        # REVERSAL or CONTINUATION
    score: float
    reference_price: float
    stop_price: float
    target_price: float
    trace_id: str
    reason: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GovernorDecision:
    action: GovernorAction
    reasons: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: float
    order_type: str               # LIMIT | MARKET
    limit_price: float | None
    client_order_id: str
    trace_id: str
    intent: str = "entry"         # entry | exit | flatten
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime
    client_order_id: str
    trace_id: str
    partial: bool = False


@dataclass(slots=True)
class OrderState:
    request: OrderRequest
    status: OrderStatus
    filled_qty: float = 0.0
    avg_price: float = 0.0
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    broker_order_id: str | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    opened_at: datetime
    stop_price: float
    target_price: float
    signal_score: float
    trace_id: str
    action: GovernorAction = GovernorAction.REVERSAL
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    realized_pnl: float = 0.0      # accrued across partial exits
    exited_qty: float = 0.0
    exit_notional: float = 0.0

    def unrealized_pnl(self, mark: float) -> float:
        return self.side.sign * (mark - self.entry_price) * self.quantity

    def update_excursions(self, mark: float) -> None:
        pnl_per_share = self.side.sign * (mark - self.entry_price)
        self.max_favorable = max(self.max_favorable, pnl_per_share)
        self.max_adverse = min(self.max_adverse, pnl_per_share)


@dataclass(slots=True, frozen=True)
class ClosedTrade:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    exit_reason: str
    trace_id: str
    signal_score: float
    action: GovernorAction
    max_favorable: float
    max_adverse: float
