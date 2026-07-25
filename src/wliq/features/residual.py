"""Online exponentially weighted beta estimation.

Betas are estimated on synchronized 1-second mid-return samples so the
regression is not distorted by asynchronous quote arrival. Memory is bounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class EwMoments:
    """EW covariance/variance accumulator for one (stock, factor) pair."""
    alpha: float
    cov: float = 0.0
    var: float = 0.0
    n: int = 0

    def update(self, x: float, y: float) -> None:
        # Zero-mean assumption is standard for high-frequency log returns.
        self.cov = (1 - self.alpha) * self.cov + self.alpha * x * y
        self.var = (1 - self.alpha) * self.var + self.alpha * x * x
        self.n += 1

    @property
    def beta(self) -> float:
        if self.n < 30 or self.var <= 1e-18:
            return 1.0  # prior: unit beta until enough evidence
        return max(-3.0, min(3.0, self.cov / self.var))


@dataclass(slots=True)
class ResidualModel:
    """Rolling EW multi-factor residual: stock vs market ETF, sector ETF, peers.

    Peers are equal-weight averaged into a single factor to keep the online
    estimator well-conditioned (individual peer betas on 1s data are noisy).
    """
    halflife_seconds: float
    _moments: dict[tuple[str, str], EwMoments] = field(default_factory=dict)

    def _alpha(self) -> float:
        return 1 - math.exp(math.log(0.5) / max(self.halflife_seconds, 1.0))

    def _get(self, symbol: str, factor: str) -> EwMoments:
        key = (symbol, factor)
        if key not in self._moments:
            self._moments[key] = EwMoments(alpha=self._alpha())
        return self._moments[key]

    def update(self, symbol: str, stock_ret: float, market_ret: float,
               sector_ret: float, peer_ret: float) -> None:
        self._get(symbol, "market").update(market_ret, stock_ret)
        self._get(symbol, "sector").update(sector_ret, stock_ret)
        self._get(symbol, "peer").update(peer_ret, stock_ret)

    def residual(self, symbol: str, stock_ret: float, market_ret: float,
                 sector_ret: float, peer_ret: float) -> float:
        bm = self._get(symbol, "market").beta
        bs = self._get(symbol, "sector").beta
        bp = self._get(symbol, "peer").beta
        # Hierarchical attribution: market first, then sector/peer residual layers
        # at reduced weight to avoid double counting collinear factors.
        return stock_ret - bm * market_ret - 0.5 * bs * (sector_ret - market_ret) \
            - 0.5 * bp * (peer_ret - market_ret)

    def betas(self, symbol: str) -> dict[str, float]:
        return {f: self._get(symbol, f).beta for f in ("market", "sector", "peer")}
