"""Entry signal scoring for reversal and continuation.

All coefficients come from YAML (StrategyConfig.weights) — nothing permanent
is hard-coded. Scores are additive over clipped, sign-oriented components so
each feature family's contribution is inspectable in the reason dict.
"""
from __future__ import annotations

import uuid

from wliq.core.config import StrategyConfig
from wliq.core.models import FeatureVector, GovernorAction, Side, Signal


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ResidualExhaustionStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.cfg = config

    def evaluate(self, f: FeatureVector, action: GovernorAction) -> Signal | None:
        if action is GovernorAction.NO_TRADE:
            return None
        move_dir = 1.0 if f.residual_z > 0 else -1.0
        w = self.cfg.weights

        if action is GovernorAction.REVERSAL:
            # Fade the residual move: short an exhausted up-burst, long a down-burst.
            components = {
                "residual_z": w.residual_z * _clip(abs(f.residual_z), 0, 5),
                "flow_exhaustion": w.flow_exhaustion * max(0.0, move_dir * f.flow_imbalance),
                "impact_decay": w.impact_decay * max(0.0, f.impact_decay),
                "replenishment": w.replenishment * max(0.0, -move_dir * f.replenishment_skew),
                "microprice": w.microprice * max(0.0, -move_dir * f.microprice_bps / 2.0),
                "peer_divergence": w.peer_divergence * max(0.0, 1.0 - f.peer_confirmation),
                "volume_surprise": w.volume_surprise * _clip(f.volume_z, 0, 4) / 4,
                "depth": w.depth_imbalance * max(0.0, -move_dir * f.depth_imbalance),
                "spread_normalization": w.spread_normalization * max(0.0, 0.6 - f.spread_pctile),
            }
            min_score = self.cfg.min_score
            side = Side.SELL if move_dir > 0 else Side.BUY
        else:  # CONTINUATION: trade WITH the residual move
            components = {
                "residual_z": w.residual_z * _clip(abs(f.residual_z), 0, 5),
                "flow_agreement": w.flow_exhaustion * max(0.0, move_dir * f.flow_imbalance),
                "impact_rising": w.impact_decay * max(0.0, -f.impact_decay),
                "opposing_depletion": w.replenishment * max(0.0, move_dir * f.replenishment_skew),
                "microprice": w.microprice * max(0.0, move_dir * f.microprice_bps / 2.0),
                "peer_confirmation": w.peer_divergence * max(0.0, f.peer_confirmation - 0.3),
                "volume_surprise": w.volume_surprise * _clip(f.volume_z, 0, 4) / 4,
                "depth": w.depth_imbalance * max(0.0, move_dir * f.depth_imbalance),
            }
            min_score = self.cfg.min_continuation_score
            side = Side.BUY if move_dir > 0 else Side.SELL

        score = sum(components.values())
        # Execution-cost filter: expected edge must clear spread + slippage.
        expected_move_bps = abs(f.residual_return) * 10_000 * self.cfg.target_residual_reversion
        cost_bps = f.spread_bps + 2.0  # spread + slippage/adverse-selection allowance
        if action is GovernorAction.REVERSAL and expected_move_bps < cost_bps:
            return None
        if score < min_score:
            return None

        stop_delta = f.mid * self.cfg.stop_bps / 10_000
        if action is GovernorAction.REVERSAL:
            target_delta = max(abs(f.residual_return) * f.mid * self.cfg.target_residual_reversion,
                               f.mid * cost_bps / 10_000)
        else:
            target_delta = max(f.mid * self.cfg.stop_bps * 1.5 / 10_000, f.mid * cost_bps / 10_000)
        stop = f.mid + stop_delta if side is Side.SELL else f.mid - stop_delta
        target = f.mid - target_delta if side is Side.SELL else f.mid + target_delta

        return Signal(
            symbol=f.symbol,
            ts=f.ts,
            side=side,
            action=action,
            score=score,
            reference_price=f.mid,
            stop_price=stop,
            target_price=target,
            trace_id=uuid.uuid4().hex[:16],
            reason={k: round(v, 4) for k, v in components.items()}
            | {"residual_z_raw": round(f.residual_z, 3), "volume_z": round(f.volume_z, 2)},
        )
