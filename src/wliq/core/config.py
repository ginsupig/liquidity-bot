"""Validated configuration.

All tunable coefficients and thresholds live in YAML. Secrets are read from
environment variables only. Live trading requires BOTH mode=live and
live_trading_enabled=true (double opt-in interlock).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RiskConfig(BaseModel):
    risk_per_trade_usd: float = 20.0
    max_position_shares: int = 500
    max_notional_per_symbol_usd: float = 2_000.0
    max_gross_exposure_usd: float = 5_000.0
    max_positions: int = 2
    max_daily_loss_usd: float = 100.0
    max_consecutive_losses: int = 3
    max_spread_bps: float = 8.0
    max_slippage_bps: float = 6.0
    max_holding_seconds: int = 600
    stale_data_ms: int = 1500
    max_order_retries: int = 2
    symbol_cooldown_seconds: int = 120
    strategy_cooldown_seconds: int = 300
    kill_switch: bool = False


class ScoringWeights(BaseModel):
    """Signal-scoring coefficients. Calibrate offline; never hard-code."""
    residual_z: float = 1.20
    flow_exhaustion: float = 0.80
    impact_decay: float = 0.75
    replenishment: float = 0.55
    microprice: float = 0.35
    peer_divergence: float = 0.65
    volume_surprise: float = 0.30
    depth_imbalance: float = 0.40
    spread_normalization: float = 0.25


class StrategyConfig(BaseModel):
    residual_z_entry: float = 2.25
    min_score: float = 3.0
    min_continuation_score: float = 3.5
    max_holding_seconds: int = 300
    target_residual_reversion: float = 0.55
    stop_bps: float = 22.0
    min_events: int = 25
    no_trade_first_seconds: int = 120
    no_trade_last_seconds: int = 300
    min_volume_z: float = 1.0
    max_peer_confirmation_for_reversal: float = 0.30
    min_peer_confirmation_for_continuation: float = 0.50
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class GovernorConfig(BaseModel):
    max_breadth_for_reversal: float = 0.75      # |breadth| above this = trend day
    max_realized_vol_bps: float = 90.0
    min_realized_vol_bps: float = 2.0
    max_spread_pctile_for_entry: float = 0.90
    max_sequence_gap_tolerance: int = 5
    require_healthy_stream: bool = True
    catalyst_blacklist: list[str] = Field(default_factory=list)  # symbols with known news
    earnings_blacklist: list[str] = Field(default_factory=list)


class PaperFillConfig(BaseModel):
    latency_ms: int = 250              # signal -> order at exchange
    cancel_latency_ms: int = 200
    queue_position_factor: float = 1.0  # assume we join behind 100% of displayed size
    require_trade_through: bool = True  # passive fills need trade beyond our price
    marketable_max_take_fraction: float = 0.50  # of displayed size at touch
    adverse_selection_bps: float = 1.0  # markout penalty applied to passive fills
    partial_fill_min_shares: int = 10


class ExecutionConfig(BaseModel):
    passive_offset_ticks: int = 0
    tick_size: float = 0.01
    cancel_after_ms: int = 1200
    paper: PaperFillConfig = Field(default_factory=PaperFillConfig)


class RecorderConfig(BaseModel):
    flush_rows: int = 1000
    flush_seconds: int = 30
    dedupe: bool = True


class VolumeModelConfig(BaseModel):
    profile_path: Path = Path("data/profiles/volume_profile.parquet")
    bucket_minutes: int = 5
    min_history_days: int = 5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_file: Path | None = Path("logs/wliq.jsonl")


class Settings(BaseModel):
    environment: Literal["sandbox", "production"] = "sandbox"
    mode: Literal["record", "research", "replay", "paper", "live"] = "record"
    live_trading_enabled: bool = False
    symbols: list[str]
    market_etf: str = "QQQ"
    sector_etf: str = "SMH"
    benchmarks: dict[str, list[str]] = Field(default_factory=dict)  # symbol -> peers
    stream_types: list[str] = Field(default_factory=lambda: ["QUOTE", "TICK"])
    subscribe_depth: bool = False
    subscribe_noii: bool = False
    record_root: Path = Path("data")
    feature_window_seconds: int = 120
    residual_halflife_seconds: int = 300
    signal_interval_ms: int = 1000
    reconcile_interval_s: int = 45          # periodic live REST reconciliation
    min_data_confidence: float = 0.7        # Sprint-1 quality gate
    sector_map: dict[str, str] = {}         # symbol -> sector bucket
    session_open: str = "09:30"
    session_close: str = "16:00"
    session_tz: str = "America/New_York"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    governor: GovernorConfig = Field(default_factory=GovernorConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    volume_model: VolumeModelConfig = Field(default_factory=VolumeModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _fail_closed(self) -> "Settings":
        if self.mode == "live" and not self.live_trading_enabled:
            raise ValueError(
                "mode=live requires live_trading_enabled=true (double opt-in). "
                "This interlock is deliberate."
            )
        if self.mode == "live" and self.environment == "sandbox":
            raise ValueError("mode=live is not permitted against environment=sandbox")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    # Secrets: environment only. Never persisted, never logged.
    @property
    def app_key(self) -> str:
        return os.getenv("WEBULL_APP_KEY", "")

    @property
    def app_secret(self) -> str:
        return os.getenv("WEBULL_APP_SECRET", "")

    @property
    def account_id(self) -> str:
        return os.getenv("WEBULL_ACCOUNT_ID", "")

    @property
    def api_host(self) -> str:
        default = "api.sandbox.webull.com" if self.environment == "sandbox" else "api.webull.com"
        return os.getenv("WEBULL_API_HOST", default)

    @property
    def mqtt_host(self) -> str:
        default = (
            "data-api.sandbox.webull.com" if self.environment == "sandbox" else "data-api.webull.com"
        )
        return os.getenv("WEBULL_MQTT_HOST", default)

    def all_streamed_symbols(self) -> list[str]:
        universe: set[str] = set(self.symbols) | {self.market_etf, self.sector_etf}
        for peers in self.benchmarks.values():
            universe.update(peers)
        return sorted(universe)
