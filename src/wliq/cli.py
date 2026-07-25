"""wliq command-line interface."""
from __future__ import annotations

import json
import signal as _signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from wliq.core.config import Settings
from wliq.core.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True)
CONFIG_OPT = typer.Option(Path("config/settings.yaml"), "--config")


def _load(config: Path) -> Settings:
    s = Settings.load(config)
    configure_logging(s.logging.level, s.logging.json_file)
    return s


@app.command()
def doctor(config: Path = CONFIG_OPT) -> None:
    """Validate configuration, credentials presence, and safety interlocks."""
    s = _load(config)
    typer.echo(f"mode={s.mode} environment={s.environment} "
               f"live_trading_enabled={s.live_trading_enabled}")
    typer.echo(f"symbols={s.symbols}")
    typer.echo(f"streamed_universe={len(s.all_streamed_symbols())} symbols")
    typer.echo(f"credentials={'present' if s.app_key and s.app_secret else 'MISSING'}")
    typer.echo(f"account_id={'present' if s.account_id else 'MISSING'}")
    typer.echo(f"volume_profile={'found' if s.volume_model.profile_path.exists() else 'not built (fallback baseline will be used)'}")
    typer.echo("Interlock: live orders require mode=live AND live_trading_enabled=true "
               "AND environment=production.")


@app.command()
def record(config: Path = CONFIG_OPT) -> None:
    """Stream Webull market data and record to partitioned Parquet. No trading."""
    s = _load(config)
    from wliq.broker.health import DataHealthMonitor
    from wliq.broker.webull_adapter import WebullAdapter
    from wliq.data.recorder import ParquetRecorder

    log = get_logger("record")
    health = DataHealthMonitor(s.risk.stale_data_ms, s.governor.max_sequence_gap_tolerance)
    adapter = WebullAdapter(s, health)
    rec = ParquetRecorder(s.record_root, s.recorder.flush_rows,
                          s.recorder.flush_seconds, s.recorder.dedupe)
    adapter.on_quote = rec.append_quote
    adapter.on_trade = rec.append_trade
    adapter.on_depth = rec.append_depth
    adapter.on_noii = rec.append_noii

    def _stop(*_a: object) -> None:
        log.info("shutdown", reason="signal")
        rec.close()
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _stop)
    _signal.signal(_signal.SIGTERM, _stop)
    log.info("recording_started", symbols=s.all_streamed_symbols())
    try:
        adapter.run_stream()
    finally:
        rec.close()


@app.command()
def download(config: Path = CONFIG_OPT, timespan: str = "M1", count: int = 500) -> None:
    """Download historical bars and (re)build the volume profile."""
    s = _load(config)
    from wliq.data.downloader import build_volume_profile, download_history
    n = download_history(s, timespan, count)
    typer.echo(f"downloaded {n} bars")
    try:
        path = build_volume_profile(s)
        typer.echo(f"volume profile written to {path}")
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"volume profile not built: {exc}")


@app.command()
def paper(config: Path = CONFIG_OPT) -> None:
    """Run the live stream through the full pipeline with paper execution."""
    s = _load(config)
    if s.mode not in ("paper", "record"):
        typer.echo("Set mode: paper in settings.yaml to run paper trading.")
        raise typer.Exit(1)
    from wliq.runtime import run_session
    run_session(s, paper_only=True)


@app.command()
def run(config: Path = CONFIG_OPT) -> None:
    """Run in the configured mode (record / paper / live)."""
    s = _load(config)
    if s.mode == "record":
        record.callback  # type: ignore[attr-defined]
        return record(config)  # type: ignore[misc]
    from wliq.runtime import run_session
    run_session(s, paper_only=(s.mode != "live"))


@app.command()
def replay(config: Path = CONFIG_OPT, date: str = typer.Option(...)) -> None:
    """Event-driven replay of a recorded date through the full pipeline."""
    s = _load(config)
    from wliq.research.replay import run_replay
    from wliq.research.report import summarize
    result = run_replay(s, date)
    typer.echo(json.dumps({
        "date": date, "signals": result.signals,
        "governor_actions": result.governor_actions,
        "risk_rejections": result.rejections,
        "summary": summarize(result.trades),
    }, indent=2, default=str))


@app.command()
def research(config: Path = CONFIG_OPT,
             dates: str = typer.Option(..., help="comma-separated YYYY-MM-DD"),
             horizon: int = 120, cost_bps: float = 3.0) -> None:
    """Mine events across dates and run purged walk-forward with baselines."""
    s = _load(config)
    import pandas as pd
    from wliq.research.miner import mine_events
    from wliq.research.walkforward import purged_walk_forward
    frames = [mine_events(s, d) for d in dates.split(",")]
    events = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if events.empty:
        typer.echo("no events mined; record more data first")
        raise typer.Exit(1)
    out_dir = Path("research_out")
    out_dir.mkdir(exist_ok=True)
    events.to_parquet(out_dir / "mined_events.parquet", index=False)
    rep = purged_walk_forward(events, s.strategy.weights.model_dump(),
                              s.strategy.min_score, horizon, cost_bps)
    typer.echo(json.dumps({
        "n_events": len(events), "horizon_s": horizon, "cost_bps": cost_bps,
        "full_model": rep.full_model, "baselines": rep.baselines,
        "mc_drawdown": rep.mc_drawdown, "stress_costs_doubled": rep.stress,
        "concentration": rep.concentration,
        "folds": [vars(f) for f in rep.folds],
    }, indent=2, default=str))
    typer.echo(f"mined events saved to {out_dir/'mined_events.parquet'}")


@app.command()
def report(config: Path = CONFIG_OPT,
           journal: Path = Path("logs/orders.jsonl")) -> None:
    """Summarize the order journal."""
    _load(config)
    from wliq.research.report import journal_summary
    typer.echo(json.dumps(journal_summary(journal), indent=2, default=str))


@app.command()
def flatten(config: Path = CONFIG_OPT,
            confirm: bool = typer.Option(False, "--yes")) -> None:
    """Emergency: flatten all broker positions (live mode only)."""
    s = _load(config)
    if not confirm:
        typer.echo("Refusing without --yes. This submits closing MARKET orders.")
        raise typer.Exit(1)
    if s.mode != "live":
        typer.echo("flatten only acts against a live account; paper positions "
                   "are process-local and die with the process.")
        raise typer.Exit(0)
    from wliq.broker.webull_adapter import WebullAdapter
    from wliq.core.journal import OrderJournal
    from wliq.core.models import OrderRequest, OrderStatus, Side
    adapter = WebullAdapter(s)
    journal = OrderJournal(Path("logs/orders.jsonl"))
    positions = adapter.read_positions()
    now = datetime.now(timezone.utc)
    for symbol, qty in positions.items():
        if abs(qty) < 1e-9:
            continue
        side = Side.SELL if qty > 0 else Side.BUY
        coid = journal.next_client_order_id(symbol, "flatten", now)
        order = OrderRequest(symbol, side, abs(qty), "MARKET", None, coid,
                             f"flatten-{now.strftime('%H%M%S')}", "flatten")
        journal.record("submit", order, OrderStatus.PENDING)
        result = adapter.place_order(order)
        journal.record("ack", order, OrderStatus.SUBMITTED, {"broker": result})
        typer.echo(f"flatten {symbol} {side.value} {abs(qty)}")


@app.command()
def dashboard(config: Path = typer.Option(Path("config/settings.yaml"), "--config"),
              out: Path = typer.Option(Path("logs/status.html"), "--out"),
              logs: Path = typer.Option(Path("logs"), "--logs")) -> None:
    """Build the single-file HTML status report from journals/ledgers."""
    from wliq.status.dashboard import status_from_logs
    path = status_from_logs(logs, out)
    typer.echo(f"status written: {path}")


@app.command()
def recover(config: Path = typer.Option(Path("config/settings.yaml"), "--config")) -> None:
    """Dry-run startup recovery: show what would be rebuilt after a crash."""
    from wliq.core.clock import WallClock
    from wliq.core.config import load_settings
    from wliq.runtime import SessionEngine
    s = load_settings(config)
    eng = SessionEngine(s, WallClock(), live_adapter=None)
    eng.recover()
    typer.echo(f"positions: {list(eng.pm.positions)}")
    for sym, p in eng.pm.positions.items():
        typer.echo(f"  {sym}: {p.side.value} {p.quantity} @ {p.entry_price:.2f}"
                   f" stop={p.stop_price:.2f}")
    typer.echo(f"open orders: {[o.request.client_order_id for o in eng.om.active_orders()]}")
    typer.echo(f"realized today: {eng.risk.realized_pnl:+.2f}  halted={eng.risk2.halted}")


@app.command()
def stress(config: Path = typer.Option(Path("config/settings.yaml"), "--config"),
           events_path: Path = typer.Option(..., "--events"),
           min_score: float = typer.Option(3.0), horizon: int = typer.Option(120)) -> None:
    """Stress grid + parameter stability + bootstrap on a mined-events parquet."""
    import pandas as pd
    from wliq.research.stress import (bootstrap_test, parameter_stability,
                                      stress_grid)
    from wliq.research.walkforward import trade_pnl_bps
    ev = pd.read_parquet(events_path)
    typer.echo(stress_grid(ev, {}, min_score, horizon).to_string(index=False))
    typer.echo(f"stability: {parameter_stability(ev, {}, min_score, horizon)}")
    from wliq.research.stress import _select
    sel = _select(ev, {}, min_score)
    typer.echo(f"bootstrap: {bootstrap_test(trade_pnl_bps(sel, horizon, cost_bps=1.0))}")


@app.command()
def train(config: Path = typer.Option(Path("config/settings.yaml"), "--config"),
          events_path: Path = typer.Option(..., "--events"),
          horizon: int = typer.Option(120)) -> None:
    """Train + calibrate the ranking model from the feature store and mined
    events. Prints the calibration report; deployment stays manual and gated
    on walk-forward outperformance vs the rule selector."""
    import pandas as pd
    from wliq.core.config import load_settings
    from wliq.ml.model import calibration_report, train_ranker
    from wliq.ml.store import FeatureStore, label_events
    s = load_settings(config)
    feats = FeatureStore(Path(s.record_root) / "feature_store").load()
    mined = pd.read_parquet(events_path)
    labeled = label_events(feats, mined, horizon=horizon)
    typer.echo(f"labeled events: {len(labeled)}")
    bundle = train_ranker(labeled)
    typer.echo(f"model: {bundle.kind}  meta: {bundle.train_meta}")
    typer.echo(calibration_report(bundle, labeled).to_string())


if __name__ == "__main__":
    app()
