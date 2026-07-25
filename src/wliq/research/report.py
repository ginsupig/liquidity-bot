"""Performance reporting: session summaries from closed trades and journals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wliq.core.models import ClosedTrade


def trades_frame(trades: list[ClosedTrade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": t.symbol, "side": t.side.value, "action": t.action.value,
        "qty": t.quantity, "entry": t.entry_price, "exit": t.exit_price,
        "opened_at": t.opened_at, "closed_at": t.closed_at,
        "holding_s": (t.closed_at - t.opened_at).total_seconds(),
        "pnl": t.pnl, "exit_reason": t.exit_reason, "score": t.signal_score,
        "mfe_per_share": t.max_favorable, "mae_per_share": t.max_adverse,
        "trace_id": t.trace_id,
    } for t in trades])


def summarize(trades: list[ClosedTrade]) -> dict:
    if not trades:
        return {"n_trades": 0, "note": "no closed trades"}
    df = trades_frame(trades)
    pnl = df["pnl"].to_numpy()
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "n_trades": len(df),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "max_drawdown": dd,
        "median_holding_s": float(df["holding_s"].median()),
        "exit_reasons": df["exit_reason"].value_counts().to_dict(),
        "by_action": df.groupby("action")["pnl"].agg(["count", "sum"]).to_dict("index"),
        "by_symbol": df.groupby("symbol")["pnl"].sum().to_dict(),
    }


def journal_summary(path: Path) -> dict:
    if not path.exists():
        return {"note": f"no journal at {path}"}
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    if df.empty:
        return {"note": "journal empty"}
    return {"orders": len(df),
            "by_event": df["event"].value_counts().to_dict() if "event" in df else {},
            "by_status": df["status"].value_counts().to_dict() if "status" in df else {}}
