"""Sprint 7 — lightweight status: heartbeat JSONL (OUFS health stream) plus a
single self-contained HTML status report. Deliberately not a web app —
operator preference is simple heartbeat/status over complex dashboards.

Sections: positions, order lifecycle, execution quality, risk state, P&L,
regime/feature snapshot, data quality, replay/attribution summaries.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;background:#101418;color:#dfe7ee;
margin:24px}h1{color:#c9a227;font-weight:600}h2{color:#7fae7f;border-bottom:
1px solid #2a3540;padding-bottom:4px;margin-top:28px}table{border-collapse:
collapse;margin:8px 0;font-size:13px}td,th{border:1px solid #2a3540;padding:
4px 10px;text-align:right}th{background:#1a2229;color:#c9a227}td:first-child,
th:first-child{text-align:left}.ok{color:#7fae7f}.warn{color:#e0b04b}
.bad{color:#e06c5b}.mono{font-family:Consolas,monospace;font-size:12px}
small{color:#6c7a86}
"""


def _table(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"<h2>{html.escape(title)}</h2><small>none</small>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for r in rows:
        tds = ""
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:,.2f}"
            tds += f"<td>{html.escape(str(v))}</td>"
        body += f"<tr>{tds}</tr>"
    return (f"<h2>{html.escape(title)}</h2>"
            f"<table><tr>{head}</tr>{body}</table>")


def render_status(path: Path, *, positions: list[dict], orders: list[dict],
                  exec_quality: dict, risk: dict, pnl: dict,
                  regime: dict, data_quality: dict,
                  attribution: list[dict] | None = None) -> Path:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head>"
             f"<body><h1>wliq status</h1><small>{now}</small>"]
    parts.append(_table(positions, "Open positions"))
    parts.append(_table(orders, "Order lifecycle (active + recent)"))
    parts.append(_table([exec_quality] if exec_quality else [],
                        "Execution quality"))
    parts.append(_table([risk] if risk else [], "Risk state"))
    parts.append(_table([pnl] if pnl else [], "P&L"))
    parts.append(_table([regime] if regime else [], "Regime / feature snapshot"))
    parts.append(_table([{"symbol": k, **v} for k, v in data_quality.items()],
                        "Data quality"))
    if attribution:
        parts.append(_table(attribution, "Attribution (by exit reason)"))
    parts.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path


def heartbeat(oufs: Any, **state: Any) -> None:
    oufs.health(**state)


def status_from_logs(logs_dir: Path, out_path: Path) -> Path:
    """Offline: build the report from journals/ledgers when no session runs."""
    orders: list[dict] = []
    jpath = logs_dir / "orders.jsonl"
    if jpath.exists():
        latest: dict[str, dict] = {}
        for line in jpath.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("client_order_id"):
                latest[row["client_order_id"]] = row
        for r in list(latest.values())[-25:]:
            orders.append({"coid": r["client_order_id"], "symbol": r["symbol"],
                           "side": r["side"], "qty": r["quantity"],
                           "status": r.get("status", "?"),
                           "intent": r.get("intent", "?")})
    from wliq.execution.ledger import RiskLedger
    day = RiskLedger(logs_dir / "risk_ledger.jsonl").load_day()
    return render_status(out_path, positions=[], orders=orders,
                         exec_quality={}, risk={"halted": day["halted"],
                                                "consecutive_losses": day["consecutive_losses"]},
                         pnl={"realized": day["realized_pnl"],
                              "n_trades": day["n_trades"]},
                         regime={}, data_quality={})
