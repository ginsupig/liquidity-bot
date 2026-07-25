from datetime import datetime, timezone

from wliq.core.models import Fill, Side
from wliq.execution.ledger import (PositionEventLog, RiskLedger,
                                   rebuild_open_orders)
from wliq.execution.order_manager import OState

T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


def test_risk_ledger_restores_day_state(tmp_path):
    led = RiskLedger(tmp_path / "risk.jsonl")
    led.append("trade_closed", pnl=-12.0)
    led.append("trade_closed", pnl=-8.0)
    led.append("trade_closed", pnl=5.0)
    led.append("halt", reason="test")
    day = RiskLedger(tmp_path / "risk.jsonl").load_day()   # fresh instance
    assert abs(day["realized_pnl"] - (-15.0)) < 1e-9
    assert day["n_trades"] == 3
    assert day["consecutive_losses"] == 0    # last trade was a win
    assert day["halted"] is True


def test_position_rebuild_nets_partials(tmp_path):
    plog = PositionEventLog(tmp_path / "pos.jsonl")
    plog.append_fill("entry", Fill("MU", Side.SELL, 60, 100.0, T0, "e1", "t1"),
                     {"stop": 100.5, "target": 99.2, "action": "REVERSAL"})
    plog.append_fill("entry", Fill("MU", Side.SELL, 40, 100.1, T0, "e1", "t1"))
    plog.append_fill("exit", Fill("MU", Side.BUY, 30, 99.8, T0, "x1", "t1"))
    plog.append_fill("entry", Fill("WDC", Side.SELL, 50, 60.0, T0, "e2", "t2"))
    plog.append_fill("exit", Fill("WDC", Side.BUY, 50, 59.5, T0, "x2", "t2"))
    book = plog.rebuild_positions()
    assert "WDC" not in book                # fully exited
    mu = book["MU"]
    assert mu.quantity == 70
    assert abs(mu.entry_price - 100.04) < 1e-9   # (60*100 + 40*100.1)/100
    assert mu.stop_price == 100.5


def test_rebuild_open_orders_skips_terminal(tmp_path):
    import json
    p = tmp_path / "orders.jsonl"
    rows = [
        {"client_order_id": "a", "symbol": "MU", "side": "SELL", "quantity": 10,
         "order_type": "LIMIT", "limit_price": 100.0, "intent": "entry",
         "status": "ACKNOWLEDGED", "trace_id": "t"},
        {"client_order_id": "b", "symbol": "MU", "side": "BUY", "quantity": 10,
         "order_type": "MARKET", "limit_price": None, "intent": "exit",
         "status": "FILLED", "trace_id": "t"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = rebuild_open_orders(p)
    assert len(out) == 1 and out[0].request.client_order_id == "a"
    assert out[0].state is OState.ACKNOWLEDGED
