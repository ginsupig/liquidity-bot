from wliq.broker.webull_adapter import WebullAdapter
from wliq.core.config import Settings
from wliq.core.models import Side


def make(monkeypatch) -> WebullAdapter:
    monkeypatch.setenv("WEBULL_APP_KEY", "k")
    monkeypatch.setenv("WEBULL_APP_SECRET", "s")
    return WebullAdapter(Settings(symbols=["MU"]))


def test_quote_normalization_aliases(monkeypatch):
    a = make(monkeypatch)
    q = a.parse_quote({"ticker": "mu", "bidPrice": "100.00", "askPrice": "100.02",
                       "bidSize": 300, "askSize": 200, "timestamp": 1752934200000})
    assert q is not None and q.symbol == "MU" and q.bid == 100.0


def test_crossed_quote_rejected(monkeypatch):
    a = make(monkeypatch)
    assert a.parse_quote({"symbol": "MU", "bid": 100.05, "ask": 100.00}) is None


def test_trade_direction_aliases(monkeypatch):
    a = make(monkeypatch)
    t = a.parse_trade({"symbol": "MU", "price": 100.01, "volume": 100, "side": "B"})
    assert t is not None and t.aggressor is Side.BUY


def test_depth_list_and_dict_levels(monkeypatch):
    a = make(monkeypatch)
    d = a.parse_depth({"symbol": "MU",
                       "bids": [[99.99, 100], [99.98, 200]],
                       "asks": [{"price": 100.01, "size": 50}]})
    assert d is not None and len(d.bids) == 2 and d.asks[0].size == 50


def test_malformed_payload_counts_drop(monkeypatch):
    a = make(monkeypatch)
    a.handle_stream_message("quotes", b"not json{")
    assert a.health.dropped_messages == 1


def test_live_order_blocked_without_interlock(monkeypatch):
    a = make(monkeypatch)
    from wliq.core.models import OrderRequest
    import pytest
    with pytest.raises(RuntimeError, match="blocked"):
        a.place_order(OrderRequest("MU", Side.BUY, 1, "MARKET", None, "c", "t"))
