"""Webull OpenAPI boundary.

ALL SDK-specific parsing lives here. The rest of the codebase only sees
normalized Quote / Trade / DepthSnapshot / Bar / Noii models.

UNVERIFIED-FIELD POLICY: where the official SDK payload shape could not be
confirmed against live sandbox output, the parsing is (a) defensive over
plausible aliases, (b) marked with an UNVERIFIED comment, and (c) returns None
rather than fabricating values. Nothing downstream depends on unverified
fields being present.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from wliq.broker.health import DataHealthMonitor
from wliq.core.config import Settings
from wliq.core.models import (Bar, DepthLevel, DepthSnapshot, Noii, OrderRequest,
                              Quote, Side, Trade)


def _f(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


class WebullAdapter:
    def __init__(self, settings: Settings, health: DataHealthMonitor | None = None) -> None:
        if not settings.app_key or not settings.app_secret:
            raise ValueError("WEBULL_APP_KEY and WEBULL_APP_SECRET are required")
        self.settings = settings
        self.health = health or DataHealthMonitor()
        self.on_quote: Callable[[Quote], None] = lambda _: None
        self.on_trade: Callable[[Trade], None] = lambda _: None
        self.on_depth: Callable[[DepthSnapshot], None] = lambda _: None
        self.on_noii: Callable[[Noii], None] = lambda _: None
        self._api_client: Any = None
        self._trade_client: Any = None

    # -------------------------------------------------------------- parsing
    @staticmethod
    def _dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            seconds = value / 1000 if value > 10_000_000_000 else value
            return datetime.fromtimestamp(seconds, timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _payload_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, (bytes, str)):
            return json.loads(payload)
        if hasattr(payload, "to_dict"):
            return payload.to_dict()
        if hasattr(payload, "__dict__"):
            return vars(payload)
        raise TypeError(f"Unsupported Webull payload: {type(payload)!r}")

    def parse_quote(self, d: dict[str, Any]) -> Quote | None:
        symbol = str(d.get("symbol") or d.get("ticker") or d.get("code") or "").upper()
        bid = _f(d, "bid", "bid_price", "bidPrice")
        ask = _f(d, "ask", "ask_price", "askPrice")
        if not symbol or bid <= 0 or ask < bid:
            return None
        seq = d.get("seq") or d.get("sequence")
        return Quote(symbol, self._dt(d.get("timestamp") or d.get("time")),
                     bid, ask, _f(d, "bid_size", "bidSize"), _f(d, "ask_size", "askSize"),
                     seq=int(seq) if seq is not None else None)

    def parse_trade(self, d: dict[str, Any]) -> Trade | None:
        symbol = str(d.get("symbol") or d.get("ticker") or d.get("code") or "").upper()
        price = _f(d, "price", "trade_price", "deal_price")
        size = _f(d, "size", "volume", "quantity")
        if not symbol or price <= 0 or size <= 0:
            return None
        direction = str(d.get("direction") or d.get("side") or "").upper()
        aggressor = (Side.BUY if direction in {"BUY", "B", "UP"}
                     else Side.SELL if direction in {"SELL", "S", "DOWN"} else None)
        seq = d.get("seq") or d.get("sequence")
        return Trade(symbol, self._dt(d.get("timestamp") or d.get("time") or d.get("trade_time")),
                     price, size, aggressor, seq=int(seq) if seq is not None else None)

    def parse_depth(self, d: dict[str, Any]) -> DepthSnapshot | None:
        # UNVERIFIED: exact L2 payload key names ("bids"/"asks" vs "bidList"/
        # "askList", level dicts vs arrays) are not confirmed. Defensive only.
        symbol = str(d.get("symbol") or d.get("ticker") or d.get("code") or "").upper()
        raw_bids = d.get("bids") or d.get("bidList") or d.get("bid_levels") or []
        raw_asks = d.get("asks") or d.get("askList") or d.get("ask_levels") or []
        if not symbol or not raw_bids or not raw_asks:
            return None

        def levels(raw: Any) -> tuple[DepthLevel, ...]:
            out = []
            for lvl in raw:
                if isinstance(lvl, dict):
                    p, s = _f(lvl, "price", "p"), _f(lvl, "size", "volume", "s")
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    p, s = float(lvl[0]), float(lvl[1])
                else:
                    continue
                if p > 0 and s > 0:
                    out.append(DepthLevel(p, s))
            return tuple(out)

        bids, asks = levels(raw_bids), levels(raw_asks)
        if not bids or not asks:
            return None
        return DepthSnapshot(symbol, self._dt(d.get("timestamp") or d.get("time")), bids, asks)

    def parse_noii(self, d: dict[str, Any]) -> Noii | None:
        # UNVERIFIED: Webull NOII payload shape not confirmed against sandbox.
        symbol = str(d.get("symbol") or d.get("ticker") or "").upper()
        if not symbol:
            return None
        side_raw = str(d.get("imbalance_side") or d.get("imbalanceSide") or "").upper()
        side = (Side.BUY if side_raw in {"BUY", "B"}
                else Side.SELL if side_raw in {"SELL", "S"} else None)
        return Noii(
            symbol=symbol, ts=self._dt(d.get("timestamp") or d.get("time")),
            paired_qty=_f(d, "paired_qty", "pairedQty"),
            imbalance_qty=_f(d, "imbalance_qty", "imbalanceQty"),
            imbalance_side=side,
            indicative_price=_f(d, "indicative_price", "indicativePrice") or None,
            near_price=_f(d, "near_price", "nearPrice") or None,
            far_price=_f(d, "far_price", "farPrice") or None,
            auction=str(d.get("auction") or "CLOSE").upper(),
        )

    def handle_stream_message(self, topic: str, payload: Any) -> None:
        try:
            d = self._payload_dict(payload)
        except (TypeError, json.JSONDecodeError):
            self.health.dropped_messages += 1
            return
        low = topic.lower()
        if "depth" in low or "book" in low:
            if (snap := self.parse_depth(d)) is not None:
                self.health.on_event(snap.symbol, snap.ts, snap.seq)
                self.on_depth(snap)
        elif "noii" in low or "auction" in low:
            if (n := self.parse_noii(d)) is not None:
                self.on_noii(n)
        elif "quote" in low or {"bid", "ask"} & set(d):
            if (q := self.parse_quote(d)) is not None:
                self.health.on_event(q.symbol, q.ts, q.seq)
                self.on_quote(q)
        elif "tick" in low or "trade" in low:
            if (t := self.parse_trade(d)) is not None:
                self.health.on_event(t.symbol, t.ts, t.seq)
                self.on_trade(t)

    # ------------------------------------------------------------- streaming
    def run_stream(self) -> None:
        from webull.data.common.category import Category
        from webull.data.common.subscribe_type import SubscribeType
        from webull.data.data_streaming_client import DataStreamingClient

        session_id = os.getenv("WEBULL_SESSION_ID", "wliq_session")
        client = DataStreamingClient(
            self.settings.app_key, self.settings.app_secret, "us", session_id,
            http_host=self.settings.api_host, mqtt_host=self.settings.mqtt_host,
        )
        symbols = self.settings.all_streamed_symbols()

        def connected(c: Any, api_client: Any, sid: Any) -> None:
            self.health.on_connect()
            types = list(self.settings.stream_types)
            if self.settings.subscribe_depth and "DEPTH" not in types:
                types.append("DEPTH")  # UNVERIFIED: depth SubscribeType name
            valid = [t for t in types if hasattr(SubscribeType, t)]
            c.subscribe(symbols, Category.US_STOCK.name, [getattr(SubscribeType, t).name for t in valid])

        client.on_connect_success = connected
        client.on_connect_failed = lambda *a: self.health.on_disconnect()
        client.on_quotes_message = lambda c, topic, msg: self.handle_stream_message(topic, msg)
        client.connect_and_loop_forever()

    # ------------------------------------------------------ historical bars
    def download_bars(self, symbol: str, timespan: str = "M1", count: int = 500) -> list[Bar]:
        from webull.core.client import ApiClient
        from webull.mdata.quotes_client import QuotesClient  # UNVERIFIED module path

        api = self._api()
        quotes = QuotesClient(api)
        resp = quotes.get_history_bar(symbol, "US_STOCK", timespan, str(count))
        if resp.status_code != 200:
            raise RuntimeError(f"bars download failed {symbol}: {resp.status_code} {resp.text}")
        rows = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])
        bars: list[Bar] = []
        for r in rows:
            d = self._payload_dict(r)
            bars.append(Bar(symbol, self._dt(d.get("time") or d.get("timestamp")),
                            _f(d, "open"), _f(d, "high"), _f(d, "low"), _f(d, "close"),
                            _f(d, "volume"), timespan))
        return bars

    # ------------------------------------------------------------ trade API
    def _api(self) -> Any:
        if self._api_client is None:
            from webull.core.client import ApiClient
            self._api_client = ApiClient(self.settings.app_key, self.settings.app_secret, "us")
            self._api_client.add_endpoint("us", self.settings.api_host)
        return self._api_client

    def _trade(self) -> Any:
        if self._trade_client is None:
            from webull.trade.trade_client import TradeClient
            self._trade_client = TradeClient(self._api())
        return self._trade_client

    def _assert_live(self) -> None:
        if self.settings.mode != "live" or not self.settings.live_trading_enabled:
            raise RuntimeError("Live order path blocked: requires mode=live AND "
                               "live_trading_enabled=true")

    def place_order(self, order: OrderRequest) -> dict[str, Any]:
        self._assert_live()
        payload = [{
            "combo_type": "NORMAL",
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": order.order_type,
            "quantity": str(int(order.quantity)),
            "support_trading_session": "CORE",
            "side": order.side.value,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            **({"limit_price": f"{order.limit_price:.2f}"} if order.limit_price is not None else {}),
        }]
        resp = self._trade().order_v3.place_order(self.settings.account_id, payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Webull order failed: {resp.status_code} {resp.text}")
        return resp.json()

    def cancel_order(self, client_order_id: str) -> dict[str, Any]:
        self._assert_live()
        resp = self._trade().order_v3.cancel_order(self.settings.account_id, client_order_id)
        if resp.status_code != 200:
            raise RuntimeError(f"Webull cancel failed: {resp.status_code} {resp.text}")
        return resp.json()

    def replace_order(self, order: OrderRequest) -> dict[str, Any]:
        """Cancel-then-new replacement. UNVERIFIED: native replace endpoint
        semantics not confirmed; explicit cancel+resubmit is used instead."""
        self._assert_live()
        self.cancel_order(order.client_order_id)
        return self.place_order(order)

    # broker reads for the reconciler ------------------------------------- #
    def read_positions(self) -> dict[str, float]:
        resp = self._trade().account_v2.get_account_position(self.settings.account_id)
        if resp.status_code != 200:
            raise RuntimeError(f"position read failed: {resp.status_code}")
        out: dict[str, float] = {}
        data = resp.json()
        items = data.get("positions") or data.get("data") or []
        for item in items:
            d = self._payload_dict(item)
            sym = str(d.get("symbol") or d.get("ticker") or "").upper()
            qty = _f(d, "quantity", "qty", "position")
            side = str(d.get("side") or "LONG").upper()
            if sym:
                out[sym] = qty if side != "SHORT" else -qty
        return out

    def read_open_orders(self) -> list[str]:
        resp = self._trade().order_v3.list_open_orders(self.settings.account_id)  # UNVERIFIED name
        if resp.status_code != 200:
            raise RuntimeError(f"open-order read failed: {resp.status_code}")
        data = resp.json()
        items = data.get("orders") or data.get("data") or []
        return [str(self._payload_dict(i).get("client_order_id", "")) for i in items]

    def read_account_ok(self) -> bool:
        resp = self._trade().account_v2.get_account_balance(self.settings.account_id)
        return resp.status_code == 200
