from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wliq.core.models import Quote, Side, Trade

T0 = datetime(2026, 7, 20, 14, 30, tzinfo=timezone.utc)  # 10:30 ET


def q(symbol: str, sec: float, bid: float, ask: float, bs: float = 500, asz: float = 500) -> Quote:
    return Quote(symbol, T0 + timedelta(seconds=sec), bid, ask, bs, asz)


def t(symbol: str, sec: float, price: float, size: float, side: Side | None = None) -> Trade:
    return Trade(symbol, T0 + timedelta(seconds=sec), price, size, side)


@pytest.fixture
def t0() -> datetime:
    return T0
