import pytest
from pydantic import ValidationError

from wliq.core.config import Settings


def test_live_requires_double_opt_in():
    with pytest.raises(ValidationError, match="double opt-in"):
        Settings(symbols=["MU"], mode="live", live_trading_enabled=False,
                 environment="production")


def test_live_blocked_in_sandbox():
    with pytest.raises(ValidationError, match="sandbox"):
        Settings(symbols=["MU"], mode="live", live_trading_enabled=True,
                 environment="sandbox")


def test_defaults_are_safe():
    s = Settings(symbols=["MU"])
    assert s.mode == "record"
    assert s.live_trading_enabled is False
    assert s.environment == "sandbox"


def test_streamed_universe_includes_etfs_and_peers():
    s = Settings(symbols=["MU"], benchmarks={"MU": ["WDC", "SNDK"]})
    u = s.all_streamed_symbols()
    assert set(u) >= {"MU", "WDC", "SNDK", "QQQ", "SMH"}
