from wliq.core.models import DepthLevel, DepthSnapshot, Quote, Side
from tests.conftest import T0


def test_microprice_weights_toward_heavy_side():
    # Heavy bid size pulls microprice toward the ask (buy pressure).
    q = Quote("MU", T0, 100.00, 100.02, bid_size=900, ask_size=100)
    assert q.microprice > q.mid


def test_spread_bps():
    q = Quote("MU", T0, 100.00, 100.10, 100, 100)
    assert abs(q.spread_bps - 10.0) < 0.1


def test_depth_shares():
    d = DepthSnapshot("MU", T0,
                      bids=(DepthLevel(99.99, 100), DepthLevel(99.98, 200)),
                      asks=(DepthLevel(100.01, 50),))
    assert d.depth_shares(Side.BUY) == 300
    assert d.depth_shares(Side.SELL) == 50
