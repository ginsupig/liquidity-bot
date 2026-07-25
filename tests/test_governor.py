from wliq.broker.health import DataHealthMonitor
from wliq.core.config import GovernorConfig, StrategyConfig
from wliq.core.models import GovernorAction
from wliq.strategy.governor import StrategyGovernor
from tests.conftest import T0
from tests.test_strategy import fv


def make(healthy: bool = True) -> StrategyGovernor:
    h = DataHealthMonitor()
    if healthy:
        h.on_connect()
        h.on_event("MU", T0)
    g = StrategyGovernor(GovernorConfig(), StrategyConfig(), h)
    return g


def test_unhealthy_stream_blocks_everything():
    g = make(healthy=False)
    d = g.decide(fv(), T0)
    assert d.action is GovernorAction.NO_TRADE
    assert "stream_disconnected" in d.reasons


def test_reversal_allowed_on_unconfirmed_burst():
    g = make()
    assert g.decide(fv(), T0).action is GovernorAction.REVERSAL


def test_peers_confirming_blocks_reversal_enables_continuation():
    g = make()
    d = g.decide(fv(peer_confirmation=0.8, impact_decay=-0.4,
                    replenishment_skew=1.0), T0)
    assert d.action is GovernorAction.CONTINUATION


def test_catalyst_blacklist():
    g = make()
    g.cfg.catalyst_blacklist.append("MU")
    d = g.decide(fv(), T0)
    assert d.action is GovernorAction.NO_TRADE
    assert "verified_catalyst" in d.reasons


def test_open_blackout():
    g = make()
    d = g.decide(fv(minutes_since_open=0.5), T0)
    assert d.action is GovernorAction.NO_TRADE


def test_low_volume_blocks():
    g = make()
    d = g.decide(fv(volume_z=0.1), T0)
    assert d.action is GovernorAction.NO_TRADE
