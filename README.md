# wliq — Webull Residual Liquidity Exhaustion Bot (v2.0)

Institutional-grade intraday liquidity/reversion engine. Detects idiosyncratic
residual bursts (beta-adjusted vs QQQ/SMH/peers) in MU/WDC/SNDK/NVDA/KLAC,
waits for confirmed exhaustion + failed retest, and fades with strict risk
control. **One pipeline** serves replay, paper, and live.

## Architecture (v2)

```
market data ──► DataQualityMonitor ──► RawPacketRecorder / ParquetRecorder
                     │ (confidence gate, dup drop)
                     ▼
              FeatureEngine (+ ResidualDynamics, ConfirmationTimer)
                     ▼
              StrategyGovernor (regime) ──► SignalStateMachine (FSM)
                     ▼                            │ Signal
              AnomalyDetector (fail closed)       ▼
              RiskManager + PortfolioRisk ──► adaptive ExecPlan
                     ▼
              OrderManager (full lifecycle FSM, reservations, timers)
                     ▼
         PaperVenue (replay/paper)  |  WebullAdapter (live)
                     ▼
              PositionManager (cumulative partial-exit PnL)
                     ▼
     OrderJournal + RiskLedger + PositionEventLog + OUFS v1 (decisions/
     outcomes/health JSONL)  ──► recovery + reconciliation + status.html
```

The only differences between modes: the **Clock** (SimClock vs WallClock),
the venue object, and live-only interlocks/reconciliation. Replay drives the
identical `SessionEngine` — determinism is enforced by test.

### Signal FSM (Sprint 2)
IDLE → IMPULSE → CONFIRM_WAIT → EXHAUSTION → FAILED_RETEST → ENTRY → MANAGE
→ EXIT → COOLDOWN, with timeouts and invalidations (governor NO_TRADE,
fast peer confirmation = common-factor move) at every armed state. The v0.2
additive score survives as a ranking/sizing input, not the trigger.

### Order lifecycle (Sprint 0)
NEW → SUBMITTING → ACKNOWLEDGED → PARTIAL → FILLED with CANCEL_PENDING /
CANCELLED / REJECTED / EXPIRED; illegal transitions raise. Buying power and
risk are reserved at creation and released proportionally on partials; one
active entry intent per symbol; TTL cancels are clock-driven (a dead tape
still expires orders); market orders fill only from the first quote strictly
after activation latency.

### Recovery & reconciliation
`SessionEngine.recover()` rebuilds positions (PositionEventLog), open orders
(journal replay), and day risk state (RiskLedger), then reconciles against
broker truth — any live mismatch **blocks trading**. A periodic REST
reconciliation (every `reconcile_interval_s`) fail-closes to the kill switch.

## Quickstart (Windows, Python 3.11)
```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e .[dev]          # add .[ml] for LightGBM
copy config\settings.example.yaml config\settings.yaml
wliq preflight
wliq record --config config\settings.yaml     # capture a session first
wliq replay --date 2026-07-24                 # then research it
wliq walkforward --events data\events.parquet
wliq stress --events data\events.parquet      # stressed costs / stability
wliq paper                                     # paper trade the live stream
wliq dashboard                                 # logs\status.html
wliq recover                                   # dry-run crash recovery
```

## Success criteria — honest status
| Criterion | Status |
|---|---|
| Deterministic replay | ✅ enforced by `test_replay_is_deterministic` |
| Identical replay/live logic | ✅ one `SessionEngine`; modes differ by Clock/venue only |
| No duplicate orders | ✅ deterministic coids + journal dedupe + OM dup guard |
| Full restart recovery | ✅ positions/orders/risk rebuilt; tested round-trip |
| Robust reconciliation | ✅ startup + periodic, fail-closed (live path UNVERIFIED against real API) |
| Institutional execution metrics | ✅ effective/realized spread, adverse selection, IS markouts |
| Stable walk-forward performance | ⚠️ machinery ready (purged WF, embargo, stress, stability, bootstrap); **no claim of edge until run on real recorded data** |
| Positive expectancy after stressed costs | ⚠️ same — this is an empirical gate, not a feature |
| Production-ready architecture | ✅ with the known limitations below |
| Modular, documented, fully tested | ✅ 103 tests |

## Known limitations (read before trusting)
1. **Webull payload shapes UNVERIFIED** — all parsing isolated in
   `broker/webull_adapter.py`; the RawPacketRecorder exists to pin schemas on
   first real session. Do not run live before verifying.
2. **Live order/fill streaming not wired** — streaming fill processing is
   implemented for the venue-event path (paper proves the OM); the live SDK
   fill subscription must be verified against real credentials.
3. Breadth input to the governor is stubbed at 0.0.
4. Passive fill model is conservative (strict trade-through default); paper
   results understate fills, by design.
5. ML ranker deploys only if it beats the rule selector in walk-forward —
   `wliq train` prints calibration; wiring it into sizing is a deliberate
   manual step.
6. "Bayesian optimization" is implemented as successive-halving random search
   with local refinement (`ml/optimize.py`) — honest label, and every trial
   count must feed the deflated Sharpe.

## OUFS v1
All decisions/outcomes/health are appended under `logs/oufs/<date>/` per the
empire standard, alongside `orders.jsonl`, `risk_ledger.jsonl`, and
`position_events.jsonl`.
