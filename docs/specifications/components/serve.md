# Serve Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §11
**Status**: Active
**CLI**: `apr serve`
**Implementation**: `crates/apr-cli/src/commands/serve_plan.rs`

---

## 1. Overview

Inference server with OpenAI-compatible API, privacy tier enforcement,
failover, and circuit breakers. Plan/run subcommands.

## 2. CLI Interface

```
apr serve plan <FILE> [--port <N>] [--privacy <sovereign|private|standard>]
apr serve run <FILE> [--port <N>] [--workers <N>] [--max-batch <N>]
```

## 3. Privacy Tiers

| Tier | Description | Network |
|------|-------------|---------|
| `sovereign` | No data leaves machine, offline-only | Blocked |
| `private` | Local network only, no external | LAN only |
| `standard` | Full network access | Unrestricted |

## 4. Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming SSE responses
- Circuit breaker for GPU OOM protection
- Failover to CPU on GPU error
- Request queuing with backpressure

---

## Provable Contracts

### Contract: `serve-v1.yaml`

```yaml
metadata:
  description: "Inference server — privacy tiers, failover, circuit breakers"
  depends_on:
    - "softmax-kernel-v1"

equations:
  privacy_enforcement:
    formula: "tier(request) <= tier(server)"
    invariants:
      - "Sovereign tier blocks all outbound network"
      - "Privacy tier is enforced at bind time, not request time"
      - "Downgrade never happens silently"

  circuit_breaker:
    formula: "state ∈ {closed, open, half_open}"
    invariants:
      - "failures >= threshold → open"
      - "open + timeout elapsed → half_open"
      - "half_open + success → closed"
      - "half_open + failure → open"

  response_streaming:
    formula: "∀ token t_i: emit(t_i) before compute(t_{i+1})"
    invariants:
      - "Tokens streamed as generated"
      - "SSE format: data: {json}\n\n"
      - "Final event: data: [DONE]"

proof_obligations:
  - type: invariant
    property: "Sovereign network isolation"
    formal: "sovereign tier → zero outbound connections"
  - type: invariant
    property: "Circuit breaker state machine"
    formal: "transitions follow closed→open→half_open→closed cycle"
  - type: invariant
    property: "Streaming order"
    formal: "token emission order matches generation order"

falsification_tests:
  - id: FALSIFY-SERVE-001
    rule: "Sovereign isolation"
    prediction: "sovereign mode rejects requests requiring network"
    if_fails: "Privacy tier check missing or bypassed"
  - id: FALSIFY-SERVE-002
    rule: "Circuit breaker trip"
    prediction: "N consecutive failures → circuit opens"
    if_fails: "Failure counter not incremented"
  - id: FALSIFY-SERVE-003
    rule: "SSE format"
    prediction: "each SSE event has 'data: ' prefix and double newline"
    if_fails: "SSE formatting incorrect"
  - id: FALSIFY-SERVE-004
    rule: "Graceful shutdown"
    prediction: "in-flight requests complete before server exits"
    if_fails: "Shutdown drops active connections"
```

### Binding Requirements

```yaml
  - contract: serve-v1.yaml
    equation: privacy_enforcement
    module_path: "aprender::serve"
    function: enforce_privacy_tier
    status: implemented

  - contract: serve-v1.yaml
    equation: circuit_breaker
    module_path: "aprender::serve::failover"
    function: CircuitBreaker::check
    status: implemented
```
