# Case Study: State Machine Playbooks

State machine playbooks define the behavior of complex systems in a declarative YAML format. This enables Extreme TDD where the specification is written first, and tests derive directly from the playbook.

## Overview

Playbooks provide:

- **Formal state definitions** - States with invariants that must hold
- **Transition rules** - Events that trigger state changes with guards
- **Forbidden transitions** - Invalid paths that should never occur
- **Test scenarios** - Executable specifications
- **Configuration** - Health, circuit breaker, routing, and performance settings

## Running Playbook Validation

```bash
probar playbook playbooks/federation-gateway.yaml --validate
```

## Playbook Structure

```yaml
version: "1.0"
name: "APR Federation Gateway"
description: "Enterprise model federation state machine"

machine:
  id: "federation_gateway"
  initial: "initializing"
  states: { ... }
  transitions: [ ... ]
  forbidden: [ ... ]

health: { ... }
circuit_breaker: { ... }
routing_policies: [ ... ]
performance: { ... }
scenarios: [ ... ]
tui: { ... }
```

## State Definitions

Each state has an ID and invariants that must always hold:

```yaml
states:
  initializing:
    id: "initializing"
    invariants:
      - description: "No models registered"
        condition: "catalog_count() == 0"
      - description: "No active requests"
        condition: "active_requests() == 0"

  ready:
    id: "ready"
    invariants:
      - description: "At least one model registered"
        condition: "catalog_count() > 0"
      - description: "At least one healthy node"
        condition: "healthy_node_count() > 0"
      - description: "Gateway accepting requests"
        condition: "gateway_status() == 'ready'"

  routing:
    id: "routing"
    invariants:
      - description: "Request in progress"
        condition: "active_requests() > 0"
      - description: "Candidate evaluation active"
        condition: "has_candidates()"

  inferring:
    id: "inferring"
    invariants:
      - description: "Target node selected"
        condition: "has_selected_node()"
      - description: "Circuit breaker allows request"
        condition: "!circuit_open_for_target()"

  streaming:
    id: "streaming"
    invariants:
      - description: "Stream active"
        condition: "active_streams() > 0"
      - description: "Tokens being generated"
        condition: "tokens_generated() >= 0"
```

## State Transitions

Transitions define how states change:

```yaml
transitions:
  # Initialization flow
  - id: "register_model"
    from: "initializing"
    to: "ready"
    event: "model_registered"
    guard: "catalog_count() >= 1"

  # Request flow
  - id: "receive_request"
    from: "ready"
    to: "routing"
    event: "inference_requested"

  - id: "select_node"
    from: "routing"
    to: "inferring"
    event: "node_selected"
    guard: "has_candidates() && !all_circuits_open()"

  - id: "no_capacity"
    from: "routing"
    to: "failed"
    event: "no_nodes_available"
    guard: "!has_candidates() || all_circuits_open()"

  # Streaming flow
  - id: "start_stream"
    from: "inferring"
    to: "streaming"
    event: "stream_started"

  - id: "complete_stream"
    from: "streaming"
    to: "completed"
    event: "stream_complete"

  # Return to ready
  - id: "return_to_ready"
    from: "completed"
    to: "ready"
    event: "response_sent"
```

### Transition Components

| Field | Description |
|-------|-------------|
| `id` | Unique transition identifier |
| `from` | Source state (or `*` for any) |
| `to` | Target state |
| `event` | Event that triggers transition |
| `guard` | Condition that must be true |

## Forbidden Transitions

Explicitly define invalid state paths:

```yaml
forbidden:
  - from: "initializing"
    to: "inferring"
    reason: "Cannot infer without registered models"

  - from: "circuit_open"
    to: "inferring"
    reason: "Cannot infer through open circuit"

  - from: "streaming"
    to: "routing"
    reason: "Cannot re-route during active stream"

  - from: "failed"
    to: "inferring"
    reason: "Must acknowledge failure before new inference"
```

## Circuit Breaker State Machine

The circuit breaker follows a standard pattern:

```yaml
circuit_breaker:
  failure_threshold: 5      # Failures to open
  reset_timeout_ms: 30000   # Time in open state
  half_open_successes: 3    # Successes to close

# Circuit breaker states are defined in the main machine
states:
  circuit_open:
    id: "circuit_open"
    invariants:
      - description: "Node marked unhealthy"
        condition: "circuit_state() == 'open'"
      - description: "Reset timeout pending"
        condition: "reset_timeout_remaining() > 0"

  circuit_half_open:
    id: "circuit_half_open"
    invariants:
      - description: "Probe request allowed"
        condition: "circuit_state() == 'half_open'"
      - description: "Single request permitted"
        condition: "probe_requests_allowed() == 1"

# Circuit breaker transitions
transitions:
  - id: "open_circuit"
    from: "*"
    to: "circuit_open"
    event: "failure_threshold_exceeded"
    guard: "consecutive_failures() >= failure_threshold()"

  - id: "half_open_circuit"
    from: "circuit_open"
    to: "circuit_half_open"
    event: "reset_timeout_elapsed"

  - id: "close_circuit"
    from: "circuit_half_open"
    to: "ready"
    event: "probe_succeeded"

  - id: "reopen_circuit"
    from: "circuit_half_open"
    to: "circuit_open"
    event: "probe_failed"
```

## Routing Policy Configuration

```yaml
routing_policies:
  - name: "health"
    weight: 2.0
    description: "Strongly penalize unhealthy nodes"

  - name: "latency"
    weight: 1.0
    max_latency_ms: 5000
    description: "Prefer low-latency nodes"

  - name: "privacy"
    weight: 1.0
    default_level: "internal"
    description: "Enforce data sovereignty"

  - name: "locality"
    weight: 1.0
    same_region_boost: 0.3
    description: "Prefer same-region nodes"

  - name: "cost"
    weight: 1.0
    description: "Balance cost vs performance"
```

## Performance Assertions

Define performance budgets with critical thresholds:

```yaml
performance:
  max_routing_ms: 10
  max_retry_backoff_ms: 1000
  max_total_latency_ms: 30000
  target_success_rate: 0.99

performance_assertions:
  - name: "routing_latency"
    condition: "routing_latency_ms() <= 10"
    critical: "routing_latency_ms() <= 50"
    failure_reason: "Routing decision too slow"

  - name: "success_rate"
    condition: "success_rate() >= 0.99"
    critical: "success_rate() >= 0.95"
    failure_reason: "Success rate below threshold"

  - name: "circuit_recovery"
    condition: "mean_recovery_time_ms() <= 60000"
    failure_reason: "Circuit recovery too slow"
```

## Test Scenarios

Scenarios are executable specifications:

```yaml
scenarios:
  - name: "happy_path"
    description: "Normal request flow"
    steps:
      - action: "register_model"
        params: { model: "whisper-v3", node: "us-west-1", capability: "transcribe" }
      - action: "start_health_monitoring"
      - action: "send_request"
        params: { capability: "transcribe" }
      - assert: "state == 'completed'"
      - assert: "stats_total() == 1"

  - name: "retry_success"
    description: "Request succeeds after retry"
    steps:
      - action: "register_model"
        params: { model: "llama-70b", node: "us-east-1", capability: "generate" }
      - action: "register_model"
        params: { model: "llama-70b", node: "eu-west-1", capability: "generate" }
      - action: "fail_node"
        params: { node: "us-east-1" }
      - action: "send_request"
        params: { capability: "generate" }
      - assert: "retry_count() == 1"
      - assert: "state == 'completed'"

  - name: "circuit_breaker_trip"
    description: "Circuit opens after failures"
    steps:
      - action: "register_model"
        params: { model: "embed", node: "node-1", capability: "embed" }
      - repeat: 5
        action: "record_failure"
        params: { node: "node-1" }
      - assert: "circuit_state('node-1') == 'open'"
      - assert: "circuit_is_open('node-1')"
```

### Scenario Actions

| Action | Description |
|--------|-------------|
| `register_model` | Register a model on a node |
| `start_health_monitoring` | Start health checks |
| `send_request` | Send inference request |
| `fail_node` | Simulate node failure |
| `record_failure` | Record failure for circuit breaker |
| `wait` | Wait for specified duration |

### Assertions

Assertions verify system state after actions:

```yaml
- assert: "state == 'completed'"
- assert: "retry_count() == 1"
- assert: "selected_node() == 'us-west'"
- assert: "routing_reason() contains 'latency'"
- assert: "circuit_is_open('node-1')"
```

## TUI Dashboard Configuration

Define TUI panels and keybindings:

```yaml
tui:
  refresh_rate_ms: 100

  panels:
    - id: "catalog"
      title: "MODEL CATALOG"
      columns: ["Model", "Node", "Region", "Capabilities", "Status"]

    - id: "health"
      title: "NODE HEALTH"
      columns: ["Node", "State", "Latency P50", "Latency P99", "Queue"]

    - id: "routing"
      title: "ROUTING DECISIONS"
      columns: ["Request", "Capability", "Selected", "Score", "Reason"]

    - id: "circuits"
      title: "CIRCUIT BREAKERS"
      columns: ["Node", "State", "Failures", "Last Failure", "Reset In"]

  status_bar:
    left: "Federation Gateway v1.0"
    center: "{{healthy_nodes}}/{{total_nodes}} nodes healthy"
    right: "{{requests_per_sec}} req/s | {{success_rate}}% success"

  keybindings:
    q: "quit"
    r: "refresh"
    h: "toggle_health_panel"
    c: "toggle_circuit_panel"
    s: "toggle_stats_panel"
    "?": "help"
```

## Deriving Tests from Playbooks

The playbook drives test generation:

```rust
use jugar_probar::playbook::{Playbook, PlaybookRunner};

#[test]
fn test_playbook_scenarios() {
    let playbook = Playbook::from_file("playbooks/federation-gateway.yaml")
        .expect("load playbook");

    let runner = PlaybookRunner::new(&playbook);

    for scenario in playbook.scenarios() {
        let result = runner.run_scenario(&scenario);
        assert!(result.passed, "Scenario '{}' failed: {}",
            scenario.name, result.error.unwrap_or_default());
    }
}

#[test]
fn test_invariants_hold() {
    let playbook = Playbook::from_file("playbooks/federation-gateway.yaml")
        .expect("load playbook");

    let runner = PlaybookRunner::new(&playbook);

    // Run through all transitions and verify invariants
    for transition in playbook.transitions() {
        runner.apply_transition(&transition);
        let state = runner.current_state();

        for invariant in state.invariants() {
            assert!(runner.evaluate(&invariant.condition),
                "Invariant '{}' violated in state '{}'",
                invariant.description, state.id);
        }
    }
}

#[test]
fn test_forbidden_paths() {
    let playbook = Playbook::from_file("playbooks/federation-gateway.yaml")
        .expect("load playbook");

    for forbidden in playbook.forbidden() {
        let runner = PlaybookRunner::new(&playbook);
        runner.force_state(&forbidden.from);

        let result = runner.try_transition_to(&forbidden.to);
        assert!(result.is_err(),
            "Forbidden transition from '{}' to '{}' was allowed",
            forbidden.from, forbidden.to);
    }
}
```

## Best Practices

1. **Write playbook first** - Define behavior before implementation
2. **Keep invariants simple** - Each invariant tests one property
3. **Test edge cases** - Cover retry limits, circuit trips, degradation
4. **Use forbidden transitions** - Explicitly disallow invalid paths
5. **Performance budgets** - Define SLAs as assertions
6. **Document scenarios** - Clear descriptions for each test case

## Further Reading

- [Federation Gateway](./federation-gateway.md)
- [Federation Routing Policies](./federation-routing.md)
- [Probar TUI Testing](./probar-tui-testing.md)
