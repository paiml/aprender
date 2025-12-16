# Case Study: Federation Gateway

The Federation Gateway provides enterprise-grade model routing across distributed infrastructure. This case study demonstrates building a fault-tolerant, policy-based routing system using Extreme TDD principles.

## Overview

The Federation Gateway solves the challenge of routing ML inference requests across multiple nodes, regions, and model deployments. Key features include:

- **Multi-region model registration** - Deploy models across geographic regions
- **Health monitoring** - Track node health with latency percentiles
- **Circuit breakers** - Automatic fault isolation
- **Policy-based routing** - Intelligent node selection
- **Streaming inference** - Real-time token streaming

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Federation Gateway                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Catalog  │  │ Health   │  │ Circuit  │  │   Router     │    │
│  │          │  │ Checker  │  │ Breaker  │  │              │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│        │            │             │               │             │
│        └────────────┴─────────────┴───────────────┘             │
│                            │                                    │
│                    ┌───────┴───────┐                            │
│                    │  Composite    │                            │
│                    │   Policy      │                            │
│                    └───────────────┘                            │
│                            │                                    │
│        ┌───────────────────┼───────────────────┐                │
│        ▼                   ▼                   ▼                │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐            │
│  │ us-west  │       │ eu-west  │       │ ap-south │            │
│  │   GPU    │       │   GPU    │       │   CPU    │            │
│  └──────────┘       └──────────┘       └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cargo run -p apr-cli --features inference --example federation_gateway
```

## Core Components

### Model Catalog

The catalog tracks which models are available and where they're deployed:

```rust
use apr_cli::federation::{
    ModelCatalog, ModelCatalogTrait, ModelId, NodeId, RegionId, Capability,
};

let catalog = Arc::new(ModelCatalog::new());

// Register a model across multiple regions
catalog.register(
    ModelId("whisper-large-v3".to_string()),
    NodeId("us-west-gpu-01".to_string()),
    RegionId("us-west-2".to_string()),
    vec![Capability::Transcribe],
).await?;

catalog.register(
    ModelId("whisper-large-v3".to_string()),
    NodeId("eu-west-gpu-01".to_string()),
    RegionId("eu-west-1".to_string()),
    vec![Capability::Transcribe],
).await?;
```

### Health Monitoring

Track node health with latency metrics:

```rust
use apr_cli::federation::{HealthChecker, NodeId};
use std::time::Duration;

let health = Arc::new(HealthChecker::default());

// Register and report health
health.register_node(NodeId("us-west-gpu-01".to_string()));
health.report_success(
    &NodeId("us-west-gpu-01".to_string()),
    Duration::from_millis(45)
);

// Check health status
let statuses = health.all_statuses();
for status in statuses {
    println!("{}: {:?} (P50: {}ms)",
        status.node_id.0,
        status.state,
        status.latency_p50.as_millis()
    );
}
```

### Circuit Breaker

Automatic fault isolation when nodes fail:

```rust
use apr_cli::federation::{CircuitBreaker, CircuitBreakerTrait, NodeId};

let cb = Arc::new(CircuitBreaker::default());

// Record failures
for _ in 0..5 {
    cb.record_failure(&NodeId("problem-node".to_string()));
}

// Circuit is now open - node excluded from routing
assert!(cb.is_open(&NodeId("problem-node".to_string())));

// After timeout, circuit enters half-open state
// A successful probe closes the circuit
cb.record_success(&NodeId("problem-node".to_string()));
```

### Gateway Builder

Create a fully configured gateway:

```rust
use apr_cli::federation::{
    GatewayBuilder, GatewayConfig, GatewayTrait,
    InferenceRequest, Capability, QoSRequirements,
};
use std::time::Duration;

let gateway = GatewayBuilder::new()
    .config(GatewayConfig {
        max_retries: 3,
        retry_delay: Duration::from_millis(100),
        request_timeout: Duration::from_secs(30),
    })
    .build();

// Execute inference
let request = InferenceRequest {
    capability: Capability::Transcribe,
    input: audio_data,
    qos: QoSRequirements::default(),
    request_id: "req-001".to_string(),
    tenant_id: Some("acme-corp".to_string()),
};

let response = gateway.infer(&request).await?;
println!("Routed to: {} (score: {:.2})", response.node_id.0, response.score);
```

## Routing Policies

The gateway uses a composite policy combining multiple factors:

| Policy | Weight | Description |
|--------|--------|-------------|
| Health | 2.0 | Strongly penalize unhealthy nodes |
| Latency | 1.0 | Prefer low-latency nodes |
| Privacy | 1.0 | Enforce data sovereignty |
| Locality | 1.0 | Prefer same-region nodes |
| Cost | 1.0 | Balance cost vs performance |

```rust
use apr_cli::federation::policy::{
    CompositePolicy, HealthPolicy, LatencyPolicy, PrivacyPolicy,
};

// Create enterprise default policy
let policy = CompositePolicy::enterprise_default();

// Or customize
let custom = CompositePolicy::new()
    .with_policy(HealthPolicy { weight: 3.0, ..Default::default() })
    .with_policy(LatencyPolicy::default())
    .with_policy(PrivacyPolicy::default());
```

## State Machine

The gateway follows a well-defined state machine:

```
                    ┌─────────────┐
                    │ initializing│
                    └──────┬──────┘
                           │ model_registered
                           ▼
    ┌──────────────────► ready ◄──────────────────┐
    │                      │                       │
    │     inference_requested                      │
    │                      ▼                       │
    │                  routing                     │
    │                      │                       │
    │        ┌─────────────┴─────────────┐         │
    │        │                           │         │
    │  node_selected            no_nodes_available │
    │        ▼                           ▼         │
    │    inferring ───────────────► failed ────────┤
    │        │                                     │
    │  ┌─────┴─────┐                               │
    │  │           │                               │
    │  ▼           ▼                               │
    │ streaming  completed                         │
    │  │           │                               │
    │  └─────┬─────┘                               │
    │        │ response_sent                       │
    └────────┴─────────────────────────────────────┘
```

## Observability

Track gateway metrics:

```rust
let stats = gateway.stats();

println!("Total Requests:  {}", stats.total_requests);
println!("Successful:      {}", stats.successful_requests);
println!("Failed:          {}", stats.failed_requests);
println!("Success Rate:    {:.1}%",
    stats.successful_requests as f64 / stats.total_requests as f64 * 100.0);
println!("Total Tokens:    {}", stats.total_tokens);
println!("Avg Latency:     {:?}", stats.avg_latency);
```

## Testing

The federation module includes comprehensive tests:

```bash
# Run all federation tests
cargo test -p apr-cli --features inference federation

# Run specific test
cargo test -p apr-cli --features inference test_full_federation_flow
```

### Test Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| Catalog | 5 | Registration, deregistration, multi-deployment |
| Health | 8 | State transitions, latency tracking |
| Circuit Breaker | 5 | Open/close/half-open states |
| Router | 6 | Policy scoring, candidate selection |
| Gateway | 10 | Full integration, streaming, retries |
| TUI | 20+ | Probar frame tests, UX coverage |

## Best Practices

1. **Always register health** - Register nodes before reporting health
2. **Set appropriate timeouts** - Balance between reliability and latency
3. **Monitor circuit breakers** - Alert when circuits open
4. **Use tenant IDs** - Enable per-tenant routing and metrics
5. **Test failure scenarios** - Verify retry and circuit breaker behavior

## Further Reading

- [Federation Routing Policies](./federation-routing.md)
- [Probar TUI Testing](./probar-tui-testing.md)
- [State Machine Playbooks](./state-machine-playbooks.md)
