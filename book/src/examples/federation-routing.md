# Case Study: Federation Routing Policies

This case study demonstrates intelligent routing policies for distributed ML inference. Each policy evaluates candidates and contributes to a composite score that determines the optimal node for each request.

## Overview

Routing policies answer the question: "Given multiple nodes that can handle this request, which one should we use?"

The federation gateway supports five built-in policies:

| Policy | Purpose | Default Weight |
|--------|---------|---------------|
| Health | Penalize unhealthy nodes | 2.0 |
| Latency | Prefer fast nodes | 1.0 |
| Privacy | Enforce data sovereignty | 1.0 |
| Locality | Prefer same-region nodes | 1.0 |
| Cost | Balance price vs performance | 1.0 |

## Running the Example

```bash
cargo run -p apr-cli --features inference --example federation_routing
```

## Health Policy

The health policy strongly penalizes unhealthy or degraded nodes:

```rust
use apr_cli::federation::policy::HealthPolicy;
use apr_cli::federation::traits::RoutingPolicyTrait;

let policy = HealthPolicy {
    weight: 2.0,           // Double importance
    healthy_score: 1.0,    // Full score for healthy
    degraded_score: 0.3,   // 30% for degraded
};

// Scoring
// Healthy node:  1.0 * 2.0 = 2.0
// Degraded node: 0.3 * 2.0 = 0.6
// Unhealthy:     0.0 * 2.0 = 0.0 (not eligible)
```

### Health States

| State | Description | Score |
|-------|-------------|-------|
| Healthy | All checks passing | 1.0 |
| Degraded | Some issues but operational | 0.3-0.5 |
| Unhealthy | Node failing, excluded | 0.0 |
| Unknown | No recent health data | 0.3 |

## Latency Policy

Scores nodes inversely proportional to their latency:

```rust
use apr_cli::federation::policy::LatencyPolicy;
use std::time::Duration;

let policy = LatencyPolicy {
    weight: 1.0,
    max_latency: Duration::from_secs(5),  // Nodes above this get score 0
};

// Scoring formula: 1.0 - (latency_ms / max_ms)
//
// Example with max_latency = 5000ms:
//   45ms  → 1.0 - (45/5000)   = 0.991
//   120ms → 1.0 - (120/5000)  = 0.976
//   200ms → 1.0 - (200/5000)  = 0.960
//   4000ms → 1.0 - (4000/5000) = 0.200
//   5000ms+ → 0.0 (not eligible)
```

### Eligibility

Nodes with latency exceeding `max_latency` are excluded from routing:

```rust
// This node is NOT eligible
assert!(!policy.is_eligible(&slow_candidate, &request));
```

## Privacy Policy

Enforces data sovereignty by filtering nodes based on privacy levels:

```rust
use apr_cli::federation::policy::PrivacyPolicy;
use apr_cli::federation::traits::{PrivacyLevel, RegionId};

let policy = PrivacyPolicy::default()
    .with_region(RegionId("eu-west-1".to_string()), PrivacyLevel::Confidential)
    .with_region(RegionId("us-east-1".to_string()), PrivacyLevel::Internal)
    .with_region(RegionId("ap-south-1".to_string()), PrivacyLevel::Public);
```

### Privacy Levels

| Level | Description | Example Use |
|-------|-------------|-------------|
| Public | No restrictions | Public APIs, demos |
| Internal | Company data | Internal tools |
| Confidential | Sensitive data | PII, financial |
| Restricted | Highest security | Healthcare, government |

### Eligibility Matrix

Request privacy level determines which nodes are eligible:

| Request | Public Region | Internal Region | Confidential Region |
|---------|---------------|-----------------|---------------------|
| Public | ✓ | ✓ | ✓ |
| Internal | ✗ | ✓ | ✓ |
| Confidential | ✗ | ✗ | ✓ |

```rust
// Request requires confidential handling
let request = InferenceRequest {
    qos: QoSRequirements {
        privacy: PrivacyLevel::Confidential,
        ..Default::default()
    },
    ..Default::default()
};

// Only eu-west-1 is eligible (Confidential region)
assert!(policy.is_eligible(&eu_candidate, &request));
assert!(!policy.is_eligible(&us_candidate, &request));
assert!(!policy.is_eligible(&ap_candidate, &request));
```

## Locality Policy

Prefers nodes in the same region as the request origin:

```rust
use apr_cli::federation::policy::LocalityPolicy;

let policy = LocalityPolicy {
    weight: 1.0,
    same_region_boost: 0.3,      // +30% for same region
    cross_region_penalty: 0.1,   // -10% for cross region
};

// If request originates from us-west-2:
//   us-west node: base + 0.3 = higher score
//   eu-west node: base - 0.1 = lower score
```

### Benefits

- Reduced network latency
- Lower data transfer costs
- Compliance with data residency requirements

## Cost Policy

Balances cost versus performance based on user tolerance:

```rust
use apr_cli::federation::policy::CostPolicy;

let policy = CostPolicy::default()
    .with_region_cost(RegionId("us-west-2".to_string()), 0.8)   // Expensive GPU
    .with_region_cost(RegionId("eu-west-1".to_string()), 0.6)   // Mid-tier
    .with_region_cost(RegionId("ap-south-1".to_string()), 0.3); // Budget CPU
```

### Cost Tolerance

The `cost_tolerance` field in QoS requirements controls the tradeoff:

| Tolerance | Behavior |
|-----------|----------|
| 0-30 | Strongly prefer cheap nodes |
| 31-50 | Balanced |
| 51-70 | Prefer performance |
| 71-100 | Accept premium for best performance |

```rust
// Budget-conscious request
let cheap_request = InferenceRequest {
    qos: QoSRequirements {
        cost_tolerance: 20,  // Strongly prefer cheap
        ..Default::default()
    },
    ..Default::default()
};

// Premium request (willing to pay for speed)
let premium_request = InferenceRequest {
    qos: QoSRequirements {
        cost_tolerance: 80,  // Accept expensive nodes
        ..Default::default()
    },
    ..Default::default()
};
```

## Composite Policy

Combines all policies with weighted scoring:

```rust
use apr_cli::federation::policy::CompositePolicy;

// Enterprise default combines all policies
let policy = CompositePolicy::enterprise_default();

// Custom composition
let custom = CompositePolicy::new()
    .with_policy(HealthPolicy { weight: 3.0, ..Default::default() })  // Triple health weight
    .with_policy(LatencyPolicy { weight: 2.0, ..Default::default() }) // Double latency weight
    .with_policy(PrivacyPolicy::default())
    .with_policy(CostPolicy::default());
```

### Scoring Formula

```
total_score = average(policy₁.score, policy₂.score, ..., policyₙ.score)
```

Where each policy's score is already weighted internally.

### Eligibility

A candidate must pass ALL policy eligibility checks:

```rust
impl RoutingPolicyTrait for CompositePolicy {
    fn is_eligible(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> bool {
        // Must pass ALL policies
        self.policies.iter().all(|p| p.is_eligible(candidate, request))
    }
}
```

## Custom Policies

Implement `RoutingPolicyTrait` for custom routing logic:

```rust
use apr_cli::federation::traits::{
    RoutingPolicyTrait, RouteCandidate, InferenceRequest,
};

struct TenantAffinityPolicy {
    weight: f64,
    tenant_preferences: HashMap<String, String>,  // tenant_id -> preferred_node
}

impl RoutingPolicyTrait for TenantAffinityPolicy {
    fn score(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> f64 {
        if let Some(tenant_id) = &request.tenant_id {
            if let Some(preferred) = self.tenant_preferences.get(tenant_id) {
                if candidate.target.node_id.0 == *preferred {
                    return 1.0 * self.weight;  // Strong boost for preferred node
                }
            }
        }
        0.5 * self.weight  // Neutral for non-preferred
    }

    fn is_eligible(&self, _candidate: &RouteCandidate, _request: &InferenceRequest) -> bool {
        true  // Affinity is a preference, not a hard requirement
    }

    fn name(&self) -> &str {
        "tenant_affinity"
    }
}
```

## Testing Policies

```rust
#[test]
fn test_latency_policy_scoring() {
    let policy = LatencyPolicy::default();
    let request = mock_request();

    let fast = mock_candidate(100, 1.0);   // 100ms latency
    let slow = mock_candidate(4000, 1.0);  // 4000ms latency

    let fast_score = policy.score(&fast, &request);
    let slow_score = policy.score(&slow, &request);

    assert!(fast_score > slow_score);
    assert!(fast_score > 0.9);  // Fast node scores high
}

#[test]
fn test_privacy_policy_eligibility() {
    let policy = PrivacyPolicy::default()
        .with_region(RegionId("eu".to_string()), PrivacyLevel::Confidential)
        .with_region(RegionId("us".to_string()), PrivacyLevel::Public);

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Confidential;

    // EU meets confidential requirement
    assert!(policy.is_eligible(&eu_candidate, &request));
    // US is public, doesn't meet confidential
    assert!(!policy.is_eligible(&us_candidate, &request));
}
```

## Best Practices

1. **Tune weights for your use case** - Production workloads may need different weights
2. **Monitor policy decisions** - Log which policies influenced routing
3. **Test edge cases** - Verify behavior when all nodes are degraded
4. **Consider fairness** - Ensure no node gets starved of traffic
5. **Update region costs** - Keep cost data current

## Further Reading

- [Federation Gateway](./federation-gateway.md)
- [State Machine Playbooks](./state-machine-playbooks.md)
