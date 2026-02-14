//! Routing policies for federation
//!
//! Policies determine HOW nodes are selected for inference requests.
//! Multiple policies can be composed (scored, weighted, chained).

use super::traits::*;
use std::time::Duration;

// ============================================================================
// Selection Criteria
// ============================================================================

/// Criteria for selecting nodes
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Required capability
    pub capability: Capability,
    /// Minimum health state
    pub min_health: HealthState,
    /// Maximum latency
    pub max_latency: Option<Duration>,
    /// Required privacy level
    pub min_privacy: PrivacyLevel,
    /// Preferred regions (in order)
    pub preferred_regions: Vec<RegionId>,
    /// Excluded nodes
    pub excluded_nodes: Vec<NodeId>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            capability: Capability::Generate,
            min_health: HealthState::Degraded,
            max_latency: None,
            min_privacy: PrivacyLevel::Public,
            preferred_regions: vec![],
            excluded_nodes: vec![],
        }
    }
}

// ============================================================================
// Concrete Routing Policies
// ============================================================================

/// Latency-based routing policy
///
/// Scores nodes inversely proportional to their latency.
/// Lower latency = higher score.
pub struct LatencyPolicy {
    /// Weight for this policy in composite scoring
    pub weight: f64,
    /// Maximum acceptable latency (nodes above this get score 0)
    pub max_latency: Duration,
}

impl Default for LatencyPolicy {
    fn default() -> Self {
        Self {
            weight: 1.0,
            max_latency: Duration::from_secs(5),
        }
    }
}

impl RoutingPolicyTrait for LatencyPolicy {
    fn score(&self, candidate: &RouteCandidate, _request: &InferenceRequest) -> f64 {
        let latency_ms = candidate.target.estimated_latency.as_millis() as f64;
        let max_ms = self.max_latency.as_millis() as f64;

        if latency_ms >= max_ms {
            return 0.0;
        }

        // Score: 1.0 at 0ms, 0.0 at max_latency
        let score = 1.0 - (latency_ms / max_ms);
        score * self.weight
    }

    fn is_eligible(&self, candidate: &RouteCandidate, _request: &InferenceRequest) -> bool {
        candidate.target.estimated_latency <= self.max_latency
    }

    fn name(&self) -> &'static str {
        "latency"
    }
}

/// Locality-based routing policy
///
/// Prefers nodes in the same region as the request origin.
/// Useful for data sovereignty and latency.
pub struct LocalityPolicy {
    /// Weight for this policy
    pub weight: f64,
    /// Score boost for same-region
    pub same_region_boost: f64,
    /// Score penalty for cross-region
    pub cross_region_penalty: f64,
}

impl Default for LocalityPolicy {
    fn default() -> Self {
        Self {
            weight: 1.0,
            same_region_boost: 0.3,
            cross_region_penalty: 0.1,
        }
    }
}

impl RoutingPolicyTrait for LocalityPolicy {
    fn score(&self, candidate: &RouteCandidate, _request: &InferenceRequest) -> f64 {
        // Check if request has tenant locality preference
        let base_score = 0.5;

        // For now, use locality score from candidate
        let score = base_score + candidate.scores.locality_score * self.same_region_boost;
        score * self.weight
    }

    fn is_eligible(&self, _candidate: &RouteCandidate, _request: &InferenceRequest) -> bool {
        true // Locality is a preference, not a hard requirement
    }

    fn name(&self) -> &'static str {
        "locality"
    }
}

/// Privacy-based routing policy
///
/// Enforces data sovereignty by filtering nodes based on privacy level.
#[derive(Default)]
pub struct PrivacyPolicy {
    /// Region privacy levels
    pub region_privacy: std::collections::HashMap<RegionId, PrivacyLevel>,
}

impl PrivacyPolicy {
    /// Add a region with its privacy level
    #[must_use]
    pub fn with_region(mut self, region: RegionId, level: PrivacyLevel) -> Self {
        self.region_privacy.insert(region, level);
        self
    }
}

impl RoutingPolicyTrait for PrivacyPolicy {
    fn score(&self, _candidate: &RouteCandidate, _request: &InferenceRequest) -> f64 {
        1.0 // Privacy is binary: eligible or not
    }

    fn is_eligible(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> bool {
        let region_level = self
            .region_privacy
            .get(&candidate.target.region_id)
            .copied()
            // Default to Internal - unknown regions can handle internal traffic
            .unwrap_or(PrivacyLevel::Internal);

        // Region must meet or exceed request's privacy requirement
        region_level >= request.qos.privacy
    }

    fn name(&self) -> &'static str {
        "privacy"
    }
}

/// Cost-based routing policy
///
/// Balances cost vs performance based on user tolerance.
pub struct CostPolicy {
    /// Weight for this policy
    pub weight: f64,
    /// Cost per region (0.0 = cheapest, 1.0 = most expensive)
    pub region_costs: std::collections::HashMap<RegionId, f64>,
}

impl Default for CostPolicy {
    fn default() -> Self {
        Self {
            weight: 1.0,
            region_costs: std::collections::HashMap::new(),
        }
    }
}

impl CostPolicy {
    #[must_use]
    pub fn with_region_cost(mut self, region: RegionId, cost: f64) -> Self {
        self.region_costs.insert(region, cost.clamp(0.0, 1.0));
        self
    }
}

impl RoutingPolicyTrait for CostPolicy {
    fn score(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> f64 {
        let region_cost = self
            .region_costs
            .get(&candidate.target.region_id)
            .copied()
            .unwrap_or(0.5);

        let cost_tolerance = request.qos.cost_tolerance as f64 / 100.0;

        // High tolerance = prefer fast (expensive)
        // Low tolerance = prefer cheap
        let score = if cost_tolerance > 0.5 {
            // User tolerates cost, score performance
            candidate.scores.throughput_score
        } else {
            // User wants cheap, invert cost
            1.0 - region_cost
        };

        score * self.weight
    }

    fn is_eligible(&self, _candidate: &RouteCandidate, _request: &InferenceRequest) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "cost"
    }
}

/// Health-based routing policy
///
/// Strongly penalizes unhealthy or degraded nodes.
pub struct HealthPolicy {
    /// Weight for this policy
    pub weight: f64,
    /// Score for healthy nodes
    pub healthy_score: f64,
    /// Score for degraded nodes
    pub degraded_score: f64,
}

impl Default for HealthPolicy {
    fn default() -> Self {
        Self {
            weight: 2.0, // Health is important!
            healthy_score: 1.0,
            degraded_score: 0.3,
        }
    }
}

impl RoutingPolicyTrait for HealthPolicy {
    fn score(&self, candidate: &RouteCandidate, _request: &InferenceRequest) -> f64 {
        candidate.scores.health_score * self.weight
    }

    fn is_eligible(&self, candidate: &RouteCandidate, _request: &InferenceRequest) -> bool {
        // Must have some health score
        candidate.scores.health_score > 0.0
    }

    fn name(&self) -> &'static str {
        "health"
    }
}

// ============================================================================
// Composite Policy
// ============================================================================

/// Combines multiple policies with weighted scoring
pub struct CompositePolicy {
    policies: Vec<Box<dyn RoutingPolicyTrait>>,
}

impl CompositePolicy {
    pub fn new() -> Self {
        Self { policies: vec![] }
    }

    #[must_use]
    pub fn with_policy(mut self, policy: impl RoutingPolicyTrait + 'static) -> Self {
        self.policies.push(Box::new(policy));
        self
    }

    /// Create default enterprise policy
    pub fn enterprise_default() -> Self {
        Self::new()
            .with_policy(HealthPolicy::default())
            .with_policy(LatencyPolicy::default())
            .with_policy(PrivacyPolicy::default())
            .with_policy(LocalityPolicy::default())
            .with_policy(CostPolicy::default())
    }
}

impl Default for CompositePolicy {
    fn default() -> Self {
        Self::enterprise_default()
    }
}

impl RoutingPolicyTrait for CompositePolicy {
    fn score(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> f64 {
        if self.policies.is_empty() {
            return 1.0;
        }

        let total: f64 = self
            .policies
            .iter()
            .map(|p| p.score(candidate, request))
            .sum();

        total / self.policies.len() as f64
    }

    fn is_eligible(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> bool {
        // Must pass ALL policies
        self.policies
            .iter()
            .all(|p| p.is_eligible(candidate, request))
    }

    fn name(&self) -> &'static str {
        "composite"
    }
}

// ============================================================================
// Wrapper type for export
// ============================================================================

/// Routing policy configuration
pub struct RoutingPolicy {
    #[allow(dead_code)]
    inner: Box<dyn RoutingPolicyTrait>,
}

impl RoutingPolicy {
    pub fn latency() -> Self {
        Self {
            inner: Box::new(LatencyPolicy::default()),
        }
    }

    pub fn locality() -> Self {
        Self {
            inner: Box::new(LocalityPolicy::default()),
        }
    }

    pub fn privacy() -> Self {
        Self {
            inner: Box::new(PrivacyPolicy::default()),
        }
    }

    pub fn cost() -> Self {
        Self {
            inner: Box::new(CostPolicy::default()),
        }
    }

    pub fn health() -> Self {
        Self {
            inner: Box::new(HealthPolicy::default()),
        }
    }

    pub fn enterprise() -> Self {
        Self {
            inner: Box::new(CompositePolicy::enterprise_default()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
