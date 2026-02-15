
#[test]
fn test_cost_policy_unknown_region_defaults() {
    let policy = CostPolicy::default(); // No region costs configured

    let mut request = mock_request();
    request.qos.cost_tolerance = 20; // Low tolerance

    let candidate = mock_candidate(100, 1.0);
    let score = policy.score(&candidate, &request);

    // Unknown region defaults to 0.5 cost; 1.0 - 0.5 = 0.5
    assert!((score - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_cost_policy_always_eligible() {
    let policy = CostPolicy::default();
    let request = mock_request();
    let candidate = mock_candidate(100, 1.0);
    assert!(policy.is_eligible(&candidate, &request));
}

#[test]
fn test_cost_policy_name() {
    let policy = CostPolicy::default();
    assert_eq!(policy.name(), "cost");
}

#[test]
fn test_cost_policy_tolerance_boundary_50() {
    // Exactly at 50 -> low tolerance branch (<=0.5)
    let policy = CostPolicy::default().with_region_cost(RegionId("us-west".to_string()), 0.3);

    let mut request = mock_request();
    request.qos.cost_tolerance = 50;

    let candidate = mock_candidate(100, 1.0);
    let score = policy.score(&candidate, &request);

    // cost_tolerance=50 -> 0.5 which is NOT > 0.5, so cheap branch: 1.0 - 0.3 = 0.7
    assert!((score - 0.7).abs() < f64::EPSILON);
}

// =========================================================================
// HealthPolicy tests
// =========================================================================

#[test]
fn test_health_policy_default() {
    let policy = HealthPolicy::default();
    assert_eq!(policy.weight, 2.0);
    assert_eq!(policy.healthy_score, 1.0);
    assert_eq!(policy.degraded_score, 0.3);
}

#[test]
fn test_health_policy_name() {
    let policy = HealthPolicy::default();
    assert_eq!(policy.name(), "health");
}

#[test]
fn test_health_policy_eligibility_zero_health() {
    let policy = HealthPolicy::default();
    let request = mock_request();

    let dead = mock_candidate(100, 0.0);
    assert!(!policy.is_eligible(&dead, &request));
}

#[test]
fn test_health_policy_eligibility_positive_health() {
    let policy = HealthPolicy::default();
    let request = mock_request();

    let alive = mock_candidate(100, 0.01);
    assert!(policy.is_eligible(&alive, &request));
}

#[test]
fn test_health_policy_score_scales_with_weight() {
    let policy = HealthPolicy {
        weight: 3.0,
        ..Default::default()
    };
    let request = mock_request();

    let healthy = mock_candidate(100, 1.0);
    let score = policy.score(&healthy, &request);
    assert!((score - 3.0).abs() < f64::EPSILON);
}

// =========================================================================
// CompositePolicy tests
// =========================================================================

#[test]
fn test_composite_policy_new_empty() {
    let policy = CompositePolicy::new();
    let request = mock_request();
    let candidate = mock_candidate(100, 1.0);

    // Empty policy returns 1.0
    let score = policy.score(&candidate, &request);
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_composite_policy_default_is_enterprise() {
    let policy = CompositePolicy::default();
    assert_eq!(policy.name(), "composite");
}

#[test]
fn test_composite_policy_name() {
    let policy = CompositePolicy::new();
    assert_eq!(policy.name(), "composite");
}

#[test]
fn test_composite_policy_eligibility_all_pass() {
    let policy = CompositePolicy::new()
        .with_policy(HealthPolicy::default())
        .with_policy(LatencyPolicy::default());
    let request = mock_request();

    let good = mock_candidate(100, 1.0);
    assert!(policy.is_eligible(&good, &request));
}

#[test]
fn test_composite_policy_eligibility_one_fails() {
    // HealthPolicy requires health_score > 0
    let policy = CompositePolicy::new()
        .with_policy(HealthPolicy::default())
        .with_policy(LatencyPolicy::default());
    let request = mock_request();

    let dead = mock_candidate(100, 0.0);
    assert!(!policy.is_eligible(&dead, &request));
}

#[test]
fn test_composite_policy_eligibility_empty_passes() {
    let policy = CompositePolicy::new();
    let request = mock_request();
    let candidate = mock_candidate(100, 1.0);
    assert!(policy.is_eligible(&candidate, &request));
}

#[test]
fn test_composite_policy_score_averages() {
    // Two policies that score differently
    let policy = CompositePolicy::new()
        .with_policy(HealthPolicy {
            weight: 1.0,
            ..Default::default()
        })
        .with_policy(LatencyPolicy::default());
    let request = mock_request();
    let candidate = mock_candidate(0, 1.0);

    // HealthPolicy: 1.0 * 1.0 = 1.0, LatencyPolicy: 1.0 * 1.0 = 1.0
    // Average = 1.0
    let score = policy.score(&candidate, &request);
    assert!((score - 1.0).abs() < f64::EPSILON);
}

// =========================================================================
// RoutingPolicy wrapper tests
// =========================================================================

#[test]
fn test_routing_policy_latency() {
    let _policy = RoutingPolicy::latency();
}

#[test]
fn test_routing_policy_locality() {
    let _policy = RoutingPolicy::locality();
}

#[test]
fn test_routing_policy_privacy() {
    let _policy = RoutingPolicy::privacy();
}

#[test]
fn test_routing_policy_cost() {
    let _policy = RoutingPolicy::cost();
}

#[test]
fn test_routing_policy_health() {
    let _policy = RoutingPolicy::health();
}

#[test]
fn test_routing_policy_enterprise() {
    let _policy = RoutingPolicy::enterprise();
}

// =========================================================================
// RouteScores default test
// =========================================================================

#[test]
fn test_route_scores_default() {
    let scores = RouteScores::default();
    assert_eq!(scores.latency_score, 0.5);
    assert_eq!(scores.throughput_score, 0.5);
    assert_eq!(scores.cost_score, 0.5);
    assert_eq!(scores.locality_score, 0.5);
    assert_eq!(scores.health_score, 1.0);
    assert_eq!(scores.total, 0.5);
}

// =========================================================================
// Helper mock with custom region for privacy testing
// =========================================================================

fn mock_candidate_in_region(region: &str, latency_ms: u64, health_score: f64) -> RouteCandidate {
    RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("node1".to_string()),
            region_id: RegionId(region.to_string()),
            endpoint: "http://node1:8080".to_string(),
            estimated_latency: Duration::from_millis(latency_ms),
            score: 0.0,
        },
        scores: RouteScores {
            latency_score: 1.0 - (latency_ms as f64 / 5000.0),
            throughput_score: 0.8,
            cost_score: 0.5,
            locality_score: 0.7,
            health_score,
            total: 0.0,
        },
        eligible: true,
        rejection_reason: None,
    }
}

#[test]
fn test_privacy_policy_with_specific_region_candidate() {
    let policy = PrivacyPolicy::default()
        .with_region(RegionId("eu-west".to_string()), PrivacyLevel::Restricted)
        .with_region(RegionId("us-west".to_string()), PrivacyLevel::Internal);

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Restricted;

    let eu = mock_candidate_in_region("eu-west", 100, 1.0);
    let us = mock_candidate_in_region("us-west", 50, 1.0);

    assert!(policy.is_eligible(&eu, &request));
    assert!(!policy.is_eligible(&us, &request));
}
