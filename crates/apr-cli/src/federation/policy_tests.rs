use super::*;

fn mock_candidate(latency_ms: u64, health_score: f64) -> RouteCandidate {
    RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("node1".to_string()),
            region_id: RegionId("us-west".to_string()),
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

fn mock_request() -> InferenceRequest {
    InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "req-1".to_string(),
        tenant_id: None,
    }
}

#[test]
fn test_latency_policy_scoring() {
    let policy = LatencyPolicy::default();
    let request = mock_request();

    // Fast node should score higher
    let fast = mock_candidate(100, 1.0);
    let slow = mock_candidate(4000, 1.0);

    let fast_score = policy.score(&fast, &request);
    let slow_score = policy.score(&slow, &request);

    assert!(fast_score > slow_score);
    assert!(fast_score > 0.9); // 100ms out of 5000ms max
}

#[test]
fn test_latency_policy_eligibility() {
    let policy = LatencyPolicy {
        max_latency: Duration::from_secs(2),
        ..Default::default()
    };
    let request = mock_request();

    let fast = mock_candidate(1000, 1.0);
    let slow = mock_candidate(3000, 1.0);

    assert!(policy.is_eligible(&fast, &request));
    assert!(!policy.is_eligible(&slow, &request));
}

#[test]
fn test_health_policy_scoring() {
    let policy = HealthPolicy::default();
    let request = mock_request();

    let healthy = mock_candidate(100, 1.0);
    let degraded = mock_candidate(100, 0.3);

    let healthy_score = policy.score(&healthy, &request);
    let degraded_score = policy.score(&degraded, &request);

    assert!(healthy_score > degraded_score);
}

#[test]
fn test_composite_policy() {
    let policy = CompositePolicy::enterprise_default();
    let request = mock_request();

    let good = mock_candidate(100, 1.0);
    let bad = mock_candidate(4000, 0.2);

    let good_score = policy.score(&good, &request);
    let bad_score = policy.score(&bad, &request);

    assert!(good_score > bad_score);
}

#[test]
fn test_privacy_policy_eligibility() {
    let policy = PrivacyPolicy::default()
        .with_region(RegionId("eu-west".to_string()), PrivacyLevel::Confidential)
        .with_region(RegionId("us-east".to_string()), PrivacyLevel::Public);

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Confidential;

    let eu_candidate = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("node-eu".to_string()),
            region_id: RegionId("eu-west".to_string()),
            endpoint: "http://eu:8080".to_string(),
            estimated_latency: Duration::from_millis(100),
            score: 0.0,
        },
        scores: RouteScores::default(),
        eligible: true,
        rejection_reason: None,
    };

    let us_candidate = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("node-us".to_string()),
            region_id: RegionId("us-east".to_string()),
            endpoint: "http://us:8080".to_string(),
            estimated_latency: Duration::from_millis(50),
            score: 0.0,
        },
        scores: RouteScores::default(),
        eligible: true,
        rejection_reason: None,
    };

    // EU meets confidential requirement
    assert!(policy.is_eligible(&eu_candidate, &request));
    // US is public, doesn't meet confidential
    assert!(!policy.is_eligible(&us_candidate, &request));
}

// =========================================================================
// SelectionCriteria tests
// =========================================================================

#[test]
fn test_selection_criteria_default() {
    let criteria = SelectionCriteria::default();
    assert!(matches!(criteria.capability, Capability::Generate));
    assert_eq!(criteria.min_health, HealthState::Degraded);
    assert!(criteria.max_latency.is_none());
    assert_eq!(criteria.min_privacy, PrivacyLevel::Public);
    assert!(criteria.preferred_regions.is_empty());
    assert!(criteria.excluded_nodes.is_empty());
}

#[test]
fn test_selection_criteria_custom() {
    let criteria = SelectionCriteria {
        capability: Capability::Transcribe,
        min_health: HealthState::Healthy,
        max_latency: Some(Duration::from_secs(2)),
        min_privacy: PrivacyLevel::Confidential,
        preferred_regions: vec![RegionId("eu-west".to_string())],
        excluded_nodes: vec![NodeId("bad-node".to_string())],
    };
    assert!(matches!(criteria.capability, Capability::Transcribe));
    assert_eq!(criteria.max_latency, Some(Duration::from_secs(2)));
    assert_eq!(criteria.preferred_regions.len(), 1);
    assert_eq!(criteria.excluded_nodes.len(), 1);
}

// =========================================================================
// LatencyPolicy edge cases
// =========================================================================

#[test]
fn test_latency_policy_at_max() {
    let policy = LatencyPolicy::default(); // max 5000ms
    let request = mock_request();

    // Exactly at max latency -> score 0
    let at_max = mock_candidate(5000, 1.0);
    let score = policy.score(&at_max, &request);
    assert_eq!(score, 0.0);
}

#[test]
fn test_latency_policy_beyond_max() {
    let policy = LatencyPolicy::default();
    let request = mock_request();

    let beyond = mock_candidate(6000, 1.0);
    let score = policy.score(&beyond, &request);
    assert_eq!(score, 0.0);
}

#[test]
fn test_latency_policy_zero_latency() {
    let policy = LatencyPolicy::default();
    let request = mock_request();

    let zero = mock_candidate(0, 1.0);
    let score = policy.score(&zero, &request);
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_latency_policy_name() {
    let policy = LatencyPolicy::default();
    assert_eq!(policy.name(), "latency");
}

#[test]
fn test_latency_policy_custom_weight() {
    let policy = LatencyPolicy {
        weight: 2.0,
        max_latency: Duration::from_secs(5),
    };
    let request = mock_request();
    let candidate = mock_candidate(0, 1.0);

    let score = policy.score(&candidate, &request);
    assert!((score - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_latency_policy_eligibility_at_boundary() {
    let policy = LatencyPolicy {
        max_latency: Duration::from_millis(1000),
        ..Default::default()
    };
    let request = mock_request();

    // Exactly at boundary
    let at_boundary = mock_candidate(1000, 1.0);
    assert!(policy.is_eligible(&at_boundary, &request));

    // Just over
    let over = mock_candidate(1001, 1.0);
    assert!(!policy.is_eligible(&over, &request));
}

// =========================================================================
// LocalityPolicy tests
// =========================================================================

#[test]
fn test_locality_policy_default() {
    let policy = LocalityPolicy::default();
    assert_eq!(policy.weight, 1.0);
    assert_eq!(policy.same_region_boost, 0.3);
    assert_eq!(policy.cross_region_penalty, 0.1);
}

#[test]
fn test_locality_policy_scoring() {
    let policy = LocalityPolicy::default();
    let request = mock_request();

    let high_locality = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("us-west".to_string()),
            endpoint: String::new(),
            estimated_latency: Duration::from_millis(50),
            score: 0.0,
        },
        scores: RouteScores {
            latency_score: 0.9,
            throughput_score: 0.8,
            cost_score: 0.5,
            locality_score: 1.0,
            health_score: 1.0,
            total: 0.0,
        },
        eligible: true,
        rejection_reason: None,
    };

    let low_locality = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("n2".to_string()),
            region_id: RegionId("ap-south".to_string()),
            endpoint: String::new(),
            estimated_latency: Duration::from_millis(200),
            score: 0.0,
        },
        scores: RouteScores {
            latency_score: 0.5,
            throughput_score: 0.3,
            cost_score: 0.5,
            locality_score: 0.0,
            health_score: 1.0,
            total: 0.0,
        },
        eligible: true,
        rejection_reason: None,
    };

    let high_score = policy.score(&high_locality, &request);
    let low_score = policy.score(&low_locality, &request);
    assert!(high_score > low_score);
}

#[test]
fn test_locality_policy_always_eligible() {
    let policy = LocalityPolicy::default();
    let request = mock_request();
    let candidate = mock_candidate(100, 1.0);
    assert!(policy.is_eligible(&candidate, &request));
}

#[test]
fn test_locality_policy_name() {
    let policy = LocalityPolicy::default();
    assert_eq!(policy.name(), "locality");
}

// =========================================================================
// PrivacyPolicy tests
// =========================================================================

#[test]
fn test_privacy_policy_name() {
    let policy = PrivacyPolicy::default();
    assert_eq!(policy.name(), "privacy");
}

#[test]
fn test_privacy_policy_score_always_one() {
    let policy = PrivacyPolicy::default();
    let request = mock_request();
    let candidate = mock_candidate(100, 1.0);
    let score = policy.score(&candidate, &request);
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_privacy_policy_unknown_region_defaults_internal() {
    let policy = PrivacyPolicy::default(); // No regions configured

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Internal;

    let candidate = mock_candidate(100, 1.0);
    // Unknown region defaults to Internal, which satisfies Internal requirement
    assert!(policy.is_eligible(&candidate, &request));
}

#[test]
fn test_privacy_policy_unknown_region_fails_confidential() {
    let policy = PrivacyPolicy::default();

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Confidential;

    let candidate = mock_candidate(100, 1.0);
    // Unknown region defaults to Internal, which does NOT satisfy Confidential
    assert!(!policy.is_eligible(&candidate, &request));
}

#[test]
fn test_privacy_policy_restricted_region() {
    let policy = PrivacyPolicy::default()
        .with_region(RegionId("us-west".to_string()), PrivacyLevel::Restricted);

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Restricted;

    let candidate = mock_candidate(100, 1.0);
    // us-west is Restricted, satisfies Restricted requirement
    assert!(policy.is_eligible(&candidate, &request));
}

#[test]
fn test_privacy_policy_public_request_any_region() {
    let policy =
        PrivacyPolicy::default().with_region(RegionId("us-west".to_string()), PrivacyLevel::Public);

    let mut request = mock_request();
    request.qos.privacy = PrivacyLevel::Public;

    let candidate = mock_candidate(100, 1.0);
    assert!(policy.is_eligible(&candidate, &request));
}

// =========================================================================
// CostPolicy tests
// =========================================================================

#[test]
fn test_cost_policy_default() {
    let policy = CostPolicy::default();
    assert_eq!(policy.weight, 1.0);
    assert!(policy.region_costs.is_empty());
}

#[test]
fn test_cost_policy_with_region_cost() {
    let policy = CostPolicy::default()
        .with_region_cost(RegionId("us-west".to_string()), 0.3)
        .with_region_cost(RegionId("eu-west".to_string()), 0.7);

    assert_eq!(policy.region_costs.len(), 2);
}

#[test]
fn test_cost_policy_clamps_cost() {
    let policy = CostPolicy::default()
        .with_region_cost(RegionId("r1".to_string()), 1.5)
        .with_region_cost(RegionId("r2".to_string()), -0.5);

    assert_eq!(
        *policy
            .region_costs
            .get(&RegionId("r1".to_string()))
            .expect("r1"),
        1.0
    );
    assert_eq!(
        *policy
            .region_costs
            .get(&RegionId("r2".to_string()))
            .expect("r2"),
        0.0
    );
}

#[test]
fn test_cost_policy_high_tolerance_prefers_throughput() {
    let policy = CostPolicy::default().with_region_cost(RegionId("us-west".to_string()), 0.9);

    let mut request = mock_request();
    request.qos.cost_tolerance = 80; // High tolerance -> prefer performance

    let candidate = mock_candidate(100, 1.0); // throughput_score = 0.8
    let score = policy.score(&candidate, &request);

    // With high cost tolerance, score should be based on throughput_score
    assert!((score - 0.8).abs() < f64::EPSILON);
}

#[test]
fn test_cost_policy_low_tolerance_prefers_cheap() {
    let policy = CostPolicy::default().with_region_cost(RegionId("us-west".to_string()), 0.8);

    let mut request = mock_request();
    request.qos.cost_tolerance = 20; // Low tolerance -> prefer cheap

    let candidate = mock_candidate(100, 1.0);
    let score = policy.score(&candidate, &request);

    // 1.0 - 0.8 = 0.2 (invert cost for cheap preference)
    assert!((score - 0.2).abs() < f64::EPSILON);
}

include!("policy_tests_cost_health.rs");
