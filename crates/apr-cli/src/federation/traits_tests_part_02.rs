
// =========================================================================
// CircuitState tests
// =========================================================================

#[test]
fn test_circuit_state_all_variants() {
    let states = [
        CircuitState::Closed,
        CircuitState::HalfOpen,
        CircuitState::Open,
    ];
    for (i, a) in states.iter().enumerate() {
        for (j, b) in states.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn test_circuit_state_copy() {
    let state = CircuitState::HalfOpen;
    let copied = state;
    assert_eq!(state, copied);
}

// =========================================================================
// LoadBalanceStrategy tests
// =========================================================================

#[test]
fn test_load_balance_default() {
    let strategy = LoadBalanceStrategy::default();
    assert!(matches!(strategy, LoadBalanceStrategy::LeastLatency));
}

#[test]
fn test_load_balance_all_variants() {
    let strategies = [
        LoadBalanceStrategy::RoundRobin,
        LoadBalanceStrategy::LeastConnections,
        LoadBalanceStrategy::LeastLatency,
        LoadBalanceStrategy::WeightedRandom,
        LoadBalanceStrategy::ConsistentHash,
    ];
    // Verify debug output
    for s in &strategies {
        let debug = format!("{:?}", s);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_load_balance_clone() {
    let strategy = LoadBalanceStrategy::WeightedRandom;
    let cloned = strategy;
    assert!(matches!(cloned, LoadBalanceStrategy::WeightedRandom));
}

// =========================================================================
// RouteTarget/RouteCandidate/RouteScores tests
// =========================================================================

#[test]
fn test_route_target_construction() {
    let target = RouteTarget {
        node_id: NodeId("n1".to_string()),
        region_id: RegionId("r1".to_string()),
        endpoint: "http://n1:8080".to_string(),
        estimated_latency: Duration::from_millis(50),
        score: 0.95,
    };
    assert_eq!(target.node_id, NodeId("n1".to_string()));
    assert_eq!(target.endpoint, "http://n1:8080");
    assert_eq!(target.estimated_latency, Duration::from_millis(50));
}

#[test]
fn test_route_target_clone() {
    let target = RouteTarget {
        node_id: NodeId("n1".to_string()),
        region_id: RegionId("r1".to_string()),
        endpoint: "http://n1:8080".to_string(),
        estimated_latency: Duration::from_millis(50),
        score: 0.5,
    };
    let cloned = target.clone();
    assert_eq!(cloned.node_id, NodeId("n1".to_string()));
    assert_eq!(cloned.score, 0.5);
}

#[test]
fn test_route_scores_construction() {
    let scores = RouteScores {
        latency_score: 0.9,
        throughput_score: 0.8,
        cost_score: 0.7,
        locality_score: 0.6,
        health_score: 1.0,
        total: 0.85,
    };
    assert_eq!(scores.latency_score, 0.9);
    assert_eq!(scores.total, 0.85);
}

#[test]
fn test_route_candidate_eligible() {
    let candidate = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("r1".to_string()),
            endpoint: String::new(),
            estimated_latency: Duration::from_millis(100),
            score: 0.8,
        },
        scores: RouteScores {
            latency_score: 0.9,
            throughput_score: 0.8,
            cost_score: 0.5,
            locality_score: 0.7,
            health_score: 1.0,
            total: 0.8,
        },
        eligible: true,
        rejection_reason: None,
    };
    assert!(candidate.eligible);
    assert!(candidate.rejection_reason.is_none());
}

#[test]
fn test_route_candidate_rejected() {
    let candidate = RouteCandidate {
        target: RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("r1".to_string()),
            endpoint: String::new(),
            estimated_latency: Duration::from_millis(100),
            score: 0.0,
        },
        scores: RouteScores {
            latency_score: 0.0,
            throughput_score: 0.0,
            cost_score: 0.0,
            locality_score: 0.0,
            health_score: 0.0,
            total: 0.0,
        },
        eligible: false,
        rejection_reason: Some("Policy rejected".to_string()),
    };
    assert!(!candidate.eligible);
    assert_eq!(
        candidate.rejection_reason,
        Some("Policy rejected".to_string())
    );
}

// =========================================================================
// ModelMetadata tests
// =========================================================================

#[test]
fn test_model_metadata_construction() {
    let meta = ModelMetadata {
        model_id: ModelId("llama-7b".to_string()),
        name: "LLaMA 7B".to_string(),
        version: "2.0".to_string(),
        capabilities: vec![Capability::Generate, Capability::Code],
        parameters: 7_000_000_000,
        quantization: Some("Q4_K".to_string()),
    };
    assert_eq!(meta.name, "LLaMA 7B");
    assert_eq!(meta.parameters, 7_000_000_000);
    assert_eq!(meta.quantization, Some("Q4_K".to_string()));
    assert_eq!(meta.capabilities.len(), 2);
}

#[test]
fn test_model_metadata_no_quantization() {
    let meta = ModelMetadata {
        model_id: ModelId("whisper".to_string()),
        name: "Whisper".to_string(),
        version: "1.0".to_string(),
        capabilities: vec![Capability::Transcribe],
        parameters: 1_500_000_000,
        quantization: None,
    };
    assert!(meta.quantization.is_none());
}

#[test]
fn test_model_metadata_clone() {
    let meta = ModelMetadata {
        model_id: ModelId("test".to_string()),
        name: "Test".to_string(),
        version: "1.0".to_string(),
        capabilities: vec![Capability::Embed],
        parameters: 100,
        quantization: None,
    };
    let cloned = meta.clone();
    assert_eq!(cloned.model_id, ModelId("test".to_string()));
}

// =========================================================================
// NodeHealth tests
// =========================================================================

#[test]
fn test_node_health_construction() {
    let health = NodeHealth {
        node_id: NodeId("test-node".to_string()),
        status: HealthState::Healthy,
        latency_p50: Duration::from_millis(25),
        latency_p99: Duration::from_millis(100),
        throughput: 500,
        gpu_utilization: Some(0.75),
        queue_depth: 3,
        last_check: std::time::Instant::now(),
    };
    assert_eq!(health.status, HealthState::Healthy);
    assert_eq!(health.throughput, 500);
    assert_eq!(health.gpu_utilization, Some(0.75));
    assert_eq!(health.queue_depth, 3);
}

#[test]
fn test_node_health_no_gpu() {
    let health = NodeHealth {
        node_id: NodeId("cpu-node".to_string()),
        status: HealthState::Healthy,
        latency_p50: Duration::from_millis(50),
        latency_p99: Duration::from_millis(200),
        throughput: 100,
        gpu_utilization: None,
        queue_depth: 0,
        last_check: std::time::Instant::now(),
    };
    assert!(health.gpu_utilization.is_none());
}

// =========================================================================
// GatewayStats tests
// =========================================================================

#[test]
fn test_gateway_stats_default() {
    let stats = GatewayStats::default();
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.successful_requests, 0);
    assert_eq!(stats.failed_requests, 0);
    assert_eq!(stats.total_tokens, 0);
    assert_eq!(stats.avg_latency, Duration::ZERO);
    assert_eq!(stats.active_streams, 0);
}

#[test]
fn test_gateway_stats_clone() {
    let stats = GatewayStats {
        total_requests: 100,
        successful_requests: 95,
        failed_requests: 5,
        total_tokens: 5000,
        avg_latency: Duration::from_millis(50),
        active_streams: 2,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_requests, 100);
    assert_eq!(cloned.active_streams, 2);
}

// =========================================================================
// FederationBuilder tests
// =========================================================================

#[test]
fn test_federation_builder_default() {
    let builder = FederationBuilder::default();
    assert!(builder.catalog.is_none());
    assert!(builder.health_checker.is_none());
    assert!(builder.router.is_none());
    assert!(builder.policies.is_empty());
    assert!(builder.middlewares.is_empty());
}

#[test]
fn test_federation_builder_new_defaults() {
    let builder = FederationBuilder::new();
    assert!(matches!(
        builder.load_balance,
        LoadBalanceStrategy::LeastLatency
    ));
}

#[test]
fn test_federation_builder_with_load_balance_all_strategies() {
    for strategy in [
        LoadBalanceStrategy::RoundRobin,
        LoadBalanceStrategy::LeastConnections,
        LoadBalanceStrategy::LeastLatency,
        LoadBalanceStrategy::WeightedRandom,
        LoadBalanceStrategy::ConsistentHash,
    ] {
        let builder = FederationBuilder::new().with_load_balance(strategy);
        let debug = format!("{:?}", builder.load_balance);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_federation_builder_with_policy() {
    use super::*;

    struct MockPolicy;
    impl RoutingPolicyTrait for MockPolicy {
        fn score(&self, _: &RouteCandidate, _: &InferenceRequest) -> f64 {
            1.0
        }
        fn is_eligible(&self, _: &RouteCandidate, _: &InferenceRequest) -> bool {
            true
        }
        fn name(&self) -> &'static str {
            "mock"
        }
    }

    let builder = FederationBuilder::new()
        .with_policy(MockPolicy)
        .with_policy(MockPolicy);
    assert_eq!(builder.policies.len(), 2);
}

#[test]
fn test_federation_builder_with_middleware() {
    use super::*;

    struct MockMiddleware;
    impl GatewayMiddleware for MockMiddleware {
        fn before_route(&self, _: &mut InferenceRequest) -> FederationResult<()> {
            Ok(())
        }
        fn after_infer(
            &self,
            _: &InferenceRequest,
            _: &mut InferenceResponse,
        ) -> FederationResult<()> {
            Ok(())
        }
        fn on_error(&self, _: &InferenceRequest, _: &FederationError) {}
    }

    let builder = FederationBuilder::new().with_middleware(MockMiddleware);
    assert_eq!(builder.middlewares.len(), 1);
}
