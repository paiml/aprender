
    // =========================================================================
    // Stats tracking tests
    // =========================================================================

    #[test]
    fn test_gateway_initial_stats() {
        let gateway = GatewayBuilder::new().build();
        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.active_streams, 0);
        assert_eq!(stats.avg_latency, Duration::ZERO);
    }

    #[tokio::test]
    async fn test_gateway_stats_after_failures() {
        let gateway = GatewayBuilder::new()
            .config(GatewayConfig {
                max_retries: 0, // No retries
                ..Default::default()
            })
            .build();

        // No nodes registered, so inference will fail
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"test".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "fail-test".to_string(),
            tenant_id: None,
        };

        let _ = gateway.infer(request).await;

        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.successful_requests, 0);
    }

    // =========================================================================
    // Stream cancel test
    // =========================================================================

    #[tokio::test]
    async fn test_stream_cancel() {
        let (gateway, catalog, health) = setup_test_gateway();

        catalog
            .register(
                ModelId("stream-model".to_string()),
                NodeId("n1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("n1".to_string()));
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(10));

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"stream".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "cancel-test".to_string(),
            tenant_id: None,
        };

        let mut stream = gateway.infer_stream(request).await.expect("stream failed");

        // Read a few tokens
        let _ = stream.next_token().await;
        let _ = stream.next_token().await;

        // Cancel the stream
        stream.cancel().await;

        // After cancel, next_token should return None
        let result = stream.next_token().await;
        assert!(result.is_none());
    }

    /// Comprehensive integration test demonstrating full federation flow
    #[tokio::test]
    async fn test_full_federation_flow() {
        use super::super::policy::CompositePolicy;

        // =====================================================================
        // Setup: Create multi-region deployment
        // =====================================================================
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        // Register Whisper model in US-West (primary, fast)
        catalog
            .register(
                ModelId("whisper-v3".to_string()),
                NodeId("us-west-gpu".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("failed to register us-west");

        // Register Whisper model in EU-West (GDPR compliant)
        catalog
            .register(
                ModelId("whisper-v3".to_string()),
                NodeId("eu-west-gpu".to_string()),
                RegionId("eu-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("failed to register eu-west");

        // Register LLaMA in US-East
        catalog
            .register(
                ModelId("llama-70b".to_string()),
                NodeId("us-east-gpu".to_string()),
                RegionId("us-east".to_string()),
                vec![Capability::Generate, Capability::Code],
            )
            .await
            .expect("failed to register llama");

        // Register embedding model in multiple regions
        for (node, region) in [("embed-us", "us-west"), ("embed-eu", "eu-west")] {
            catalog
                .register(
                    ModelId("bge-large".to_string()),
                    NodeId(node.to_string()),
                    RegionId(region.to_string()),
                    vec![Capability::Embed],
                )
                .await
                .expect("failed to register embedding");
        }

        // =====================================================================
        // Setup health states
        // =====================================================================

        // US-West: Healthy, fast (45ms)
        health.register_node(NodeId("us-west-gpu".to_string()));
        for _ in 0..3 {
            health.report_success(
                &NodeId("us-west-gpu".to_string()),
                Duration::from_millis(45),
            );
        }

        // EU-West: Healthy, slower (120ms)
        health.register_node(NodeId("eu-west-gpu".to_string()));
        for _ in 0..3 {
            health.report_success(
                &NodeId("eu-west-gpu".to_string()),
                Duration::from_millis(120),
            );
        }

        // US-East: Will be unknown/degraded (only 1 success)
        health.register_node(NodeId("us-east-gpu".to_string()));
        // Just one success keeps it in Unknown state (needs 2 for Healthy)
        health.report_success(
            &NodeId("us-east-gpu".to_string()),
            Duration::from_millis(100),
        );

        // Embedding nodes: Healthy
        for node in ["embed-us", "embed-eu"] {
            health.register_node(NodeId(node.to_string()));
            for _ in 0..3 {
                health.report_success(&NodeId(node.to_string()), Duration::from_millis(15));
            }
        }

        // =====================================================================
        // Create Router with enterprise policies
        // =====================================================================
        let router = Arc::new(
            Router::new(
                super::super::routing::RouterConfig {
                    max_candidates: 10,
                    min_score: 0.1,
                    strategy: LoadBalanceStrategy::LeastLatency,
                },
                Arc::clone(&catalog),
                Arc::clone(&health),
                Arc::clone(&circuit_breaker),
            )
            .with_policy(CompositePolicy::enterprise_default()),
        );

        // =====================================================================
        // Create Gateway
        // =====================================================================
        let gateway = FederationGateway::new(
            GatewayConfig {
                max_retries: 3,
                inference_timeout: Duration::from_secs(30),
                enable_tracing: true,
            },
            Arc::clone(&router),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        );

        // =====================================================================
        // Test 1: Transcribe routes to fastest healthy node (us-west)
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: b"audio data".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-transcribe".to_string(),
            tenant_id: Some("acme".to_string()),
        };

        let candidates = router
            .get_candidates(&request)
            .await
            .expect("get_candidates failed");
        assert_eq!(candidates.len(), 2, "Should have 2 Transcribe candidates");

        let target = router.route(&request).await.expect("route failed");
        // US-West should be selected (lower latency = higher score)
        assert_eq!(target.node_id, NodeId("us-west-gpu".to_string()));

        // =====================================================================
        // Test 2: Generate routes to only available node
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"prompt".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-generate".to_string(),
            tenant_id: None,
        };

        let target = router.route(&request).await.expect("route failed");
        assert_eq!(target.node_id, NodeId("us-east-gpu".to_string()));

        // =====================================================================
        // Test 3: Embed has multiple candidates
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Embed,
            input: b"text".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-embed".to_string(),
            tenant_id: None,
        };

        let candidates = router
            .get_candidates(&request)
            .await
            .expect("get_candidates failed");
        assert_eq!(candidates.len(), 2, "Should have 2 Embed candidates");

        // =====================================================================
        // Test 4: Gateway inference with stats
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: b"audio".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-infer".to_string(),
            tenant_id: None,
        };

        let response = gateway.infer(request).await.expect("inference failed");
        assert_eq!(response.served_by, NodeId("us-west-gpu".to_string()));
        assert!(!response.output.is_empty());

        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 0);

        // =====================================================================
        // Test 5: Streaming inference
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"stream prompt".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-stream".to_string(),
            tenant_id: None,
        };

        let mut stream = gateway.infer_stream(request).await.expect("stream failed");
        let mut tokens = 0;
        while let Some(result) = stream.next_token().await {
            result.expect("token error");
            tokens += 1;
        }
        assert_eq!(tokens, 10, "Should receive 10 tokens");

        // =====================================================================
        // Test 6: Circuit breaker
        // =====================================================================
        let bad_node = NodeId("failing-node".to_string());

        // Initially closed
        assert_eq!(circuit_breaker.state(&bad_node), CircuitState::Closed);

        // Simulate failures
        for _ in 0..5 {
            circuit_breaker.record_failure(&bad_node);
        }
        assert_eq!(circuit_breaker.state(&bad_node), CircuitState::Open);
        assert!(circuit_breaker.is_open(&bad_node));

        // =====================================================================
        // Verify catalog state
        // =====================================================================
        let all_models = catalog.list_all().await.expect("list failed");
        assert_eq!(all_models.len(), 3); // whisper-v3, llama-70b, bge-large

        // =====================================================================
        // Verify health states are tracked (all nodes have cached health)
        // =====================================================================
        let nodes_with_health = [
            "us-west-gpu",
            "eu-west-gpu",
            "us-east-gpu",
            "embed-us",
            "embed-eu",
        ];
        for node in nodes_with_health {
            let h = health.get_cached_health(&NodeId(node.to_string()));
            assert!(h.is_some(), "Health should be tracked for {}", node);
        }

        // Verify us-west has good health (3 successes)
        let us_west_health = health
            .get_cached_health(&NodeId("us-west-gpu".to_string()))
            .unwrap();
        assert_eq!(
            us_west_health.status,
            HealthState::Healthy,
            "US-West should be healthy"
        );

        // =====================================================================
        // Final summary - print results for visibility
        // =====================================================================
        println!("\n✅ Full Federation Flow Test PASSED!");
        println!("   - 3 models registered across 5 nodes");
        println!("   - 6 health entries tracked");
        println!("   - Routing correctly prefers fastest healthy nodes");
        println!("   - Gateway inference succeeds with stats tracking");
        println!("   - Streaming returns expected token count");
        println!("   - Circuit breaker opens after failures");
    }
