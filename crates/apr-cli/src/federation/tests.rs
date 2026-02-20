
    fn setup_test_gateway() -> (FederationGateway, Arc<ModelCatalog>, Arc<HealthChecker>) {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Arc::new(Router::new(
            super::super::routing::RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let gateway = FederationGateway::new(
            GatewayConfig::default(),
            router,
            Arc::clone(&health),
            circuit_breaker,
        );

        (gateway, catalog, health)
    }

    #[tokio::test]
    async fn test_infer_no_nodes() {
        let (gateway, _, _) = setup_test_gateway();

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"hello".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-1".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_infer_with_node() {
        let (gateway, catalog, health) = setup_test_gateway();

        // Register a node
        catalog
            .register(
                ModelId("test-model".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"hello".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-2".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer(request).await;
        assert!(result.is_ok());

        let response = result.expect("inference failed");
        assert_eq!(response.served_by, NodeId("node-1".to_string()));
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let (gateway, catalog, health) = setup_test_gateway();

        catalog
            .register(
                ModelId("test".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Embed],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        // Make some requests
        for i in 0..3 {
            let request = InferenceRequest {
                capability: Capability::Embed,
                input: vec![i],
                qos: QoSRequirements::default(),
                request_id: format!("test-{}", i),
                tenant_id: None,
            };

            let _ = gateway.infer(request).await;
        }

        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_streaming() {
        let (gateway, catalog, health) = setup_test_gateway();

        catalog
            .register(
                ModelId("stream-model".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"stream test".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "stream-1".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer_stream(request).await;
        assert!(result.is_ok());

        let mut stream = result.expect("stream creation failed");

        // Read tokens
        let mut token_count = 0;
        while let Some(result) = stream.next_token().await {
            assert!(result.is_ok());
            token_count += 1;
        }

        assert_eq!(token_count, 10);
    }

    #[test]
    fn test_gateway_builder() {
        let gateway = GatewayBuilder::new()
            .config(GatewayConfig {
                max_retries: 5,
                inference_timeout: Duration::from_secs(60),
                enable_tracing: false,
            })
            .middleware(LoggingMiddleware::new("test"))
            .build();

        assert_eq!(gateway.config.max_retries, 5);
        assert_eq!(gateway.middlewares.len(), 1);
    }

    // =========================================================================
    // GatewayConfig tests
    // =========================================================================

    #[test]
    fn test_gateway_config_default() {
        let config = GatewayConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.inference_timeout, Duration::from_secs(30));
        assert!(config.enable_tracing);
    }

    #[test]
    fn test_gateway_config_clone() {
        let config = GatewayConfig {
            max_retries: 5,
            inference_timeout: Duration::from_secs(60),
            enable_tracing: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_retries, 5);
        assert!(!cloned.enable_tracing);
    }

    // =========================================================================
    // GatewayBuilder extended tests
    // =========================================================================

    #[test]
    fn test_gateway_builder_default() {
        let builder = GatewayBuilder::default();
        assert!(builder.catalog.is_none());
        assert!(builder.health.is_none());
        assert!(builder.circuit_breaker.is_none());
        assert!(builder.router.is_none());
        assert!(builder.middlewares.is_empty());
    }

    #[test]
    fn test_gateway_builder_with_catalog() {
        let catalog = Arc::new(ModelCatalog::new());
        let builder = GatewayBuilder::new().catalog(catalog);
        assert!(builder.catalog.is_some());
    }

    #[test]
    fn test_gateway_builder_with_health() {
        let health = Arc::new(HealthChecker::default());
        let builder = GatewayBuilder::new().health(health);
        assert!(builder.health.is_some());
    }

    #[test]
    fn test_gateway_builder_with_circuit_breaker() {
        let cb = Arc::new(CircuitBreaker::default());
        let builder = GatewayBuilder::new().circuit_breaker(cb);
        assert!(builder.circuit_breaker.is_some());
    }

    #[test]
    fn test_gateway_builder_with_router() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let cb = Arc::new(CircuitBreaker::default());
        let router = Arc::new(Router::new(
            super::super::routing::RouterConfig::default(),
            catalog,
            health,
            cb,
        ));
        let builder = GatewayBuilder::new().router(router);
        assert!(builder.router.is_some());
    }

    #[test]
    fn test_gateway_builder_with_middleware() {
        let builder = GatewayBuilder::new()
            .middleware(LoggingMiddleware::new("test"))
            .middleware(RateLimitMiddleware::new(100));
        assert_eq!(builder.middlewares.len(), 2);
    }

    #[test]
    fn test_gateway_builder_full_chain() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let cb = Arc::new(CircuitBreaker::default());

        let gateway = GatewayBuilder::new()
            .config(GatewayConfig {
                max_retries: 5,
                inference_timeout: Duration::from_secs(120),
                enable_tracing: false,
            })
            .catalog(Arc::clone(&catalog))
            .health(Arc::clone(&health))
            .circuit_breaker(Arc::clone(&cb))
            .middleware(LoggingMiddleware::new("gw"))
            .build();

        assert_eq!(gateway.config.max_retries, 5);
        assert_eq!(gateway.middlewares.len(), 1);
    }

    // =========================================================================
    // LoggingMiddleware tests
    // =========================================================================

    #[test]
    fn test_logging_middleware_creation() {
        let middleware = LoggingMiddleware::new("test-prefix");
        assert_eq!(middleware.prefix, "test-prefix");
    }

    #[test]
    fn test_logging_middleware_before_route() {
        let middleware = LoggingMiddleware::new("test");
        let mut request = InferenceRequest {
            capability: Capability::Generate,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        let result = middleware.before_route(&mut request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_logging_middleware_after_infer() {
        let middleware = LoggingMiddleware::new("test");
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        let mut response = InferenceResponse {
            output: b"output".to_vec(),
            served_by: NodeId("n1".to_string()),
            latency: Duration::from_millis(50),
            tokens: Some(5),
        };
        let result = middleware.after_infer(&request, &mut response);
        assert!(result.is_ok());
    }

    #[test]
    fn test_logging_middleware_on_error() {
        let middleware = LoggingMiddleware::new("test");
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        let error = FederationError::Internal("test error".to_string());
        // Should not panic
        middleware.on_error(&request, &error);
    }

    // =========================================================================
    // RateLimitMiddleware tests
    // =========================================================================

    #[test]
    fn test_rate_limit_middleware_creation() {
        let _middleware = RateLimitMiddleware::new(1000);
    }

    #[test]
    fn test_rate_limit_middleware_before_route() {
        let middleware = RateLimitMiddleware::new(100);
        let mut request = InferenceRequest {
            capability: Capability::Embed,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        assert!(middleware.before_route(&mut request).is_ok());
    }

    #[test]
    fn test_rate_limit_middleware_after_infer() {
        let middleware = RateLimitMiddleware::new(100);
        let request = InferenceRequest {
            capability: Capability::Embed,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        let mut response = InferenceResponse {
            output: vec![],
            served_by: NodeId("n1".to_string()),
            latency: Duration::from_millis(10),
            tokens: None,
        };
        assert!(middleware.after_infer(&request, &mut response).is_ok());
    }

    #[test]
    fn test_rate_limit_middleware_on_error() {
        let middleware = RateLimitMiddleware::new(100);
        let request = InferenceRequest {
            capability: Capability::Embed,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-1".to_string(),
            tenant_id: None,
        };
        let error = FederationError::Internal("err".to_string());
        middleware.on_error(&request, &error); // Should not panic
    }

    // =========================================================================
    // Gateway with middleware integration test
    // =========================================================================

    #[tokio::test]
    async fn test_gateway_with_logging_middleware() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Arc::new(Router::new(
            super::super::routing::RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let gateway = FederationGateway::new(
            GatewayConfig::default(),
            router,
            Arc::clone(&health),
            circuit_breaker,
        )
        .with_middleware(LoggingMiddleware::new("test-gw"));

        assert_eq!(gateway.middlewares.len(), 1);

        // Register a node so inference works
        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("n1".to_string()));
        for _ in 0..3 {
            health.report_success(&NodeId("n1".to_string()), Duration::from_millis(10));
        }

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"test".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "mw-test".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer(request).await;
        assert!(result.is_ok());
    }
