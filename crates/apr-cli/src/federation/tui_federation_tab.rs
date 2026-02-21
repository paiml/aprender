
    #[test]
    fn test_federation_tab_titles() {
        let titles = FederationTab::titles();
        assert_eq!(titles.len(), 7);
        assert!(titles[0].contains("Catalog"));
        assert!(titles[1].contains("Health"));
        assert!(titles[2].contains("Routing"));
        assert!(titles[3].contains("Circuits"));
        assert!(titles[4].contains("Stats"));
        assert!(titles[5].contains("Policies"));
        assert!(titles[6].contains("Help"));
    }

    #[test]
    fn test_federation_tab_index() {
        assert_eq!(FederationTab::Catalog.index(), 0);
        assert_eq!(FederationTab::Health.index(), 1);
        assert_eq!(FederationTab::Routing.index(), 2);
        assert_eq!(FederationTab::Circuits.index(), 3);
        assert_eq!(FederationTab::Stats.index(), 4);
        assert_eq!(FederationTab::Policies.index(), 5);
        assert_eq!(FederationTab::Help.index(), 6);
    }

    #[test]
    fn test_federation_tab_from_index() {
        assert_eq!(FederationTab::from_index(0), FederationTab::Catalog);
        assert_eq!(FederationTab::from_index(1), FederationTab::Health);
        assert_eq!(FederationTab::from_index(2), FederationTab::Routing);
        assert_eq!(FederationTab::from_index(3), FederationTab::Circuits);
        assert_eq!(FederationTab::from_index(4), FederationTab::Stats);
        assert_eq!(FederationTab::from_index(5), FederationTab::Policies);
        assert_eq!(FederationTab::from_index(6), FederationTab::Help);
        assert_eq!(FederationTab::from_index(99), FederationTab::Help);
    }

    #[test]
    fn test_federation_app_new() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.current_tab, FederationTab::Catalog);
        assert!(!app.should_quit);
        assert!(app.routing_history.is_empty());
        assert_eq!(app.policies.len(), 5);
    }

    #[test]
    fn test_federation_app_tab_navigation() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.current_tab, FederationTab::Catalog);
        app.next_tab();
        assert_eq!(app.current_tab, FederationTab::Health);
        app.next_tab();
        assert_eq!(app.current_tab, FederationTab::Routing);
        app.prev_tab();
        assert_eq!(app.current_tab, FederationTab::Health);
    }

    #[test]
    fn test_federation_app_routing_history() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        let record = RoutingRecord {
            request_id: "req-1".to_string(),
            capability: "Transcribe".to_string(),
            selected_node: "node-1".to_string(),
            score: 0.95,
            reason: "lowest latency".to_string(),
            timestamp: std::time::Instant::now(),
        };

        app.record_routing(record);
        assert_eq!(app.routing_history.len(), 1);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("very long string", 10), "very lo...");
        assert_eq!(truncate("exactly10", 10), "exactly10");
    }

    // =========================================================================
    // FederationTab extended tests
    // =========================================================================

    #[test]
    fn test_federation_tab_default() {
        let tab = FederationTab::default();
        assert_eq!(tab, FederationTab::Catalog);
    }

    #[test]
    fn test_federation_tab_roundtrip_all() {
        for i in 0..7 {
            let tab = FederationTab::from_index(i);
            assert_eq!(tab.index(), i);
        }
    }

    #[test]
    fn test_federation_tab_from_index_overflow() {
        // Any index >= 6 should map to Help
        assert_eq!(FederationTab::from_index(7), FederationTab::Help);
        assert_eq!(FederationTab::from_index(100), FederationTab::Help);
        assert_eq!(FederationTab::from_index(usize::MAX), FederationTab::Help);
    }

    #[test]
    fn test_federation_tab_copy() {
        let tab = FederationTab::Routing;
        let copied = tab;
        assert_eq!(tab, copied);
    }

    #[test]
    fn test_federation_tab_debug() {
        let tab = FederationTab::Stats;
        let debug = format!("{:?}", tab);
        assert_eq!(debug, "Stats");
    }

    // =========================================================================
    // Truncate edge cases
    // =========================================================================

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn test_truncate_exactly_at_limit() {
        assert_eq!(truncate("1234567890", 10), "1234567890");
    }

    #[test]
    fn test_truncate_one_over() {
        assert_eq!(truncate("12345678901", 10), "1234567...");
    }

    #[test]
    fn test_truncate_max_len_3() {
        // max_len=3, 3-3=0 -> "..."
        assert_eq!(truncate("abcdef", 3), "...");
    }

    #[test]
    fn test_truncate_max_len_4() {
        assert_eq!(truncate("abcdef", 4), "a...");
    }

    // =========================================================================
    // RoutingRecord construction
    // =========================================================================

    #[test]
    fn test_routing_record_construction() {
        let record = RoutingRecord {
            request_id: "req-42".to_string(),
            capability: "Generate".to_string(),
            selected_node: "gpu-node-1".to_string(),
            score: 0.95,
            reason: "lowest latency".to_string(),
            timestamp: std::time::Instant::now(),
        };
        assert_eq!(record.request_id, "req-42");
        assert_eq!(record.capability, "Generate");
        assert_eq!(record.score, 0.95);
    }

    #[test]
    fn test_routing_record_clone() {
        let record = RoutingRecord {
            request_id: "req-1".to_string(),
            capability: "Embed".to_string(),
            selected_node: "node-1".to_string(),
            score: 0.5,
            reason: "only available".to_string(),
            timestamp: std::time::Instant::now(),
        };
        let cloned = record.clone();
        assert_eq!(cloned.request_id, "req-1");
    }

    // =========================================================================
    // CircuitDisplay construction
    // =========================================================================

    #[test]
    fn test_circuit_display_construction() {
        let display = CircuitDisplay {
            node_id: "node-1".to_string(),
            state: CircuitState::Open,
            failure_count: 5,
            last_failure: Some(std::time::Instant::now()),
            reset_remaining: Some(Duration::from_secs(25)),
        };
        assert_eq!(display.node_id, "node-1");
        assert_eq!(display.state, CircuitState::Open);
        assert_eq!(display.failure_count, 5);
        assert!(display.last_failure.is_some());
        assert!(display.reset_remaining.is_some());
    }

    #[test]
    fn test_circuit_display_closed() {
        let display = CircuitDisplay {
            node_id: "healthy-node".to_string(),
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: None,
            reset_remaining: None,
        };
        assert_eq!(display.state, CircuitState::Closed);
        assert!(display.last_failure.is_none());
    }

    // =========================================================================
    // PolicyDisplay construction
    // =========================================================================

    #[test]
    fn test_policy_display_enabled() {
        let display = PolicyDisplay {
            name: "latency".to_string(),
            weight: 1.0,
            enabled: true,
            description: "Prefer low-latency nodes".to_string(),
        };
        assert!(display.enabled);
        assert_eq!(display.weight, 1.0);
    }

    #[test]
    fn test_policy_display_disabled() {
        let display = PolicyDisplay {
            name: "cost".to_string(),
            weight: 0.5,
            enabled: false,
            description: "Disabled for testing".to_string(),
        };
        assert!(!display.enabled);
    }

    // =========================================================================
    // FederationApp extended tests
    // =========================================================================

    #[test]
    fn test_federation_app_with_gateway() {
        use super::super::gateway::FederationGateway;
        use super::super::routing::{Router, RouterConfig};

        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Arc::new(Router::new(
            RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let gateway = Arc::new(FederationGateway::new(
            super::super::gateway::GatewayConfig::default(),
            router,
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let app = FederationApp::new(
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        )
        .with_gateway(gateway);

        assert!(app.gateway.is_some());
    }

    #[test]
    fn test_federation_app_success_rate_no_gateway() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        // Without gateway, default is 1.0
        assert!((app.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_federation_app_requests_per_sec_no_gateway() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert!((app.requests_per_sec() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_federation_app_healthy_node_count() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.healthy_node_count(), 0);
        assert_eq!(app.total_node_count(), 0);
    }

    #[test]
    fn test_federation_app_select_next() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.selected_row, 0);
        app.select_next();
        assert_eq!(app.selected_row, 1);
        app.select_next();
        assert_eq!(app.selected_row, 2);
    }

    #[test]
    fn test_federation_app_select_prev_at_zero() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert_eq!(app.selected_row, 0);
        app.select_prev(); // saturating_sub(1) at 0 stays 0
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn test_federation_app_select_prev_from_nonzero() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.selected_row = 5;
        app.select_prev();
        assert_eq!(app.selected_row, 4);
    }

    #[test]
    fn test_federation_app_tab_navigation_resets_row() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.selected_row = 5;
        app.next_tab();
        assert_eq!(app.selected_row, 0);

        app.selected_row = 3;
        app.prev_tab();
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn test_federation_app_next_tab_wraps() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        // Navigate to last tab (Help, index 6)
        app.current_tab = FederationTab::Help;
        app.next_tab();
        // Should wrap to Catalog (index 0)
        assert_eq!(app.current_tab, FederationTab::Catalog);
    }

    #[test]
    fn test_federation_app_prev_tab_wraps() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        // At Catalog (index 0), prev should wrap to Help (index 6)
        app.current_tab = FederationTab::Catalog;
        app.prev_tab();
        assert_eq!(app.current_tab, FederationTab::Help);
    }

    #[test]
    fn test_federation_app_routing_history_overflow() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        app.max_history = 3;

        // Add 5 records
        for i in 0..5 {
            app.record_routing(RoutingRecord {
                request_id: format!("req-{}", i),
                capability: "Generate".to_string(),
                selected_node: "node-1".to_string(),
                score: 0.9,
                reason: "test".to_string(),
                timestamp: std::time::Instant::now(),
            });
        }

        // Should be capped at max_history
        assert_eq!(app.routing_history.len(), 3);
        // First entries should be removed (FIFO)
        assert_eq!(app.routing_history[0].request_id, "req-2");
        assert_eq!(app.routing_history[1].request_id, "req-3");
        assert_eq!(app.routing_history[2].request_id, "req-4");
    }

    #[test]
    fn test_federation_app_status_message() {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());
        let mut app = FederationApp::new(catalog, health, circuit_breaker);

        assert!(app.status_message.is_none());
        app.status_message = Some("Connected to 3 nodes".to_string());
        assert_eq!(app.status_message, Some("Connected to 3 nodes".to_string()));
    }
