
    /// Test that serve_model properly delegates to prepare_serve_state
    #[tokio::test]
    async fn test_serve_model_delegates_to_prepare_serve_state() {
        // Test that serve_model returns the same error as prepare_serve_state
        // for invalid extensions
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.xyz", false, false, true).await;
        assert!(result.is_err());

        // The error should match what prepare_serve_state returns
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    /// Test address parsing in serve_model context
    #[test]
    fn test_serve_model_address_formats() {
        // Test various IPv4 address format combinations
        let ipv4_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 3000), ("192.168.1.1", 80)];

        for (host, port) in ipv4_cases {
            let addr_str = format!("{}:{}", host, port);
            let result: std::result::Result<std::net::SocketAddr, _> = addr_str.parse();
            assert!(result.is_ok(), "IPv4 address parsing failed for {addr_str}");
        }

        // Test IPv6 address (needs brackets for SocketAddr parsing)
        let ipv6_addr: std::result::Result<std::net::SocketAddr, _> = "[::1]:8080".parse();
        assert!(ipv6_addr.is_ok(), "IPv6 address parsing failed");
    }

    /// Test that PreparedServer can be created with demo state
    #[test]
    fn test_prepared_server_with_demo_state() {
        let state = crate::api::AppState::demo().expect("Failed to create demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Gguf,
        };

        // Verify the struct fields
        assert!(!prepared.batch_mode_enabled);
        assert_eq!(prepared.model_type, ModelType::Gguf);

        // Verify debug output is useful
        let debug = format!("{:?}", prepared);
        assert!(debug.contains("batch_mode_enabled: false"));
    }
