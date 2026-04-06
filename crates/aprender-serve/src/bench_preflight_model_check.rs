
// ============================================================================
// Tests (EXTREME TDD: Tests written FIRST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_check_rejects_empty_model_name() {
        let check = ModelAvailabilityCheck::new(String::new());
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_model_check_requires_available_models() {
        let check = ModelAvailabilityCheck::new("phi2".to_string());
        // No available models set
        let result = check.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_model_check_name() {
        let check = ModelAvailabilityCheck::new("phi2".to_string());
        assert_eq!(check.name(), "model_availability_check");
        assert_eq!(check.requested_model(), "phi2");
    }

    // =========================================================================
    // ResponseSchemaCheck Tests
    // =========================================================================

    #[test]
    fn test_schema_check_llama_cpp_completion() {
        let check = ResponseSchemaCheck::llama_cpp_completion();
        assert_eq!(check.name(), "response_schema_check");
        assert!(check.validate().is_ok()); // Has required fields
    }

    #[test]
    fn test_schema_check_ollama_generate() {
        let check = ResponseSchemaCheck::ollama_generate();
        assert!(check.validate().is_ok());
    }

    #[test]
    fn test_schema_check_validates_required_fields() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string(), "tokens".to_string()]);
        let json: serde_json::Value = serde_json::json!({
            "content": "Hello",
            "tokens": 5
        });
        assert!(check.validate_json(&json).is_ok());
    }

    #[test]
    fn test_schema_check_fails_on_missing_field() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string(), "tokens".to_string()]);
        let json: serde_json::Value = serde_json::json!({
            "content": "Hello"
            // missing "tokens"
        });
        let result = check.validate_json(&json);
        assert!(
            matches!(result, Err(PreflightError::SchemaMismatch { missing_field }) if missing_field == "tokens")
        );
    }

    #[test]
    fn test_schema_check_validates_field_types() {
        let check = ResponseSchemaCheck::new(vec!["count".to_string()])
            .with_type_constraint("count".to_string(), "number".to_string());

        // Correct type
        let json: serde_json::Value = serde_json::json!({ "count": 42 });
        assert!(check.validate_json(&json).is_ok());

        // Wrong type
        let json: serde_json::Value = serde_json::json!({ "count": "42" });
        let result = check.validate_json(&json);
        assert!(matches!(
            result,
            Err(PreflightError::FieldTypeMismatch { .. })
        ));
    }

    #[test]
    fn test_schema_check_rejects_non_object() {
        let check = ResponseSchemaCheck::new(vec!["content".to_string()]);
        let json: serde_json::Value = serde_json::json!("not an object");
        let result = check.validate_json(&json);
        assert!(matches!(
            result,
            Err(PreflightError::ResponseParseError { .. })
        ));
    }

    #[test]
    fn test_schema_check_rejects_empty_required_fields() {
        let check = ResponseSchemaCheck::new(vec![]);
        let result = check.validate();
        assert!(result.is_err());
    }

    // =========================================================================
    // PreflightRunner Tests
    // =========================================================================

    #[test]
    fn test_runner_runs_all_checks() {
        let mut runner = PreflightRunner::new();

        // Add a passing check
        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // Add another passing check
        let schema = ResponseSchemaCheck::new(vec!["foo".to_string()]);
        runner.add_check(Box::new(schema));

        let result = runner.run();
        assert!(result.is_ok());
        assert_eq!(result.expect("test").len(), 2);
    }

    #[test]
    fn test_runner_stops_on_first_failure_jidoka() {
        let mut runner = PreflightRunner::new();

        // Add a passing check
        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // Add a failing check (empty required fields)
        let schema = ResponseSchemaCheck::new(vec![]);
        runner.add_check(Box::new(schema));

        // Add another check that won't run
        let config2 = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config2)));

        let result = runner.run();
        assert!(result.is_err());
        // Only first check passed before failure
        assert_eq!(runner.passed_checks().len(), 1);
    }

    #[test]
    fn test_runner_empty_passes() {
        let mut runner = PreflightRunner::new();
        let result = runner.run();
        assert!(result.is_ok());
        assert!(result.expect("test").is_empty());
    }

    #[test]
    fn test_runner_clears_passed_on_rerun() {
        let mut runner = PreflightRunner::new();

        let config = DeterministicInferenceConfig::default();
        runner.add_check(Box::new(DeterminismCheck::new(config)));

        // First run
        let _ = runner.run();
        assert_eq!(runner.passed_checks().len(), 1);

        // Second run should clear and repopulate
        let _ = runner.run();
        assert_eq!(runner.passed_checks().len(), 1);
    }

    // =========================================================================
    // IMP-143: Real-World Server Verification Tests (EXTREME TDD)
    // =========================================================================
    // These tests verify actual connectivity to external servers.
    // Run with: cargo test test_imp_143 --lib -- --ignored

    /// IMP-143a: Verify llama.cpp server preflight check works with real server
    #[test]
    #[ignore = "Requires running llama.cpp server on port 8082"]
    fn test_imp_143a_llamacpp_real_server_check() {
        // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082
        let mut check = ServerAvailabilityCheck::llama_cpp(8082);

        // Attempt real connection
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("IMP-143a: Should create HTTP client");

        let health_url = check.health_url();
        match client.get(&health_url).send() {
            Ok(response) => {
                let status = response.status().as_u16();
                check.set_health_status(status);

                // IMP-143a: If server is running, check should pass
                let result = check.validate();
                assert!(
                    result.is_ok(),
                    "IMP-143a: llama.cpp server check should pass when server is running. Status: {}, Error: {:?}",
                    status,
                    result.err()
                );
            },
            Err(e) => {
                panic!(
                    "IMP-143a: Could not connect to llama.cpp server at {}. \
                    Start with: llama-server -m model.gguf --host 127.0.0.1 --port 8082. \
                    Error: {}",
                    health_url, e
                );
            },
        }
    }

    /// IMP-143b: Verify Ollama server preflight check works with real server
    #[test]
    #[ignore = "Requires running Ollama server on port 11434"]
    fn test_imp_143b_ollama_real_server_check() {
        // This test requires: ollama serve
        let mut check = ServerAvailabilityCheck::ollama(11434);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("IMP-143b: Should create HTTP client");

        let health_url = check.health_url();
        match client.get(&health_url).send() {
            Ok(response) => {
                let status = response.status().as_u16();
                check.set_health_status(status);

                // IMP-143b: If server is running, check should pass
                let result = check.validate();
                assert!(
                    result.is_ok(),
                    "IMP-143b: Ollama server check should pass when server is running. Status: {}, Error: {:?}",
                    status,
                    result.err()
                );
            },
            Err(e) => {
                panic!(
                    "IMP-143b: Could not connect to Ollama server at {}. \
                    Start with: ollama serve. \
                    Error: {}",
                    health_url, e
                );
            },
        }
    }

    /// IMP-143c: Preflight runner should detect server availability
    #[test]
    fn test_imp_143c_preflight_detects_unavailable_server() {
        // Use a port that's unlikely to have a server running
        let mut check = ServerAvailabilityCheck::llama_cpp(59999);
        check.set_health_status(0); // Connection refused test

        // IMP-143c: Check should fail for unavailable server
        let result = check.validate();
        assert!(
            result.is_err(),
            "IMP-143c: Preflight should detect unavailable server"
        );
    }

    /// IMP-143d: Preflight reports correct error for connection failures
    #[test]
    fn test_imp_143d_preflight_error_reporting() {
        let mut check = ServerAvailabilityCheck::llama_cpp(59998);
        check.set_health_status(503); // Service unavailable

        let result = check.validate();
        match result {
            Err(PreflightError::HealthCheckFailed { status, url, .. }) => {
                assert_eq!(status, 503, "IMP-143d: Should report correct status code");
                assert!(url.contains("59998"), "IMP-143d: Should report correct URL");
            },
            _ => panic!("IMP-143d: Should return HealthCheckFailed error"),
        }
    }
include!("bench_preflight_canonical_inputs.rs");
}
