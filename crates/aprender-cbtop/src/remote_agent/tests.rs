//! Tests for remote agent module.

use super::*;

#[test]
fn test_host_config_builder() {
    let config = HostConfig::new("server1.example.com", "admin")
        .with_port(2222)
        .with_connect_timeout_ms(5000)
        .with_architecture("x86_64")
        .with_label("env", "production");

    assert_eq!(config.host, "server1.example.com");
    assert_eq!(config.username, "admin");
    assert_eq!(config.port, 2222);
    assert_eq!(config.connect_timeout_ms, 5000);
    assert_eq!(config.architecture, Some("x86_64".to_string()));
    assert_eq!(config.labels.get("env"), Some(&"production".to_string()));
}

#[test]
fn test_host_state_availability() {
    let config = HostConfig::new("host1", "user");
    let mut state = HostState::new(config);

    // Initially unknown, should be available
    assert!(state.is_available());

    // After success, should be healthy
    state.record_success(100);
    assert!(state.is_available());
    assert!(matches!(state.health, HostHealth::Healthy));

    // After degraded response, still available
    state.record_success(6000);
    assert!(state.is_available());
    assert!(matches!(state.health, HostHealth::Degraded { .. }));

    // After 3 failures, unreachable
    state.record_failure("error 1");
    state.record_failure("error 2");
    state.record_failure("error 3");
    assert!(!state.is_available());
    assert!(matches!(state.health, HostHealth::Unreachable { .. }));
}

#[test]
fn test_host_state_latency_tracking() {
    let config = HostConfig::new("host1", "user");
    let mut state = HostState::new(config);

    state.record_success(100);
    assert_eq!(state.avg_latency_ms, 100.0);

    state.record_success(200);
    assert_eq!(state.avg_latency_ms, 150.0);

    state.record_success(300);
    assert!((state.avg_latency_ms - 200.0).abs() < 0.01);
}

#[test]
fn test_remote_agent_add_hosts() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user1"));
    agent.add_host(HostConfig::new("host2", "user2"));

    assert_eq!(agent.host_count(), 2);
    assert_eq!(agent.available_count(), 2);
}

#[test]
fn test_remote_agent_remove_host() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user1"));
    assert_eq!(agent.host_count(), 1);

    let removed = agent.remove_host("host1");
    assert!(removed.is_some());
    assert_eq!(agent.host_count(), 0);
}

#[test]
fn test_remote_agent_execute_health_check() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user1"));

    let result = agent.execute_on_host("host1:22", "echo health");
    assert!(result.is_ok());
    let cmd_result = result.unwrap();
    assert!(cmd_result.success());
    assert_eq!(cmd_result.stdout, "OK");
}

#[test]
fn test_remote_agent_health_check_all() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user1"));
    agent.add_host(HostConfig::new("host2", "user2"));

    let health = agent.health_check();

    assert_eq!(health.len(), 2);
    assert!(matches!(health.get("host1:22"), Some(HostHealth::Healthy)));
    assert!(matches!(health.get("host2:22"), Some(HostHealth::Healthy)));
}

#[test]
fn test_remote_agent_collect_benchmarks() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user1").with_architecture("x86_64"));
    agent.add_host(HostConfig::new("host2", "user2").with_architecture("aarch64"));

    let result = agent.collect_benchmarks();
    assert!(result.is_ok());

    let aggregated = result.unwrap();
    assert_eq!(aggregated.hosts_succeeded, 2);
    assert_eq!(aggregated.hosts_failed, 0);
    assert!(aggregated.throughput_geomean > 0.0);
}

#[test]
fn test_aggregation_strategy_geometric_mean() {
    let config = RemoteAgentConfig {
        aggregation: AggregationStrategy::GeometricMean,
        ..Default::default()
    };
    let agent = RemoteAgent::new(config);

    let benchmarks = vec![
        HostBenchmark {
            host: "h1".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 100.0,
            latency_p50_us: 50.0,
            latency_p99_us: 100.0,
            memory_bytes: 1024,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
        HostBenchmark {
            host: "h2".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 400.0,
            latency_p50_us: 100.0,
            latency_p99_us: 200.0,
            memory_bytes: 2048,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
    ];

    let (throughput, latency_p50, latency_p99, succeeded, failed) =
        agent.aggregate_results(&benchmarks, 0);

    // Geometric mean of 100 and 400 is 200
    assert!((throughput - 200.0).abs() < 0.01);
    // Arithmetic mean of 50 and 100 is 75
    assert!((latency_p50 - 75.0).abs() < 0.01);
    // Max of 100 and 200 is 200
    assert_eq!(latency_p99, 200.0);
    assert_eq!(succeeded, 2);
    assert_eq!(failed, 0);
}

#[test]
fn test_aggregation_strategy_median() {
    let config = RemoteAgentConfig {
        aggregation: AggregationStrategy::Median,
        ..Default::default()
    };
    let agent = RemoteAgent::new(config);

    let benchmarks = vec![
        HostBenchmark {
            host: "h1".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 100.0,
            latency_p50_us: 50.0,
            latency_p99_us: 100.0,
            memory_bytes: 1024,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
        HostBenchmark {
            host: "h2".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 200.0,
            latency_p50_us: 75.0,
            latency_p99_us: 150.0,
            memory_bytes: 2048,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
        HostBenchmark {
            host: "h3".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 300.0,
            latency_p50_us: 100.0,
            latency_p99_us: 200.0,
            memory_bytes: 3072,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
    ];

    let (throughput, latency_p50, latency_p99, _, _) = agent.aggregate_results(&benchmarks, 0);

    // Median of [100, 200, 300] is 200
    assert_eq!(throughput, 200.0);
    assert_eq!(latency_p50, 75.0);
    assert_eq!(latency_p99, 150.0);
}

#[test]
fn test_aggregation_strategy_minimum() {
    let config = RemoteAgentConfig {
        aggregation: AggregationStrategy::Minimum,
        ..Default::default()
    };
    let agent = RemoteAgent::new(config);

    let benchmarks = vec![
        HostBenchmark {
            host: "h1".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 100.0,
            latency_p50_us: 50.0,
            latency_p99_us: 100.0,
            memory_bytes: 1024,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
        HostBenchmark {
            host: "h2".to_string(),
            architecture: "x86_64".to_string(),
            throughput_ops: 200.0,
            latency_p50_us: 75.0,
            latency_p99_us: 150.0,
            memory_bytes: 2048,
            gpu_utilization: None,
            timestamp_ns: 0,
        },
    ];

    let (throughput, latency_p50, latency_p99, _, _) = agent.aggregate_results(&benchmarks, 0);

    assert_eq!(throughput, 100.0);
    assert_eq!(latency_p50, 50.0);
    assert_eq!(latency_p99, 100.0);
}

#[test]
fn test_hosts_with_label() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user").with_label("env", "prod"));
    agent.add_host(HostConfig::new("host2", "user").with_label("env", "staging"));
    agent.add_host(HostConfig::new("host3", "user").with_label("env", "prod"));

    let prod_hosts = agent.hosts_with_label("env", "prod");
    assert_eq!(prod_hosts.len(), 2);
}

#[test]
fn test_hosts_with_arch() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    agent.add_host(HostConfig::new("host1", "user").with_architecture("x86_64"));
    agent.add_host(HostConfig::new("host2", "user").with_architecture("aarch64"));
    agent.add_host(HostConfig::new("host3", "user").with_architecture("x86_64"));

    let x86_hosts = agent.hosts_with_arch("x86_64");
    assert_eq!(x86_hosts.len(), 2);

    let arm_hosts = agent.hosts_with_arch("aarch64");
    assert_eq!(arm_hosts.len(), 1);
}

#[test]
fn test_command_result_success() {
    let result = CommandResult {
        host: "host1".to_string(),
        exit_code: 0,
        stdout: "output".to_string(),
        stderr: String::new(),
        duration_ms: 100,
    };
    assert!(result.success());

    let failed = CommandResult {
        host: "host1".to_string(),
        exit_code: 1,
        stdout: String::new(),
        stderr: "error".to_string(),
        duration_ms: 50,
    };
    assert!(!failed.success());
}

#[test]
fn test_aggregated_result_success_rate() {
    let result = AggregatedResult {
        host_results: vec![],
        throughput_geomean: 100.0,
        latency_p50_mean_us: 50.0,
        latency_p99_max_us: 200.0,
        hosts_succeeded: 3,
        hosts_failed: 1,
        collection_time_ms: 1000,
    };

    assert!((result.success_rate() - 0.75).abs() < 0.01);
}

#[test]
fn test_remote_error_display() {
    let err = RemoteError::ConnectionFailed {
        host: "host1".to_string(),
        reason: "timeout".to_string(),
    };
    assert!(err.to_string().contains("host1"));
    assert!(err.to_string().contains("timeout"));

    let err = RemoteError::AllHostsFailed {
        failures: vec!["error1".to_string(), "error2".to_string()],
    };
    assert!(err.to_string().contains("error1"));
}

#[test]
fn test_json_parsing() {
    use super::json;

    let json_str = r#"{"host":"server1","arch":"x86_64","throughput":1000000,"latency_p50":50,"latency_p99":200,"memory":1073741824}"#;

    // Test string extraction
    assert_eq!(
        json::extract_json_string(json_str, "host"),
        Some("server1".to_string())
    );
    assert_eq!(
        json::extract_json_string(json_str, "arch"),
        Some("x86_64".to_string())
    );

    // Test number extraction
    assert_eq!(
        json::extract_json_number(json_str, "throughput"),
        Some(1000000.0)
    );
    assert_eq!(
        json::extract_json_number(json_str, "latency_p50"),
        Some(50.0)
    );
    assert_eq!(
        json::extract_json_number(json_str, "memory"),
        Some(1073741824.0)
    );
}

#[test]
fn test_history_limit() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);
    agent.max_history = 3;

    agent.add_host(HostConfig::new("host1", "user"));

    // Execute more commands than history limit
    for _ in 0..5 {
        let _ = agent.execute_on_host("host1:22", "echo health");
    }

    assert_eq!(agent.history().len(), 3);
}

// FKR-045: Multi-host heterogeneous collection
#[test]
fn test_fkr_045_heterogeneous_collection() {
    let config = RemoteAgentConfig::default();
    let mut agent = RemoteAgent::new(config);

    // Add hosts with different architectures
    agent.add_host(HostConfig::new("x86-host", "user").with_architecture("x86_64"));
    agent.add_host(HostConfig::new("arm-host", "user").with_architecture("aarch64"));
    agent.add_host(HostConfig::new("riscv-host", "user").with_architecture("riscv64"));

    let result = agent.collect_benchmarks();
    assert!(result.is_ok());

    let aggregated = result.unwrap();

    // Verify all 3 hosts contributed results
    assert_eq!(aggregated.hosts_succeeded, 3);
    assert_eq!(aggregated.host_results.len(), 3);

    // Verify different architectures are represented
    let archs: Vec<&str> = aggregated
        .host_results
        .iter()
        .map(|r| r.architecture.as_str())
        .collect();
    assert!(archs.contains(&"x86_64"));
    assert!(archs.contains(&"aarch64"));
    assert!(archs.contains(&"riscv64"));
}
