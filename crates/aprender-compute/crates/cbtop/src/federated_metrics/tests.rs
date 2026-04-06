use super::*;

#[test]
fn test_g_counter() {
    let mut counter = GCounter::new();

    counter.increment("host1", 5);
    counter.increment("host2", 3);

    assert_eq!(counter.value(), 8);
    assert_eq!(counter.host_count("host1"), 5);
    assert_eq!(counter.host_count("host2"), 3);
}

#[test]
fn test_g_counter_merge() {
    let mut c1 = GCounter::new();
    c1.increment("host1", 5);
    c1.increment("host2", 3);

    let mut c2 = GCounter::new();
    c2.increment("host1", 3); // Less than c1
    c2.increment("host2", 7); // More than c1
    c2.increment("host3", 2); // New host

    c1.merge(&c2);

    assert_eq!(c1.host_count("host1"), 5); // Max(5, 3) = 5
    assert_eq!(c1.host_count("host2"), 7); // Max(3, 7) = 7
    assert_eq!(c1.host_count("host3"), 2); // New host
    assert_eq!(c1.value(), 14);
}

#[test]
fn test_lww_register() {
    let mut reg = LwwRegister::new("initial", 100, "host1");

    assert_eq!(reg.value(), &"initial");

    // Update with newer timestamp
    reg.update("newer", 200, "host2");
    assert_eq!(reg.value(), &"newer");

    // Update with older timestamp (ignored)
    reg.update("older", 150, "host3");
    assert_eq!(reg.value(), &"newer");
}

#[test]
fn test_or_set() {
    let mut set = OrSet::new();

    set.add("elem1".to_string(), "tag1".to_string());
    set.add("elem2".to_string(), "tag2".to_string());

    assert!(set.contains(&"elem1".to_string()));
    assert!(set.contains(&"elem2".to_string()));
    assert!(!set.contains(&"elem3".to_string()));

    set.remove(&"elem1".to_string());
    assert!(!set.contains(&"elem1".to_string()));
    assert!(set.contains(&"elem2".to_string()));
}

#[test]
fn test_or_set_merge() {
    let mut s1 = OrSet::new();
    s1.add("a".to_string(), "tag-a".to_string());

    let mut s2 = OrSet::new();
    s2.add("b".to_string(), "tag-b".to_string());

    s1.merge(&s2);

    assert!(s1.contains(&"a".to_string()));
    assert!(s1.contains(&"b".to_string()));
}

#[test]
fn test_aggregated_metrics() {
    let mut agg = AggregatedMetrics::new("latency");

    agg.add_sample("host1", 100.0);
    agg.add_sample("host1", 200.0);
    agg.add_sample("host2", 150.0);

    assert_eq!(agg.values.len(), 3);
    assert_eq!(agg.mean(), 150.0);
    assert_eq!(agg.min, 100.0);
    assert_eq!(agg.max, 200.0);
}

#[test]
fn test_aggregated_percentiles() {
    let mut agg = AggregatedMetrics::new("latency");

    for i in 1..=100 {
        agg.add_sample("host1", i as f64);
    }

    // p50 on values 1-100 should be around 50-51 (index-based calculation)
    assert!((agg.p50() - 50.5).abs() < 2.0);
    assert!((agg.p95() - 95.0).abs() < 2.0);
    assert!((agg.p99() - 99.0).abs() < 2.0);
}

#[test]
fn test_federation_record() {
    let config = FederationConfig::default();
    let mut fed = MetricsFederation::new("local", config);

    let id = fed.record("cpu_usage", 75.0).unwrap();

    assert_eq!(id.host_id, "local");
    assert_eq!(fed.total_samples(), 1);
}

#[test]
fn test_federation_add_host() {
    let config = FederationConfig::default();
    let mut fed = MetricsFederation::new("local", config);

    fed.add_host("remote1");
    fed.add_host("remote2");

    assert_eq!(fed.active_host_count(), 3); // local + 2 remote
}

#[test]
fn test_federation_merge() {
    let config = FederationConfig::default();
    let mut fed1 = MetricsFederation::new("host1", config.clone());
    let mut fed2 = MetricsFederation::new("host2", config);

    fed1.record("metric", 100.0).unwrap();
    fed1.record("metric", 110.0).unwrap();

    fed2.record("metric", 200.0).unwrap();
    fed2.record("metric", 210.0).unwrap();

    let merged = fed1.merge(&fed2).unwrap();

    assert_eq!(merged, 2); // 2 samples from fed2
    assert_eq!(fed1.total_samples(), 4);
}

#[test]
fn test_federation_idempotent_merge() {
    let config = FederationConfig::default();
    let mut fed1 = MetricsFederation::new("host1", config.clone());
    let mut fed2 = MetricsFederation::new("host2", config);

    fed1.record("metric", 100.0).unwrap();
    fed2.record("metric", 200.0).unwrap();

    // First merge
    let merged1 = fed1.merge(&fed2).unwrap();
    assert_eq!(merged1, 1);

    // Second merge (should be idempotent)
    let merged2 = fed1.merge(&fed2).unwrap();
    assert_eq!(merged2, 0); // No new samples

    assert_eq!(fed1.total_samples(), 2);
}

#[test]
fn test_federation_skew_detection() {
    let config = FederationConfig { skew_threshold_percent: 40.0, ..Default::default() };
    let mut fed = MetricsFederation::new("local", config);

    fed.add_host("fast_host");
    fed.add_host("slow_host");

    // Fast host sends many samples
    for _ in 0..100 {
        let sample = MetricSample::new("fast_host", fed.tick(), fed.sequence, "latency", 10.0);
        fed.sequence += 1;
        fed.add_sample(sample).unwrap();
    }

    // Slow host sends few samples
    for _ in 0..20 {
        let sample = MetricSample::new("slow_host", fed.tick(), fed.sequence, "latency", 10.0);
        fed.sequence += 1;
        fed.add_sample(sample).unwrap();
    }

    let skewed = fed.detect_skewed_hosts();
    assert!(!skewed.is_empty());
}

#[test]
fn test_federation_memory_limit() {
    let config = FederationConfig {
        memory_limit_bytes: 1000, // Very small limit
        ..Default::default()
    };
    let mut fed = MetricsFederation::new("local", config);

    // Try to add many samples
    let mut hit_limit = false;
    for i in 0..1000 {
        if fed.record(format!("metric_{}", i), i as f64).is_err() {
            hit_limit = true;
            break;
        }
    }

    assert!(hit_limit, "Should hit memory limit");
}

#[test]
fn test_federation_health_update() {
    let config = FederationConfig::default();
    let mut fed = MetricsFederation::new("host1", config);

    fed.add_host("host2");

    // host1 sends 80 samples, host2 sends 20
    for _ in 0..80 {
        fed.record("metric", 1.0).unwrap();
    }

    for _ in 0..20 {
        let sample = MetricSample::new("host2", fed.tick(), fed.sequence, "metric", 1.0);
        fed.sequence += 1;
        fed.add_sample(sample).unwrap();
    }

    fed.update_health();

    // host2 should have lower health
    let host2 = fed.get_host("host2").unwrap();
    assert!(host2.health < 1.0);
}

#[test]
fn test_sampling_rate_adaptation() {
    let config = FederationConfig::default();
    let mut fed = MetricsFederation::new("local", config);

    fed.add_host("remote");

    // Set high latency for remote
    if let Some(host) = fed.hosts.get_mut("remote") {
        host.latency_ms = 200.0;
    }

    fed.adapt_sampling_rates(50.0);

    let local = fed.get_host("local").unwrap();
    let remote = fed.get_host("remote").unwrap();

    // Remote should have lower sampling rate due to latency
    assert!(remote.sampling_rate < local.sampling_rate);
}

#[test]
fn test_error_display() {
    let err = FederatedError::ClockDriftExceeded { drift_ms: 150, max_ms: 100 };
    assert!(err.to_string().contains("150"));
    assert!(err.to_string().contains("100"));
}

// FKR-049: CRDT convergence across partitions
#[test]
fn test_fkr_049_crdt_convergence() {
    let config = FederationConfig::default();

    // Simulate 3-node cluster
    let mut node1 = MetricsFederation::new("node1", config.clone());
    let mut node2 = MetricsFederation::new("node2", config.clone());
    let mut node3 = MetricsFederation::new("node3", config);

    // Register all nodes with each other
    node1.add_host("node2");
    node1.add_host("node3");
    node2.add_host("node1");
    node2.add_host("node3");
    node3.add_host("node1");
    node3.add_host("node2");

    // Phase 1: Each node records samples independently (simulating partition)
    for i in 0..10 {
        node1.record("metric", 100.0 + i as f64).unwrap();
        node2.record("metric", 200.0 + i as f64).unwrap();
        node3.record("metric", 300.0 + i as f64).unwrap();
    }

    // Verify isolation
    assert_eq!(node1.total_samples(), 10);
    assert_eq!(node2.total_samples(), 10);
    assert_eq!(node3.total_samples(), 10);

    // Phase 2: Partition heals - merge all nodes
    node1.merge(&node2).unwrap();
    node1.merge(&node3).unwrap();

    // Phase 3: Verify convergence
    assert_eq!(node1.total_samples(), 30); // All samples merged

    // Verify percentiles are correct
    let agg = node1.get_aggregated("metric").unwrap();
    assert_eq!(agg.values.len(), 30);

    // p50 should be around 200 (middle of 100-309 range)
    let p50 = agg.p50();
    assert!(p50 > 150.0 && p50 < 250.0, "p50 {} should be ~200", p50);

    // Verify no duplicates (idempotent merge)
    node1.merge(&node2).unwrap();
    assert_eq!(node1.total_samples(), 30); // Still 30, no duplicates
}
