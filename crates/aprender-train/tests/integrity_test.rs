//! Integration tests for Behavioral Integrity & Lineage Module (ENT-013, ENT-014, ENT-015)

use entrenar::integrity::{
    behavioral::{
        BehavioralIntegrity, BehavioralIntegrityBuilder, IntegrityAssessment,
        MetamorphicRelationType, MetamorphicViolation,
    },
    lineage::{CausalLineage, LamportTimestamp, LineageEvent, LineageEventType},
    trace_storage::{CompressionAlgorithm, TraceStoragePolicy},
};

// ============================================================================
// ENT-014: Lamport Timestamp Integration Tests
// ============================================================================

#[test]
fn test_lamport_timestamp_distributed_scenario() {
    // Simulate three nodes in a distributed system
    let mut node_a = LamportTimestamp::new("node-a");
    let mut node_b = LamportTimestamp::new("node-b");
    let mut node_c = LamportTimestamp::new("node-c");

    // Node A sends message to Node B
    node_a.increment();
    let msg_a_to_b = node_a.clone();

    // Node B receives and sends to Node C
    node_b.merge(&msg_a_to_b);
    node_b.increment();
    let msg_b_to_c = node_b.clone();

    // Node C receives
    node_c.merge(&msg_b_to_c);

    // Verify causal ordering
    assert!(msg_a_to_b.happens_before(&msg_b_to_c));
    assert!(msg_b_to_c.happens_before(&node_c));
    assert!(msg_a_to_b.happens_before(&node_c));
}

#[test]
fn test_causal_lineage_full_experiment_lifecycle() {
    let mut lineage = CausalLineage::new();

    // Start multiple runs
    let ts1 = LamportTimestamp::new("node-1");
    let run1_start = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001");
    lineage.add_event(run1_start);

    let ts2 = LamportTimestamp::new("node-2");
    let run2_start = LineageEvent::new(ts2, LineageEventType::RunStarted, "run-002");
    lineage.add_event(run2_start);

    // Log metrics for run 1
    let ts3 = LamportTimestamp::with_counter("node-1", 2);
    let metric1 = LineageEvent::new(ts3, LineageEventType::MetricLogged, "run-001");
    lineage.add_event(metric1);

    // Save artifact for run 1
    let ts4 = LamportTimestamp::with_counter("node-1", 3);
    let artifact = LineageEvent::new(ts4, LineageEventType::ArtifactSaved, "run-001");
    lineage.add_event(artifact);

    // Complete run 1
    let ts5 = LamportTimestamp::with_counter("node-1", 4);
    let complete = LineageEvent::new(ts5, LineageEventType::RunCompleted, "run-001");
    lineage.add_event(complete);

    // Promote model from run 1
    let ts6 = LamportTimestamp::with_counter("node-1", 5);
    let promote = LineageEvent::new(ts6, LineageEventType::ModelPromoted, "run-001")
        .with_context("version: v1.0");
    lineage.add_event(promote);

    // Verify lineage
    assert_eq!(lineage.events_for_run("run-001").len(), 5);
    assert_eq!(lineage.events_for_run("run-002").len(), 1);
    assert_eq!(lineage.events_of_type(LineageEventType::ModelPromoted).len(), 1);

    // Verify ordering
    let events = lineage.events_in_order();
    assert_eq!(events.len(), 6);

    // Verify run 1 precedes its promotion
    let run1_events: Vec<_> = lineage.events_for_run("run-001");
    assert!(run1_events
        .first()
        .expect("operation should succeed")
        .timestamp
        .happens_before(&run1_events.last().expect("collection should not be empty").timestamp));
}

#[test]
fn test_lineage_event_context_propagation() {
    let mut lineage = CausalLineage::new();
    let ts = LamportTimestamp::new("node-1");

    let event = LineageEvent::new(ts, LineageEventType::ModelPromoted, "run-001")
        .with_context("model_version=v2.1.0,promoted_by=auto-gate,threshold=0.95");

    lineage.add_event(event);

    let retrieved = lineage.latest_event_for_run("run-001").expect("operation should succeed");
    assert!(retrieved.context.as_ref().expect("operation should succeed").contains("v2.1.0"));
    assert!(retrieved.context.as_ref().expect("operation should succeed").contains("auto-gate"));
}

// ============================================================================
// ENT-015: Trace Storage Policy Integration Tests
// ============================================================================

#[test]
fn test_trace_storage_policy_presets_comparison() {
    let minimal = TraceStoragePolicy::minimal();
    let dev = TraceStoragePolicy::development();
    let prod = TraceStoragePolicy::production();
    let archive = TraceStoragePolicy::archival();

    // Both minimal and dev have 7 days retention, prod and archive have more
    assert!(dev.retention_days <= prod.retention_days);
    assert!(prod.retention_days < archive.retention_days);

    // Dev should have highest sample rate (full sampling)
    assert!(dev.sample_rate > prod.sample_rate);
    assert!(prod.sample_rate > archive.sample_rate);
    // Minimal has lowest sample rate for minimal overhead
    assert!(minimal.sample_rate < archive.sample_rate);
}

#[test]
fn test_trace_storage_sampling_consistency() {
    let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 0.5);

    // Sampling should be deterministic
    let traces: Vec<String> = (0..100).map(|i| format!("trace-{i}")).collect();

    let sampled_first: Vec<_> = traces.iter().filter(|t| policy.should_sample(t)).collect();

    let sampled_second: Vec<_> = traces.iter().filter(|t| policy.should_sample(t)).collect();

    assert_eq!(sampled_first, sampled_second);
}

#[test]
fn test_trace_storage_compression_estimates() {
    let data_size = 1_000_000u64; // 1 MB

    let none = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, u64::MAX, 1.0);
    let zstd = TraceStoragePolicy::new(CompressionAlgorithm::Zstd, 7, u64::MAX, 1.0);
    let lz4 = TraceStoragePolicy::new(CompressionAlgorithm::Lz4, 7, u64::MAX, 1.0);

    let none_estimate = none.estimate_compressed_size(data_size);
    let zstd_estimate = zstd.estimate_compressed_size(data_size);
    let lz4_estimate = lz4.estimate_compressed_size(data_size);

    // No compression = same size
    assert_eq!(none_estimate, data_size);

    // Zstd should have better compression than LZ4
    assert!(zstd_estimate < lz4_estimate);

    // Both should be smaller than uncompressed
    assert!(lz4_estimate < data_size);
}

#[test]
fn test_trace_storage_limit_checking() {
    let policy = TraceStoragePolicy::new(
        CompressionAlgorithm::Zstd,
        30,
        100 * 1024 * 1024, // 100 MB
        1.0,
    );

    // Small additions should not exceed
    assert!(!policy.would_exceed_limit(50 * 1024 * 1024, 10 * 1024 * 1024));

    // Large additions should exceed
    assert!(policy.would_exceed_limit(80 * 1024 * 1024, 100 * 1024 * 1024));
}

// ============================================================================
// ENT-013: Behavioral Integrity Integration Tests
// ============================================================================

#[test]
fn test_behavioral_integrity_promotion_workflow() {
    // Scenario: Model passes all quality gates
    let integrity = BehavioralIntegrityBuilder::new("model-v2.0")
        .equivalence_score(0.98)
        .syscall_match(0.95)
        .timing_variance(0.05)
        .semantic_equiv(0.97)
        .test_count(10000)
        .build();

    assert!(integrity.passes_gate(0.9));
    assert_eq!(integrity.assessment(), IntegrityAssessment::Excellent);
    assert!(!integrity.has_critical_violations());

    let summary = integrity.summary();
    assert!(summary.contains("PASS"));
    assert!(summary.contains("model-v2.0"));
}

#[test]
fn test_behavioral_integrity_fails_with_critical_violation() {
    let violation = MetamorphicViolation::new(
        "MV-001",
        MetamorphicRelationType::Identity,
        "Model produces different outputs for identical inputs",
        "Input: standard test vector",
        "[0.5, 0.3, 0.2]",
        "[0.2, 0.5, 0.3]",
        0.95, // Critical severity
    );

    let integrity = BehavioralIntegrityBuilder::new("model-v2.1")
        .equivalence_score(0.98)
        .syscall_match(0.95)
        .timing_variance(0.05)
        .semantic_equiv(0.97)
        .violation(violation)
        .test_count(10000)
        .build();

    // High scores but critical violation = fail
    assert!(!integrity.passes_gate(0.9));
    assert_eq!(integrity.assessment(), IntegrityAssessment::Critical);

    let summary = integrity.summary();
    assert!(summary.contains("FAIL"));
}

#[test]
fn test_behavioral_integrity_fails_with_high_timing_variance() {
    let integrity = BehavioralIntegrityBuilder::new("model-unstable")
        .equivalence_score(0.98)
        .syscall_match(0.95)
        .timing_variance(0.35) // Too high!
        .semantic_equiv(0.97)
        .test_count(5000)
        .build();

    // Even with good scores, high timing variance fails
    assert!(!integrity.passes_gate(0.9));
}

#[test]
fn test_behavioral_integrity_violation_analysis() {
    let mut integrity = BehavioralIntegrity::new(0.85, 0.80, 0.1, 0.88, "model-v1.0");

    // Add various violations
    integrity.add_violation(MetamorphicViolation::new(
        "MV-001",
        MetamorphicRelationType::Additive,
        "Additive property violated",
        "f(x+1) != f(x)+1",
        "expected: 10.0",
        "actual: 10.5",
        0.6,
    ));

    integrity.add_violation(MetamorphicViolation::new(
        "MV-002",
        MetamorphicRelationType::Additive,
        "Another additive violation",
        "f(x+2) != f(x)+2",
        "expected: 12.0",
        "actual: 11.8",
        0.4,
    ));

    integrity.add_violation(MetamorphicViolation::new(
        "MV-003",
        MetamorphicRelationType::Permutation,
        "Permutation invariance violated",
        "f(permute(x)) != permute(f(x))",
        "expected order",
        "different order",
        0.85, // Critical
    ));

    // Analyze violations
    let by_type = integrity.violations_by_type();
    assert_eq!(by_type.get(&MetamorphicRelationType::Additive).expect("key should exist").len(), 2);
    assert_eq!(
        by_type.get(&MetamorphicRelationType::Permutation).expect("key should exist").len(),
        1
    );

    let counts = integrity.violation_counts();
    assert_eq!(counts.critical, 1);
    assert_eq!(counts.warnings, 1);
    assert_eq!(counts.minor, 1);
    assert_eq!(counts.total, 3);

    let most_severe = integrity.most_severe_violation().expect("operation should succeed");
    assert_eq!(most_severe.id, "MV-003");
}

#[test]
fn test_behavioral_integrity_composite_score_calculation() {
    // Test with known values
    let integrity = BehavioralIntegrity::new(
        0.8, // equivalence (weight 0.3)
        0.7, // syscall (weight 0.2)
        0.2, // timing variance (inverted, weight 0.2)
        0.9, // semantic (weight 0.3)
        "test-model",
    );

    // Expected: 0.3*0.8 + 0.2*0.7 + 0.2*(1-0.2) + 0.3*0.9
    //         = 0.24 + 0.14 + 0.16 + 0.27
    //         = 0.81
    let score = integrity.composite_score();
    assert!((score - 0.81).abs() < 0.01);
}

// ============================================================================
// Cross-Module Integration Tests
// ============================================================================

#[test]
fn test_integrity_with_lineage_tracking() {
    // Simulate a complete model lifecycle with integrity tracking
    let mut lineage = CausalLineage::new();

    // Start run
    let ts1 = LamportTimestamp::new("integrity-node");
    let start = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001");
    lineage.add_event(start);

    // Check integrity before promotion
    let integrity = BehavioralIntegrityBuilder::new("candidate-model")
        .equivalence_score(0.95)
        .syscall_match(0.92)
        .timing_variance(0.08)
        .semantic_equiv(0.94)
        .test_count(5000)
        .build();

    if integrity.passes_gate(0.9) {
        // Log promotion event
        let ts2 = LamportTimestamp::with_counter("integrity-node", 2);
        let context = format!(
            "integrity_score={:.2},assessment={}",
            integrity.composite_score(),
            integrity.assessment()
        );
        let promote = LineageEvent::new(ts2, LineageEventType::ModelPromoted, "run-001")
            .with_context(context);
        lineage.add_event(promote);
    }

    // Verify promotion was recorded
    let promotions = lineage.events_of_type(LineageEventType::ModelPromoted);
    assert_eq!(promotions.len(), 1);

    let promotion = promotions[0];
    assert!(promotion.context.is_some());
    assert!(promotion
        .context
        .as_ref()
        .expect("operation should succeed")
        .contains("integrity_score"));
    assert!(promotion.context.as_ref().expect("operation should succeed").contains("assessment"));
}

#[test]
fn test_storage_policy_with_lineage_events() {
    let policy = TraceStoragePolicy::new(CompressionAlgorithm::Zstd, 30, 10 * 1024 * 1024, 0.5);

    let mut lineage = CausalLineage::new();

    // Only add events that pass sampling
    for i in 0..100 {
        let trace_id = format!("event-{i}");
        if policy.should_sample(&trace_id) {
            let ts = LamportTimestamp::with_counter("sampler", i as u64);
            let event = LineageEvent::new(ts, LineageEventType::MetricLogged, &trace_id);
            lineage.add_event(event);
        }
    }

    // Sampling is deterministic based on hash - not exactly 50% but consistent
    // Verify that sampling occurred (some events filtered) and is deterministic
    let event_count = lineage.events_in_order().len();
    assert!(
        event_count > 0 && event_count < 100,
        "Expected some events filtered, got {event_count}"
    );

    // Verify determinism by checking again with same policy
    let mut lineage2 = CausalLineage::new();
    for i in 0..100 {
        let trace_id = format!("event-{i}");
        if policy.should_sample(&trace_id) {
            let ts = LamportTimestamp::with_counter("sampler", i as u64);
            let event = LineageEvent::new(ts, LineageEventType::MetricLogged, &trace_id);
            lineage2.add_event(event);
        }
    }
    assert_eq!(event_count, lineage2.events_in_order().len());
}

#[test]
fn test_serialization_roundtrip_all_types() {
    // Test that all integrity types can be serialized and deserialized

    // LamportTimestamp
    let ts = LamportTimestamp::with_counter("node-1", 42);
    let ts_json = serde_json::to_string(&ts).expect("JSON serialization should succeed");
    let ts_parsed: LamportTimestamp =
        serde_json::from_str(&ts_json).expect("JSON deserialization should succeed");
    assert_eq!(ts.counter, ts_parsed.counter);

    // TraceStoragePolicy
    let policy = TraceStoragePolicy::production();
    let policy_json = serde_json::to_string(&policy).expect("JSON serialization should succeed");
    let policy_parsed: TraceStoragePolicy =
        serde_json::from_str(&policy_json).expect("JSON deserialization should succeed");
    assert_eq!(policy.compression, policy_parsed.compression);

    // BehavioralIntegrity
    let integrity = BehavioralIntegrity::new(0.9, 0.85, 0.1, 0.88, "model-v1");
    let integrity_json =
        serde_json::to_string(&integrity).expect("JSON serialization should succeed");
    let integrity_parsed: BehavioralIntegrity =
        serde_json::from_str(&integrity_json).expect("JSON deserialization should succeed");
    assert!(
        (integrity.equivalence_score - integrity_parsed.equivalence_score).abs() < f64::EPSILON
    );
}
