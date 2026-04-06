//! Experiment Span Tests (REN-001)
//!
//! Extreme TDD tests for entrenar experiment tracking integration.
//! Cross-project: ENT-EPIC-001 (entrenar experiment tracking spec v1.8.0 §5)

use std::collections::HashMap;

// =============================================================================
// Test 1: SpanType::Experiment variant exists
// =============================================================================

#[test]
fn test_span_type_experiment_variant_exists() {
    use renacer::experiment_span::SpanType;

    // SpanType::Experiment must exist
    let span_type = SpanType::Experiment;
    assert_eq!(span_type, SpanType::Experiment);
}

#[test]
fn test_span_type_has_all_variants() {
    use renacer::experiment_span::SpanType;

    // Verify all expected variants exist
    let _syscall = SpanType::Syscall;
    let _gpu = SpanType::Gpu;
    let _experiment = SpanType::Experiment;
}

// =============================================================================
// Test 2: ExperimentMetadata struct with required fields
// =============================================================================

#[test]
fn test_experiment_metadata_struct_fields() {
    use renacer::experiment_span::ExperimentMetadata;

    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.95);
    metrics.insert("f1_score".to_string(), 0.92);

    let metadata = ExperimentMetadata {
        model_name: "gpt-2".to_string(),
        epoch: Some(10),
        step: Some(1000),
        loss: Some(0.0025),
        metrics,
    };

    assert_eq!(metadata.model_name, "gpt-2");
    assert_eq!(metadata.epoch, Some(10));
    assert_eq!(metadata.step, Some(1000));
    assert_eq!(metadata.loss, Some(0.0025));
    assert_eq!(metadata.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(metadata.metrics.get("f1_score"), Some(&0.92));
}

#[test]
fn test_experiment_metadata_optional_fields() {
    use renacer::experiment_span::ExperimentMetadata;

    // All optional fields can be None
    let metadata = ExperimentMetadata {
        model_name: "bert-base".to_string(),
        epoch: None,
        step: None,
        loss: None,
        metrics: HashMap::new(),
    };

    assert_eq!(metadata.model_name, "bert-base");
    assert!(metadata.epoch.is_none());
    assert!(metadata.step.is_none());
    assert!(metadata.loss.is_none());
    assert!(metadata.metrics.is_empty());
}

#[test]
fn test_experiment_metadata_default() {
    use renacer::experiment_span::ExperimentMetadata;

    let metadata = ExperimentMetadata::default();

    assert!(metadata.model_name.is_empty());
    assert!(metadata.epoch.is_none());
    assert!(metadata.step.is_none());
    assert!(metadata.loss.is_none());
    assert!(metadata.metrics.is_empty());
}

// =============================================================================
// Test 3: ExperimentSpan::new_experiment constructor
// =============================================================================

#[test]
fn test_new_experiment_constructor() {
    use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

    let mut metrics = HashMap::new();
    metrics.insert("perplexity".to_string(), 15.5);

    let metadata = ExperimentMetadata {
        model_name: "llama-7b".to_string(),
        epoch: Some(5),
        step: Some(500),
        loss: Some(0.15),
        metrics,
    };

    let span = ExperimentSpan::new_experiment("training_step", metadata);

    assert_eq!(span.name, "training_step");
    assert_eq!(span.metadata.model_name, "llama-7b");
    assert_eq!(span.metadata.epoch, Some(5));
    assert_eq!(span.metadata.step, Some(500));
    assert_eq!(span.metadata.loss, Some(0.15));
}

#[test]
fn test_experiment_span_has_trace_id() {
    use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

    let metadata = ExperimentMetadata::default();
    let span = ExperimentSpan::new_experiment("test", metadata);

    // Span should have a valid trace_id
    assert_ne!(span.trace_id, [0u8; 16]);
}

#[test]
fn test_experiment_span_has_span_id() {
    use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

    let metadata = ExperimentMetadata::default();
    let span = ExperimentSpan::new_experiment("test", metadata);

    // Span should have a valid span_id
    assert_ne!(span.span_id, [0u8; 8]);
}

#[test]
fn test_experiment_span_timestamps() {
    use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

    let metadata = ExperimentMetadata::default();
    let span = ExperimentSpan::new_experiment("test", metadata);

    // Start time should be set
    assert!(span.start_time_nanos > 0);
}

// =============================================================================
// Test 4: Golden Trace Comparison API - compare_traces and EquivalenceScore
// =============================================================================

#[test]
fn test_equivalence_score_struct() {
    use renacer::experiment_span::EquivalenceScore;

    let score =
        EquivalenceScore { syscall_match: 0.95, timing_variance: 0.02, semantic_equiv: 0.98 };

    assert_eq!(score.syscall_match, 0.95);
    assert_eq!(score.timing_variance, 0.02);
    assert_eq!(score.semantic_equiv, 0.98);
}

#[test]
fn test_equivalence_score_overall() {
    use renacer::experiment_span::EquivalenceScore;

    let score =
        EquivalenceScore { syscall_match: 0.90, timing_variance: 0.10, semantic_equiv: 0.95 };

    // Overall score should be a weighted combination
    let overall = score.overall();
    assert!(overall >= 0.0 && overall <= 1.0);
}

#[test]
fn test_equivalence_score_is_equivalent() {
    use renacer::experiment_span::EquivalenceScore;

    let good_score =
        EquivalenceScore { syscall_match: 0.95, timing_variance: 0.05, semantic_equiv: 0.98 };

    let bad_score =
        EquivalenceScore { syscall_match: 0.50, timing_variance: 0.80, semantic_equiv: 0.40 };

    // Good score should be considered equivalent (threshold ~0.9)
    assert!(good_score.is_equivalent());
    // Bad score should not be equivalent
    assert!(!bad_score.is_equivalent());
}

#[test]
fn test_compare_traces_identical() {
    use renacer::experiment_span::compare_traces;
    use renacer::unified_trace::UnifiedTrace;

    // Create identical traces
    let trace1 = UnifiedTrace::new(1234, "test_process".to_string());
    let trace2 = UnifiedTrace::new(1234, "test_process".to_string());

    let score = compare_traces(&trace1, &trace2);

    // Identical traces should have perfect equivalence
    assert!(score.syscall_match >= 0.99);
    assert!(score.semantic_equiv >= 0.99);
}

#[test]
fn test_compare_traces_different() {
    use renacer::experiment_span::compare_traces;
    use renacer::unified_trace::{SyscallSpan, UnifiedTrace};
    use std::borrow::Cow;

    // Create different traces
    let mut trace1 = UnifiedTrace::new(1234, "test_process".to_string());
    let mut trace2 = UnifiedTrace::new(1234, "test_process".to_string());

    // Add different syscalls
    trace1.add_syscall(SyscallSpan {
        span_id: 1,
        parent_span_id: 0,
        name: Cow::Borrowed("read"),
        args: vec![],
        return_value: 100,
        timestamp_nanos: 1000,
        duration_nanos: 50,
        errno: None,
    });

    trace2.add_syscall(SyscallSpan {
        span_id: 2,
        parent_span_id: 0,
        name: Cow::Borrowed("write"),
        args: vec![],
        return_value: 100,
        timestamp_nanos: 1000,
        duration_nanos: 50,
        errno: None,
    });

    let score = compare_traces(&trace1, &trace2);

    // Different syscalls should result in lower match score
    assert!(score.syscall_match < 0.99);
}

#[test]
fn test_compare_traces_timing_variance() {
    use renacer::experiment_span::compare_traces;
    use renacer::unified_trace::{SyscallSpan, UnifiedTrace};
    use std::borrow::Cow;

    // Create traces with same syscalls but different timing
    let mut trace1 = UnifiedTrace::new(1234, "test_process".to_string());
    let mut trace2 = UnifiedTrace::new(1234, "test_process".to_string());

    // Same syscall, different duration
    trace1.add_syscall(SyscallSpan {
        span_id: 1,
        parent_span_id: 0,
        name: Cow::Borrowed("read"),
        args: vec![],
        return_value: 100,
        timestamp_nanos: 1000,
        duration_nanos: 50,
        errno: None,
    });

    trace2.add_syscall(SyscallSpan {
        span_id: 1,
        parent_span_id: 0,
        name: Cow::Borrowed("read"),
        args: vec![],
        return_value: 100,
        timestamp_nanos: 1000,
        duration_nanos: 500, // 10x slower
        errno: None,
    });

    let score = compare_traces(&trace1, &trace2);

    // Same syscalls should have high match, but timing variance should reflect difference
    assert!(score.syscall_match >= 0.9);
    assert!(score.timing_variance > 0.0);
}

// =============================================================================
// Test 5: ExperimentMetadata serialization (for JSON attributes)
// =============================================================================

#[test]
fn test_experiment_metadata_to_json() {
    use renacer::experiment_span::ExperimentMetadata;

    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.95);

    let metadata = ExperimentMetadata {
        model_name: "test-model".to_string(),
        epoch: Some(10),
        step: Some(1000),
        loss: Some(0.05),
        metrics,
    };

    let json = metadata.to_json();
    assert!(json.contains("test-model"));
    assert!(json.contains("epoch"));
}

#[test]
fn test_experiment_metadata_from_json() {
    use renacer::experiment_span::ExperimentMetadata;

    let json = r#"{"model_name":"test","epoch":5,"step":100,"loss":0.1,"metrics":{}}"#;
    let metadata = ExperimentMetadata::from_json(json).expect("test");

    assert_eq!(metadata.model_name, "test");
    assert_eq!(metadata.epoch, Some(5));
    assert_eq!(metadata.step, Some(100));
}

// =============================================================================
// Test 6: Integration with SpanRecord
// =============================================================================

#[test]
fn test_experiment_span_to_span_record() {
    use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};
    use renacer::span_record::SpanKind;

    let mut metrics = HashMap::new();
    metrics.insert("loss".to_string(), 0.05);

    let metadata = ExperimentMetadata {
        model_name: "gpt-neo".to_string(),
        epoch: Some(1),
        step: Some(100),
        loss: Some(0.05),
        metrics,
    };

    let experiment_span = ExperimentSpan::new_experiment("training", metadata);
    let span_record = experiment_span.to_span_record();

    assert_eq!(span_record.span_name, "training");
    assert_eq!(span_record.span_kind, SpanKind::Internal);

    // Attributes should contain experiment metadata
    let attrs = span_record.parse_attributes();
    assert_eq!(attrs.get("experiment.model_name"), Some(&"gpt-neo".to_string()));
    assert_eq!(attrs.get("experiment.epoch"), Some(&"1".to_string()));
}
