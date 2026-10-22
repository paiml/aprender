// #5717: validate_observability_contracts, one test per branch.

fn ev5717(id: u64, event_type: AwsEventType, previous_event_id: Option<u64>) -> TraceEvent {
    TraceEvent {
        id,
        timestamp: String::new(),
        event_type,
        previous_event_id,
        step: TraceStep::Tokenize,
        iteration: 0,
        layer: None,
        input_shape: Vec::new(),
        output_shape: Vec::new(),
        stats: TensorStats::default(),
        duration_us: 0,
        error: None,
        cause: None,
        details: TraceDetails::default(),
    }
}

fn tracer5717(events: Vec<TraceEvent>) -> InferenceTracer {
    let mut tracer = InferenceTracer::new(TraceConfig::enabled());
    tracer.events = events;
    tracer
}

#[test]
fn observability_empty_trace_has_no_violations_5717() {
    assert!(tracer5717(Vec::new())
        .validate_observability_contracts()
        .is_empty());
}

#[test]
fn observability_paired_increasing_trace_is_clean_5717() {
    let t = tracer5717(vec![
        ev5717(1, AwsEventType::TaskStateEntered, None),
        ev5717(2, AwsEventType::TaskStateExited, Some(1)),
    ]);
    assert!(t.validate_observability_contracts().is_empty());
}

#[test]
fn observability_non_increasing_id_is_one_monotonic_error_5717() {
    use crate::brick::profiler::ContractSeverity;
    let t = tracer5717(vec![
        ev5717(5, AwsEventType::TaskStateEntered, None),
        ev5717(5, AwsEventType::TaskStateExited, Some(5)),
        ev5717(3, AwsEventType::TaskStateEntered, None),
        ev5717(4, AwsEventType::TaskStateExited, Some(3)),
    ]);
    let v = t.validate_observability_contracts();
    let mono: Vec<_> = v
        .iter()
        .filter(|(_, m)| m.contains("MONOTONIC_COUNTER"))
        .collect();
    assert_eq!(mono.len(), 1, "one violation is enough: {v:?}");
    assert_eq!(mono[0].0, ContractSeverity::Error);
    assert!(mono[0].1.contains("event ID 5 <= previous ID 5"), "{v:?}");
}

#[test]
fn observability_exit_without_matching_entry_is_orphan_error_5717() {
    use crate::brick::profiler::ContractSeverity;
    let t = tracer5717(vec![
        ev5717(1, AwsEventType::TaskStateEntered, None),
        ev5717(2, AwsEventType::TaskStateExited, Some(99)),
    ]);
    let v = t.validate_observability_contracts();
    let orphan: Vec<_> = v.iter().filter(|(_, m)| m.contains("ORPHAN_SPAN")).collect();
    assert_eq!(orphan.len(), 1, "{v:?}");
    assert_eq!(orphan[0].0, ContractSeverity::Error);
    assert!(
        orphan[0].1.contains("id=2") && orphan[0].1.contains("previous_event_id=99"),
        "{v:?}"
    );
}

#[test]
fn observability_exit_with_no_link_is_not_an_orphan_5717() {
    let t = tracer5717(vec![
        ev5717(1, AwsEventType::TaskStateEntered, None),
        ev5717(2, AwsEventType::TaskStateExited, None),
    ]);
    let v = t.validate_observability_contracts();
    assert!(v.iter().all(|(_, m)| !m.contains("ORPHAN_SPAN")), "{v:?}");
}

#[test]
fn observability_unclosed_entry_is_dangling_warning_5717() {
    use crate::brick::profiler::ContractSeverity;
    let t = tracer5717(vec![
        ev5717(1, AwsEventType::TaskStateEntered, None),
        ev5717(2, AwsEventType::TaskStateEntered, None),
        ev5717(3, AwsEventType::TaskStateExited, Some(1)),
    ]);
    let v = t.validate_observability_contracts();
    assert_eq!(v.len(), 1, "{v:?}");
    assert_eq!(v[0].0, ContractSeverity::Warning);
    assert!(v[0].1.contains("DANGLING_ENTRY") && v[0].1.contains("id=2"), "{v:?}");
}
