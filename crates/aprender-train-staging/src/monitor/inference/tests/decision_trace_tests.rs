//! DecisionTrace Property Tests

use super::helpers::arb_decision_trace;
use crate::monitor::inference::path::DecisionPath;
use crate::monitor::inference::{DecisionTrace, LinearPath};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_decision_trace_serialization_roundtrip(trace in arb_decision_trace()) {
        let bytes = trace.to_bytes();
        let restored: DecisionTrace<LinearPath> =
            DecisionTrace::from_bytes(&bytes).expect("Deserialization failed");

        prop_assert_eq!(trace.timestamp_ns, restored.timestamp_ns);
        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
        prop_assert!((trace.output - restored.output).abs() < 1e-5);
        // Latency has microsecond precision
        prop_assert!(
            (trace.latency_ns as i64 - restored.latency_ns as i64).abs() < 1000,
            "Latency mismatch: {} vs {}",
            trace.latency_ns,
            restored.latency_ns
        );
    }

    #[test]
    fn prop_decision_trace_confidence_from_path(trace in arb_decision_trace()) {
        let trace_confidence = trace.confidence();
        let path_confidence = trace.path.confidence();
        prop_assert!((trace_confidence - path_confidence).abs() < 1e-6);
    }
}
