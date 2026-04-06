//! RingCollector and HashChainCollector Property Tests

use super::helpers::arb_decision_trace;
use crate::monitor::inference::{
    DecisionTrace, HashChainCollector, LinearPath, RingCollector, TraceCollector,
};
use proptest::prelude::*;

// =============================================================================
// RingCollector Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_ring_collector_bounded(traces in prop::collection::vec(arb_decision_trace(), 0..100)) {
        let mut collector = RingCollector::<LinearPath, 32>::new();

        for trace in traces {
            collector.record(trace);
        }

        prop_assert!(collector.len() <= 32, "Ring buffer exceeded capacity: {}", collector.len());
    }

    #[test]
    fn prop_ring_collector_recent_order(traces in prop::collection::vec(arb_decision_trace(), 1..50)) {
        let mut collector = RingCollector::<LinearPath, 64>::new();

        for trace in &traces {
            collector.record(trace.clone());
        }

        let recent = collector.recent(traces.len());

        // Most recent should be last inserted
        if !recent.is_empty() {
            prop_assert_eq!(
                recent[0].sequence,
                traces.last().expect("collection should not be empty").sequence,
                "Most recent trace mismatch"
            );
        }
    }

    #[test]
    fn prop_ring_collector_all_preserves_order(n_traces in 1..20usize) {
        let mut collector = RingCollector::<LinearPath, 64>::new();

        for i in 0..n_traces {
            let path = LinearPath::new(vec![i as f32], 0.0, 0.0, 0.0);
            let trace = DecisionTrace::new(0, i as u64, 0, path, 0.0, 0);
            collector.record(trace);
        }

        let all = collector.all();

        // Should be in insertion order (oldest first)
        for i in 0..all.len() {
            prop_assert_eq!(all[i].sequence, i as u64, "Order mismatch at index {}", i);
        }
    }
}

// =============================================================================
// HashChainCollector Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_hash_chain_integrity(traces in prop::collection::vec(arb_decision_trace(), 1..20)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let verification = collector.verify_chain();
        prop_assert!(verification.valid, "Chain integrity failed: {:?}", verification.error);
    }

    #[test]
    fn prop_hash_chain_sequence_monotonic(traces in prop::collection::vec(arb_decision_trace(), 1..20)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let entries = collector.entries();
        for i in 0..entries.len() {
            prop_assert_eq!(entries[i].sequence, i as u64, "Sequence not monotonic at {}", i);
        }
    }

    #[test]
    fn prop_hash_chain_linking(traces in prop::collection::vec(arb_decision_trace(), 2..10)) {
        let mut collector = HashChainCollector::<LinearPath>::new();

        for trace in traces {
            collector.record(trace);
        }

        let entries = collector.entries();

        // Each entry's prev_hash should match previous entry's hash
        for i in 1..entries.len() {
            prop_assert_eq!(
                entries[i].prev_hash,
                entries[i-1].hash,
                "Hash linking broken at index {}",
                i
            );
        }
    }
}
