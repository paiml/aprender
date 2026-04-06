//! Test Helpers for Inference Monitor Property Tests
//!
//! Provides arbitrary generators (proptest strategies) for property-based testing.

use crate::monitor::inference::{DecisionTrace, LeafInfo, LinearPath, TreePath, TreeSplit};
use proptest::prelude::*;

pub fn arb_linear_path() -> impl Strategy<Value = LinearPath> {
    (
        prop::collection::vec(-10.0f32..10.0, 1..20), // contributions
        -10.0f32..10.0,                               // intercept
        -10.0f32..10.0,                               // logit
        -10.0f32..10.0,                               // prediction
        prop::option::of(-1.0f32..1.0),               // probability
    )
        .prop_map(|(contributions, intercept, logit, prediction, probability)| {
            let mut path = LinearPath::new(contributions, intercept, logit, prediction);
            if let Some(prob) = probability {
                path = path.with_probability(prob.abs().min(1.0));
            }
            path
        })
}

pub fn arb_tree_split() -> impl Strategy<Value = TreeSplit> {
    (
        0..100usize,      // feature_idx
        -100.0f32..100.0, // threshold
        any::<bool>(),    // went_left
        1..10000usize,    // n_samples
    )
        .prop_map(|(feature_idx, threshold, went_left, n_samples)| TreeSplit {
            feature_idx,
            threshold,
            went_left,
            n_samples,
        })
}

pub fn arb_tree_path() -> impl Strategy<Value = TreePath> {
    (
        prop::collection::vec(arb_tree_split(), 0..10), // splits
        -10.0f32..10.0,                                 // prediction
        1..1000usize,                                   // n_samples
        prop::option::of(prop::collection::vec(0.0f32..1.0, 2..10)), // class_dist
    )
        .prop_map(|(splits, prediction, n_samples, class_distribution)| {
            let leaf = LeafInfo { prediction, n_samples, class_distribution };
            TreePath::new(splits, leaf)
        })
}

pub fn arb_decision_trace() -> impl Strategy<Value = DecisionTrace<LinearPath>> {
    (
        arb_linear_path(),
        0..u64::MAX,     // timestamp_ns
        0..u64::MAX,     // sequence
        0..u64::MAX,     // input_hash
        -10.0f32..10.0,  // output
        0..1_000_000u64, // latency_ns
    )
        .prop_map(|(path, timestamp_ns, sequence, input_hash, output, latency_ns)| {
            DecisionTrace::new(timestamp_ns, sequence, input_hash, path, output, latency_ns)
        })
}
