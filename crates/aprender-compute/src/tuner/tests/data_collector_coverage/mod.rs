//! Comprehensive coverage tests for `data_collector.rs`.
//!
//! Covers: constructors, online learning toggle, record_feedback, get_feedback,
//! record_prediction_error, detect_concept_drift (all branches), should_retrain
//! (all branches), mark_trained, merge, prepare_weighted_training_data,
//! training_stats with feedback, save_apr/load_apr round-trip, load_apr error
//! paths (bad magic, CRC mismatch, truncated file), UserFeedback default,
//! auto_retrain edge cases, to_json/from_json, and ready_to_train.

use super::super::*;
use crate::brick::BrickProfiler;

// ============================================================================
// Helper: create a TrainingSample with given throughput
// ============================================================================

fn make_sample(throughput: f32) -> TrainingSample {
    let features =
        TunerFeatures::builder().model_params_b(7.0).hidden_dim(4096).batch_size(1).build();
    TrainingSample {
        features,
        throughput_tps: throughput,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1700000000".to_string(),
        hardware_id: "test-hw".to_string(),
    }
}

fn make_collector_with_samples(n: usize) -> TunerDataCollector {
    let mut collector = TunerDataCollector::new();
    for i in 0..n {
        collector.samples.push(make_sample(100.0 + i as f32));
    }
    collector
}

mod constructors_and_basics;
mod drift_and_retrain;
mod feedback_and_errors;
mod json_and_training;
mod persistence;
mod structs_and_edge_cases;
