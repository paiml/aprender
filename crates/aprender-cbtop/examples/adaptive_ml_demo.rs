#![allow(clippy::disallowed_methods)]
//! Adaptive ML Thresholds Demo
//!
//! Demonstrates workload-specific threshold learning with ML.
//!
//! Run with: cargo run --example adaptive_ml_demo -p cbtop

use cbtop::{AdaptiveThresholdMl, MlThresholdConfig, TimeSeriesFeatures, WorkloadClass};

/// Train the ML threshold learner with sample data.
fn train_learner(ml: &mut AdaptiveThresholdMl) {
    // Create FFN workload samples (high variance pattern)
    println!("Training on FFN workload samples (high variance)...");
    let ffn_samples: Vec<f64> =
        (0..50).map(|i| 10.0 + (i as f64 * 0.5) + (i % 7) as f64 * 2.0).collect();
    for chunk in ffn_samples.chunks(10) {
        if chunk.len() >= 10 {
            ml.train(chunk, false).ok();
        }
    }

    // Create Matmul workload samples (low variance pattern)
    println!("Training on Matmul workload samples (low variance)...");
    let matmul_samples: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1)).collect();
    for chunk in matmul_samples.chunks(10) {
        if chunk.len() >= 10 {
            ml.train(chunk, false).ok();
        }
    }
}

/// Display per-workload thresholds.
fn show_thresholds(ml: &AdaptiveThresholdMl) {
    println!("\n=== Learned Per-Workload Thresholds ===");
    for class in [WorkloadClass::Ffn, WorkloadClass::Matmul, WorkloadClass::Attention] {
        let threshold = ml.get_threshold(class);
        println!("{:?}: CV threshold = {:.2}%", class, threshold);
    }
}

/// Demonstrate anomaly detection with normal and anomalous samples.
fn demo_anomaly_detection(ml: &mut AdaptiveThresholdMl) {
    println!("\n=== Anomaly Detection ===");
    let ffn_samples: Vec<f64> =
        (0..50).map(|i| 10.0 + (i as f64 * 0.5) + (i % 7) as f64 * 2.0).collect();

    let normal_chunk = &ffn_samples[20..30];
    if let Ok(result) = ml.detect_anomaly(normal_chunk) {
        println!(
            "Normal sample: is_anomaly={}, score={:.2}, reason: {}",
            result.is_anomaly, result.score, result.reason
        );
    }

    let anomalous: Vec<f64> = vec![10.0, 50.0, 10.0, 80.0, 10.0, 90.0, 10.0, 100.0, 10.0, 110.0];
    if let Ok(result) = ml.detect_anomaly(&anomalous) {
        println!(
            "Anomalous sample: is_anomaly={}, score={:.2}, reason: {}",
            result.is_anomaly, result.score, result.reason
        );
    }
}

/// Demonstrate workload classification and drift detection.
fn demo_classification_and_drift(ml: &mut AdaptiveThresholdMl) {
    let matmul_samples: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1)).collect();
    let anomalous: Vec<f64> = vec![10.0, 50.0, 10.0, 80.0, 10.0, 90.0, 10.0, 100.0, 10.0, 110.0];

    println!("\n=== Workload Classification ===");
    let features_low_cv = TimeSeriesFeatures::extract(&matmul_samples[..20]).unwrap();
    let features_high_cv = TimeSeriesFeatures::extract(&anomalous).unwrap();

    println!(
        "Low CV sample -> {:?} (CV={:.2}%)",
        ml.classify_workload(&features_low_cv),
        features_low_cv.cv
    );
    println!(
        "High CV sample -> {:?} (CV={:.2}%)",
        ml.classify_workload(&features_high_cv),
        features_high_cv.cv
    );

    println!("\n=== Drift Detection ===");
    let drifted: Vec<f64> = (0..30).map(|i| 200.0 + (i as f64 * 0.5)).collect();
    if let Ok(drift_zscore) = ml.check_drift(&drifted) {
        if let Some(zscore) = drift_zscore {
            println!("Drift detected: z-score = {:.2} (threshold: 3.0)", zscore);
        } else {
            println!("No significant drift detected");
        }
    }
}

/// Show classification metrics and available workload classes.
fn show_metrics_and_classes(ml: &AdaptiveThresholdMl) {
    println!("\n=== Classification Metrics ===");
    let metrics = ml.get_metrics();
    println!("True positives: {}", metrics.true_positives);
    println!("False positives: {}", metrics.false_positives);
    println!("True negatives: {}", metrics.true_negatives);
    println!("False negatives: {}", metrics.false_negatives);
    println!("Precision: {:.2}%", metrics.precision() * 100.0);
    println!("Recall: {:.2}%", metrics.recall() * 100.0);

    println!("\n=== Available Workload Classes ===");
    let classes = [
        WorkloadClass::Ffn,
        WorkloadClass::Matmul,
        WorkloadClass::Attention,
        WorkloadClass::Quantize,
        WorkloadClass::MemoryBound,
        WorkloadClass::ComputeBound,
        WorkloadClass::Unknown,
    ];
    for class in classes {
        println!("  {:?} (default CV threshold: {:.1}%)", class, class.default_cv_threshold());
    }
}

fn main() {
    println!("=== Adaptive ML Thresholds Demo ===\n");

    let config = MlThresholdConfig {
        min_training_samples: 5,
        min_confidence: 0.5,
        drift_zscore_threshold: 3.0,
        ..Default::default()
    };
    let mut ml = AdaptiveThresholdMl::new(config);

    train_learner(&mut ml);
    show_thresholds(&ml);
    demo_anomaly_detection(&mut ml);
    demo_classification_and_drift(&mut ml);
    show_metrics_and_classes(&ml);

    println!("\nAdaptive ML thresholds demo complete!");
}
