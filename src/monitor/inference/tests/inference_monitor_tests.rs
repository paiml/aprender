//! InferenceMonitor Tests

use crate::monitor::inference::{
    Explainable, InferenceMonitor, LinearPath, RingCollector, SafetyAndon, SafetyIntegrityLevel,
};

// =============================================================================
// Utility Function Tests
// =============================================================================

#[test]
fn test_monotonic_ns_increasing() {
    use crate::monitor::inference::monotonic_ns;

    let ts1 = monotonic_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    let ts2 = monotonic_ns();
    assert!(ts2 > ts1, "monotonic_ns should be strictly increasing over time");
}

#[test]
fn test_monotonic_ns_non_zero() {
    use crate::monitor::inference::monotonic_ns;

    // The first call might be 0 if called at exactly initialization time
    // but subsequent calls should not be 0
    std::thread::sleep(std::time::Duration::from_micros(1));
    let ts = monotonic_ns();
    assert!(ts > 0, "monotonic_ns should return a positive value after initialization");
}

// =============================================================================
// InferenceMonitor Tests
// =============================================================================

// Mock model for testing InferenceMonitor
struct MockModel;

impl Explainable for MockModel {
    type Path = LinearPath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let features_per_sample = x.len() / n_samples;
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * features_per_sample;
            let sample = &x[start..start + features_per_sample];
            let output: f32 = sample.iter().sum();
            outputs.push(output);
            paths.push(LinearPath::new(sample.to_vec(), 0.0, output, output));
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let output: f32 = sample.iter().sum();
        LinearPath::new(sample.to_vec(), 0.0, output, output)
    }
}

#[test]
fn test_inference_monitor_creation() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_with_latency_budget() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector).with_latency_budget_ns(5_000_000);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_with_andon() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);
    let monitor = InferenceMonitor::new(model, collector).with_andon(andon);
    assert_eq!(monitor.sequence(), 0);
}

#[test]
fn test_inference_monitor_predict() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let n_samples = 2;

    let outputs = monitor.predict(&features, n_samples);

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], 6.0); // 1 + 2 + 3
    assert_eq!(outputs[1], 15.0); // 4 + 5 + 6
    assert_eq!(monitor.sequence(), 2);
}

#[test]
fn test_inference_monitor_model_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);

    let _model_ref = monitor.model();
}

#[test]
fn test_inference_monitor_collector_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let monitor = InferenceMonitor::new(model, collector);

    let _collector_ref = monitor.collector();
}

#[test]
fn test_inference_monitor_collector_mut_accessor() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let _collector_ref = monitor.collector_mut();
}

#[test]
fn test_inference_monitor_traces_are_recorded() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    let features = [1.0f32, 2.0, 3.0];
    monitor.predict(&features, 1);

    let traces = monitor.collector().all();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].output, 6.0);
}

#[test]
fn test_inference_monitor_multiple_predictions() {
    let model = MockModel;
    let collector = RingCollector::<LinearPath, 64>::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    // First batch
    monitor.predict(&[1.0, 2.0, 3.0], 1);
    assert_eq!(monitor.sequence(), 1);

    // Second batch
    monitor.predict(&[4.0, 5.0, 6.0], 1);
    assert_eq!(monitor.sequence(), 2);

    let traces = monitor.collector().all();
    assert_eq!(traces.len(), 2);
}
