use super::*;

#[test]
fn test_latency_distribution_basic() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert_eq!(dist.sample_count, 100);
    // Median of 1-100 is ~50.5
    assert!((dist.p50 - 50.5).abs() < 2.0);
    // P99 of 1-100 is ~99
    assert!((dist.p99 - 99.0).abs() < 2.0);
    assert!(dist.min == 1.0);
    assert!(dist.max == 100.0);
}

#[test]
fn test_empty_samples() {
    let samples: Vec<f64> = vec![];
    assert!(LatencyDistribution::analyze(&samples).is_none());
}

#[test]
fn test_single_sample() {
    let samples = vec![42.0];
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert_eq!(dist.sample_count, 1);
    assert_eq!(dist.jitter, 0.0);
    assert_eq!(dist.p50, 42.0);
}

#[test]
fn test_tail_ratio() {
    // Samples with heavy tail
    let mut samples: Vec<f64> = (1..=95).map(|x| x as f64).collect();
    samples.extend(vec![500.0, 600.0, 700.0, 800.0, 1000.0]); // Tail

    let dist = LatencyDistribution::analyze(&samples).unwrap();
    assert!(dist.tail_ratio > 3.0);
    assert!(dist.has_tail_problem());
}

#[test]
fn test_jitter_calculation() {
    // Stable samples (zero jitter)
    let stable: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0, 10.0];
    let dist = LatencyDistribution::analyze(&stable).unwrap();
    assert_eq!(dist.jitter, 0.0);

    // Unstable samples (high jitter)
    let unstable: Vec<f64> = vec![10.0, 50.0, 10.0, 50.0, 10.0, 50.0];
    let dist2 = LatencyDistribution::analyze(&unstable).unwrap();
    assert!(dist2.jitter >= dist.jitter);
}

#[test]
fn test_histogram_counts() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    assert!(hist.verify_counts());
    assert_eq!(hist.total_samples, 100);
}

#[test]
fn test_entropy_uniform() {
    // Uniform distribution should have high entropy
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    // Should be close to 1.0 for uniform
    assert!(hist.entropy > 0.9);
}

#[test]
fn test_entropy_concentrated() {
    // Concentrated distribution should have low entropy
    let samples: Vec<f64> = vec![50.0; 100];
    let hist = LatencyHistogram::build(&samples, 10);

    // All in one bucket = 0 entropy
    assert!(hist.entropy < 0.1);
}

#[test]
fn test_tail_severity() {
    assert_eq!(TailSeverity::from_ratio(1.5), TailSeverity::Excellent);
    assert_eq!(TailSeverity::from_ratio(2.5), TailSeverity::Good);
    assert_eq!(TailSeverity::from_ratio(4.0), TailSeverity::Warning);
    assert_eq!(TailSeverity::from_ratio(10.0), TailSeverity::Critical);
}

#[test]
fn test_percentile_ordering() {
    let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert!(dist.p50 <= dist.p90);
    assert!(dist.p90 <= dist.p99);
    assert!(dist.p99 <= dist.p999);
}
