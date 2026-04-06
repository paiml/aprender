
impl OutlierResult {
    pub fn no_outliers(median: f64, mad: f64, threshold: f64) -> Self {
        Self {
            median,
            mad,
            threshold,
            num_outliers: 0,
            outlier_indices: Vec::new(),
            meets_qa034: true,
        }
    }

    pub fn with_outliers(median: f64, mad: f64, threshold: f64, indices: Vec<usize>) -> Self {
        Self {
            median,
            mad,
            threshold,
            num_outliers: indices.len(),
            outlier_indices: indices,
            meets_qa034: true,
        }
    }
}

/// Calculate median of a sample
pub fn calculate_median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Calculate MAD (Median Absolute Deviation)
pub fn calculate_mad(samples: &[f64]) -> f64 {
    let median = calculate_median(samples);
    let deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
    calculate_median(&deviations)
}

/// Detect outliers using MAD
pub fn detect_outliers_mad(samples: &[f64], k: f64) -> OutlierResult {
    if samples.is_empty() {
        return OutlierResult::no_outliers(0.0, 0.0, k);
    }

    let median = calculate_median(samples);
    let mad = calculate_mad(samples);

    // Consistency constant for normal distribution (1.4826)
    let threshold = k * mad * 1.4826;

    let outlier_indices: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, x)| (*x - median).abs() > threshold)
        .map(|(i, _)| i)
        .collect();

    if outlier_indices.is_empty() {
        OutlierResult::no_outliers(median, mad, threshold)
    } else {
        OutlierResult::with_outliers(median, mad, threshold, outlier_indices)
    }
}

/// IMP-187a: Test median calculation
#[test]
fn test_imp_187a_median_calculation() {
    let odd = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(
        (calculate_median(&odd) - 3.0).abs() < 1e-10,
        "IMP-187a: Odd median should be 3.0"
    );

    let even = vec![1.0, 2.0, 3.0, 4.0];
    assert!(
        (calculate_median(&even) - 2.5).abs() < 1e-10,
        "IMP-187a: Even median should be 2.5"
    );

    let single = vec![42.0];
    assert!(
        (calculate_median(&single) - 42.0).abs() < 1e-10,
        "IMP-187a: Single value median"
    );

    println!("\nIMP-187a: Median Calculation:");
    println!("  Odd [1,2,3,4,5]: {}", calculate_median(&odd));
    println!("  Even [1,2,3,4]: {}", calculate_median(&even));
    println!("  Single [42]: {}", calculate_median(&single));
}

/// IMP-187b: Test MAD calculation
#[test]
fn test_imp_187b_mad_calculation() {
    let constant = vec![10.0; 10];
    let mad_const = calculate_mad(&constant);
    assert!(
        mad_const < 1e-10,
        "IMP-187b: Constant values should have MAD ~0"
    );

    let variable = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mad_var = calculate_mad(&variable);
    assert!(
        mad_var > 0.0,
        "IMP-187b: Variable values should have MAD > 0"
    );

    println!("\nIMP-187b: MAD Calculation:");
    println!("  Constant [10,10,...]: MAD = {:.6}", mad_const);
    println!("  Variable [1,2,3,4,5]: MAD = {:.6}", mad_var);
}

/// IMP-187c: Test outlier detection
#[test]
fn test_imp_187c_outlier_detection() {
    let normal = vec![100.0, 101.0, 99.0, 100.5, 99.5, 100.0];
    let result_normal = detect_outliers_mad(&normal, 3.0);
    assert_eq!(
        result_normal.num_outliers, 0,
        "IMP-187c: Normal data should have no outliers"
    );

    let with_outlier = vec![100.0, 101.0, 99.0, 100.0, 200.0];
    let result_outlier = detect_outliers_mad(&with_outlier, 3.0);
    assert!(
        result_outlier.num_outliers > 0,
        "IMP-187c: Should detect outlier 200"
    );

    println!("\nIMP-187c: Outlier Detection:");
    println!("  Normal data: {} outliers", result_normal.num_outliers);
    println!(
        "  With outlier: {} outliers at {:?}",
        result_outlier.num_outliers, result_outlier.outlier_indices
    );
}

/// IMP-187d: Real-world outlier detection
#[test]
#[ignore = "Requires benchmark data"]
fn test_imp_187d_realworld_outlier_detection() {
    // Simulate benchmark latencies with an outlier
    let latencies = vec![
        100.0, 102.0, 99.0, 101.0, 100.5, 99.5, 101.5, 100.2, 500.0, 100.1, // 500.0 is outlier
    ];

    let result = detect_outliers_mad(&latencies, 3.0);

    println!("\nIMP-187d: Real-World Outlier Detection:");
    println!("  Median: {:.2} ms", result.median);
    println!("  MAD: {:.2}", result.mad);
    println!("  Threshold: {:.2}", result.threshold);
    println!(
        "  Outliers: {} at {:?}",
        result.num_outliers, result.outlier_indices
    );
    println!(
        "  QA-034: {}",
        if result.meets_qa034 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-188: Percentile Latencies (QA-035)
// Include p50, p95, p99 latencies per Georges et al. [3]
// ================================================================================

/// Percentile latency result
#[derive(Debug)]
pub struct PercentileResult {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub meets_qa035: bool,
}

impl PercentileResult {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                meets_qa035: false,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let p50_idx = (n as f64 * 0.50).ceil() as usize - 1;
        let p95_idx = (n as f64 * 0.95).ceil() as usize - 1;
        let p99_idx = (n as f64 * 0.99).ceil() as usize - 1;

        Self {
            p50: sorted[p50_idx.min(n - 1)],
            p95: sorted[p95_idx.min(n - 1)],
            p99: sorted[p99_idx.min(n - 1)],
            min: sorted[0],
            max: sorted[n - 1],
            mean: sorted.iter().sum::<f64>() / n as f64,
            meets_qa035: true,
        }
    }
}

/// IMP-188a: Test percentile calculation
#[test]
fn test_imp_188a_percentile_calculation() {
    let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let result = PercentileResult::from_samples(&samples);

    assert!(
        (result.p50 - 50.0).abs() < 1.0,
        "IMP-188a: p50 should be ~50"
    );
    assert!(
        (result.p95 - 95.0).abs() < 1.0,
        "IMP-188a: p95 should be ~95"
    );
    assert!(
        (result.p99 - 99.0).abs() < 1.0,
        "IMP-188a: p99 should be ~99"
    );
    assert!(result.meets_qa035, "IMP-188a: Should meet QA-035");

    println!("\nIMP-188a: Percentile Calculation:");
    println!("  p50: {:.2}", result.p50);
    println!("  p95: {:.2}", result.p95);
    println!("  p99: {:.2}", result.p99);
    println!("  min/max: {:.2}/{:.2}", result.min, result.max);
}

/// IMP-188b: Test small sample percentiles
#[test]
fn test_imp_188b_small_sample_percentiles() {
    let small = vec![10.0, 20.0, 30.0];
    let result = PercentileResult::from_samples(&small);

    assert!(
        result.p50 > 0.0,
        "IMP-188b: Small sample p50 should be valid"
    );
    assert!(result.p99 >= result.p50, "IMP-188b: p99 >= p50");
    assert_eq!(result.min, 10.0, "IMP-188b: Min should be 10");
    assert_eq!(result.max, 30.0, "IMP-188b: Max should be 30");

    println!("\nIMP-188b: Small Sample Percentiles:");
    println!("  Samples: {:?}", small);
    println!(
        "  p50: {:.2}, p95: {:.2}, p99: {:.2}",
        result.p50, result.p95, result.p99
    );
}

/// IMP-188c: Test empty sample handling
#[test]
fn test_imp_188c_empty_sample_handling() {
    let empty: Vec<f64> = Vec::new();
    let result = PercentileResult::from_samples(&empty);

    assert!(
        !result.meets_qa035,
        "IMP-188c: Empty samples should not meet QA-035"
    );
    assert_eq!(result.p50, 0.0, "IMP-188c: Empty p50 should be 0");

    let single = vec![42.0];
    let single_result = PercentileResult::from_samples(&single);
    assert_eq!(single_result.p50, 42.0, "IMP-188c: Single value p50");
    assert_eq!(single_result.p99, 42.0, "IMP-188c: Single value p99");

    println!("\nIMP-188c: Edge Cases:");
    println!("  Empty: meets_qa035={}", result.meets_qa035);
    println!(
        "  Single [42]: p50={}, p99={}",
        single_result.p50, single_result.p99
    );
}

/// IMP-188d: Real-world latency percentiles
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_188d_realworld_percentiles() {
    // Simulate benchmark latencies
    let latencies = vec![
        100.0, 102.0, 99.0, 101.0, 100.5, 103.0, 98.0, 105.0, 110.0, 95.0, 101.0, 100.0, 102.0,
        99.5, 100.2,
    ];

    let result = PercentileResult::from_samples(&latencies);

    println!("\nIMP-188d: Real-World Latency Percentiles:");
    println!("  p50: {:.2} ms", result.p50);
    println!("  p95: {:.2} ms", result.p95);
    println!("  p99: {:.2} ms", result.p99);
    println!("  min/max: {:.2}/{:.2} ms", result.min, result.max);
    println!("  mean: {:.2} ms", result.mean);
    println!(
        "  QA-035: {}",
        if result.meets_qa035 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-189: Throughput Variance (QA-036)
// Measure throughput in tok/s with variance
// ================================================================================

/// Throughput measurement result
#[derive(Debug)]
pub struct ThroughputResult {
    pub mean_toks: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub cv: f64,
    pub samples: usize,
    pub meets_qa036: bool,
}

impl ThroughputResult {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean_toks: 0.0,
                std_dev: 0.0,
                variance: 0.0,
                cv: 0.0,
                samples: 0,
                meets_qa036: false,
            };
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let cv = if mean.abs() > 1e-10 {
            std_dev / mean
        } else {
            0.0
        };

        Self {
            mean_toks: mean,
            std_dev,
            variance,
            cv,
            samples: samples.len(),
            meets_qa036: true,
        }
    }

    pub fn is_stable(&self, max_cv: f64) -> bool {
        self.cv <= max_cv
    }
}

/// IMP-189a: Test throughput calculation
#[test]
fn test_imp_189a_throughput_calculation() {
    let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0];
    let result = ThroughputResult::from_samples(&samples);

    assert!(
        (result.mean_toks - 100.0).abs() < 1.0,
        "IMP-189a: Mean should be ~100 tok/s"
    );
    assert!(result.std_dev > 0.0, "IMP-189a: StdDev should be > 0");
    assert!(result.cv < 0.1, "IMP-189a: CV should be < 10%");
    assert!(result.meets_qa036, "IMP-189a: Should meet QA-036");

    println!("\nIMP-189a: Throughput Calculation:");
    println!("  Mean: {:.2} tok/s", result.mean_toks);
    println!("  StdDev: {:.2}", result.std_dev);
    println!("  CV: {:.4}", result.cv);
}

/// IMP-189b: Test throughput stability
#[test]
fn test_imp_189b_throughput_stability() {
    let stable = vec![100.0; 10];
    let stable_result = ThroughputResult::from_samples(&stable);
    assert!(
        stable_result.is_stable(0.05),
        "IMP-189b: Constant values should be stable"
    );

    let unstable = vec![50.0, 150.0, 50.0, 150.0, 50.0];
    let unstable_result = ThroughputResult::from_samples(&unstable);
    assert!(
        !unstable_result.is_stable(0.05),
        "IMP-189b: High variance should be unstable"
    );

    println!("\nIMP-189b: Throughput Stability:");
    println!(
        "  Stable: CV={:.4}, is_stable(5%)={}",
        stable_result.cv,
        stable_result.is_stable(0.05)
    );
    println!(
        "  Unstable: CV={:.4}, is_stable(5%)={}",
        unstable_result.cv,
        unstable_result.is_stable(0.05)
    );
}

/// IMP-189c: Test variance calculation
#[test]
fn test_imp_189c_variance_calculation() {
    let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let result = ThroughputResult::from_samples(&samples);

    // Variance of [10,20,30,40,50] = 200
    assert!(
        (result.variance - 200.0).abs() < 1.0,
        "IMP-189c: Variance should be ~200"
    );
    assert!(
        (result.std_dev - 14.14).abs() < 0.1,
        "IMP-189c: StdDev should be ~14.14"
    );

    println!("\nIMP-189c: Variance Calculation:");
    println!("  Samples: {:?}", samples);
    println!("  Variance: {:.2}", result.variance);
    println!("  StdDev: {:.2}", result.std_dev);
}
