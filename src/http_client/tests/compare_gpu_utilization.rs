
impl GpuUtilizationComparison {
    pub fn compare(measurements: Vec<GpuUtilizationMeasurement>) -> Self {
        if measurements.is_empty() {
            return Self {
                measurements: Vec::new(),
                most_efficient: "none".to_string(),
                least_efficient: "none".to_string(),
                overall_avg: 0.0,
            };
        }

        let overall_avg = measurements
            .iter()
            .map(|m| m.avg_utilization_percent)
            .sum::<f64>()
            / measurements.len() as f64;

        let most = measurements
            .iter()
            .max_by(|a, b| {
                a.avg_utilization_percent
                    .partial_cmp(&b.avg_utilization_percent)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        let least = measurements
            .iter()
            .min_by(|a, b| {
                a.avg_utilization_percent
                    .partial_cmp(&b.avg_utilization_percent)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        Self {
            measurements,
            most_efficient: most,
            least_efficient: least,
            overall_avg,
        }
    }
}

/// IMP-167b: Test GPU utilization comparison
#[test]
fn test_imp_167b_gpu_utilization_comparison() {
    let measurements = vec![
        GpuUtilizationMeasurement::from_samples("llama.cpp", &[92.0, 94.0, 91.0, 93.0, 90.0]),
        GpuUtilizationMeasurement::from_samples("Ollama", &[78.0, 82.0, 80.0, 79.0, 81.0]),
        GpuUtilizationMeasurement::from_samples("Realizar", &[65.0, 70.0, 68.0, 67.0, 69.0]),
    ];

    let comparison = GpuUtilizationComparison::compare(measurements);

    // IMP-167b: llama.cpp should be most efficient
    assert_eq!(
        comparison.most_efficient, "llama.cpp",
        "IMP-167b: llama.cpp should have highest GPU utilization"
    );

    // IMP-167b: Realizar should be least efficient
    assert_eq!(
        comparison.least_efficient, "Realizar",
        "IMP-167b: Realizar should have lowest GPU utilization"
    );

    println!("\nIMP-167b: GPU Utilization Comparison:");
    for m in &comparison.measurements {
        println!(
            "  {}: {:.1}% avg, QA-014={}",
            m.server_name, m.avg_utilization_percent, m.meets_qa014
        );
    }
    println!("  Most efficient: {}", comparison.most_efficient);
    println!("  Least efficient: {}", comparison.least_efficient);
    println!("  Overall average: {:.1}%", comparison.overall_avg);
}

/// IMP-167c: GPU utilization over time analysis
#[derive(Debug, Clone)]
pub struct GpuUtilizationTimeSeries {
    /// Time points (in seconds)
    pub timestamps: Vec<f64>,
    /// Utilization at each time point
    pub utilization: Vec<f64>,
    /// Whether utilization is stable (CV < 15%)
    pub is_stable: bool,
    /// CV of utilization
    pub cv: f64,
}

impl GpuUtilizationTimeSeries {
    pub fn analyze(timestamps: &[f64], utilization: &[f64]) -> Self {
        if utilization.is_empty() {
            return Self {
                timestamps: Vec::new(),
                utilization: Vec::new(),
                is_stable: true,
                cv: 0.0,
            };
        }

        let n = utilization.len();
        let mean = utilization.iter().sum::<f64>() / n as f64;
        let variance = utilization.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

        Self {
            timestamps: timestamps.to_vec(),
            utilization: utilization.to_vec(),
            is_stable: cv < 0.15, // 15% CV threshold
            cv,
        }
    }
}

/// IMP-167c: Test GPU utilization time series
#[test]
fn test_imp_167c_gpu_utilization_timeseries() {
    // Stable utilization
    let stable_times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let stable_util = vec![88.0, 90.0, 89.0, 91.0, 88.0, 90.0];
    let stable = GpuUtilizationTimeSeries::analyze(&stable_times, &stable_util);

    assert!(
        stable.is_stable,
        "IMP-167c: Low variance utilization should be stable"
    );
    assert!(
        stable.cv < 0.05,
        "IMP-167c: CV should be low, got {:.4}",
        stable.cv
    );

    // Unstable utilization (spiky)
    let unstable_util = vec![90.0, 30.0, 85.0, 25.0, 88.0, 35.0];
    let unstable = GpuUtilizationTimeSeries::analyze(&stable_times, &unstable_util);

    assert!(
        !unstable.is_stable,
        "IMP-167c: High variance utilization should be unstable"
    );

    println!("\nIMP-167c: GPU Utilization Time Series:");
    println!(
        "  Stable: CV={:.4}, is_stable={}",
        stable.cv, stable.is_stable
    );
    println!(
        "  Unstable: CV={:.4}, is_stable={}",
        unstable.cv, unstable.is_stable
    );
}

/// IMP-167d: Real-world GPU utilization (placeholder)
#[test]
#[ignore = "Requires GPU monitoring tools (nvidia-smi)"]
fn test_imp_167d_realworld_gpu_utilization() {
    // This test would require nvidia-smi or similar GPU monitoring
    // test values based on typical observations
    let measurements = vec![
        GpuUtilizationMeasurement::from_samples("llama.cpp", &[92.0, 94.0, 91.0, 93.0, 90.0]),
        GpuUtilizationMeasurement::from_samples("Realizar", &[68.0, 72.0, 70.0, 69.0, 71.0]),
    ];

    let comparison = GpuUtilizationComparison::compare(measurements);

    println!("\nIMP-167d: Real-World GPU Utilization:");
    for m in &comparison.measurements {
        println!(
            "  {}: {:.1}% avg, efficiency={:.1}%, QA-014={}",
            m.server_name,
            m.avg_utilization_percent,
            m.utilization_efficiency(),
            m.meets_qa014
        );
    }
}

// =========================================================================
// IMP-168: Memory Leak Detection (QA-015, EXTREME TDD)
// =========================================================================
// Per spec QA-015: No memory leaks over 1000 inference cycles
// Run with: cargo test test_imp_168 --lib --features bench-http

/// IMP-168a: Memory leak detector
#[derive(Debug, Clone)]
pub struct MemoryLeakDetector {
    /// Memory samples at each checkpoint (in MB)
    pub memory_samples: Vec<f64>,
    /// Number of inference cycles at each checkpoint
    pub cycle_counts: Vec<usize>,
    /// Leak rate (MB per 1000 cycles)
    pub leak_rate_per_1000: f64,
    /// Whether leak is detected (> 10 MB per 1000 cycles)
    pub leak_detected: bool,
    /// Confidence in detection (based on R² of linear fit)
    pub confidence: f64,
}

impl MemoryLeakDetector {
    pub fn analyze(cycle_counts: &[usize], memory_mb: &[f64]) -> Self {
        if cycle_counts.len() < 2 || memory_mb.len() < 2 {
            return Self {
                memory_samples: memory_mb.to_vec(),
                cycle_counts: cycle_counts.to_vec(),
                leak_rate_per_1000: 0.0,
                leak_detected: false,
                confidence: 0.0,
            };
        }

        // Linear regression to find leak rate
        let n = cycle_counts.len() as f64;
        let sum_x: f64 = cycle_counts.iter().map(|&x| x as f64).sum();
        let sum_y: f64 = memory_mb.iter().sum();
        let sum_xy: f64 = cycle_counts
            .iter()
            .zip(memory_mb.iter())
            .map(|(&x, &y)| x as f64 * y)
            .sum();
        let sum_xx: f64 = cycle_counts.iter().map(|&x| (x as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R² for confidence
        let mean_y = sum_y / n;
        let ss_tot: f64 = memory_mb.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = cycle_counts
            .iter()
            .zip(memory_mb.iter())
            .map(|(&x, &y)| {
                let predicted = slope * x as f64 + intercept;
                (y - predicted).powi(2)
            })
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Leak rate per 1000 cycles
        let leak_rate = slope * 1000.0;

        // Leak detected if rate > 10 MB per 1000 cycles with high confidence
        let leak_detected = leak_rate > 10.0 && r_squared > 0.7;

        Self {
            memory_samples: memory_mb.to_vec(),
            cycle_counts: cycle_counts.to_vec(),
            leak_rate_per_1000: leak_rate,
            leak_detected,
            confidence: r_squared,
        }
    }

    /// Estimate memory after N cycles
    pub fn estimate_memory_at(&self, cycles: usize) -> f64 {
        if self.memory_samples.is_empty() {
            return 0.0;
        }
        let base = self.memory_samples[0];
        base + (self.leak_rate_per_1000 / 1000.0) * cycles as f64
    }
}

/// IMP-168a: Test memory leak detection
#[test]
fn test_imp_168a_memory_leak_detection() {
    // No leak: memory stays constant
    let no_leak_cycles = vec![0, 200, 400, 600, 800, 1000];
    let no_leak_memory = vec![1000.0, 1002.0, 998.0, 1001.0, 999.0, 1000.0];
    let no_leak = MemoryLeakDetector::analyze(&no_leak_cycles, &no_leak_memory);

    assert!(
        !no_leak.leak_detected,
        "IMP-168a: Stable memory should not detect leak"
    );
    assert!(
        no_leak.leak_rate_per_1000.abs() < 5.0,
        "IMP-168a: Leak rate should be near zero, got {:.2}",
        no_leak.leak_rate_per_1000
    );

    // Clear leak: memory grows linearly
    let leak_cycles = vec![0, 200, 400, 600, 800, 1000];
    let leak_memory = vec![1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0];
    let leak = MemoryLeakDetector::analyze(&leak_cycles, &leak_memory);

    assert!(
        leak.leak_detected,
        "IMP-168a: Growing memory should detect leak"
    );
    assert!(
        leak.leak_rate_per_1000 > 40.0,
        "IMP-168a: Leak rate should be ~50 MB/1000 cycles, got {:.2}",
        leak.leak_rate_per_1000
    );

    println!("\nIMP-168a: Memory Leak Detection:");
    println!(
        "  No leak: rate={:.2} MB/1000 cycles, detected={}, confidence={:.2}",
        no_leak.leak_rate_per_1000, no_leak.leak_detected, no_leak.confidence
    );
    println!(
        "  Leak: rate={:.2} MB/1000 cycles, detected={}, confidence={:.2}",
        leak.leak_rate_per_1000, leak.leak_detected, leak.confidence
    );
}

/// IMP-168b: Long-term memory stability test
#[derive(Debug, Clone)]
pub struct MemoryStabilityTest {
    /// Initial memory (MB)
    pub initial_memory_mb: f64,
    /// Final memory (MB)
    pub final_memory_mb: f64,
    /// Total cycles run
    pub total_cycles: usize,
    /// Memory growth (MB)
    pub growth_mb: f64,
    /// Growth percentage
    pub growth_percent: f64,
    /// Passes QA-015 (no significant growth over 1000 cycles)
    pub passes_qa015: bool,
}

impl MemoryStabilityTest {
    pub fn evaluate(initial_mb: f64, final_mb: f64, cycles: usize) -> Self {
        let growth_mb = final_mb - initial_mb;
        let growth_percent = if initial_mb > 0.0 {
            (growth_mb / initial_mb) * 100.0
        } else {
            0.0
        };

        // QA-015: No significant memory growth over 1000 cycles
        // Allow up to 5% growth or 50 MB, whichever is larger
        let max_allowed_mb = (initial_mb * 0.05).max(50.0);
        let passes = growth_mb < max_allowed_mb;

        Self {
            initial_memory_mb: initial_mb,
            final_memory_mb: final_mb,
            total_cycles: cycles,
            growth_mb,
            growth_percent,
            passes_qa015: passes,
        }
    }
}

/// IMP-168b: Test memory stability
#[test]
fn test_imp_168b_memory_stability() {
    // Stable: minimal growth
    let stable = MemoryStabilityTest::evaluate(1000.0, 1010.0, 1000);
    assert!(
        stable.passes_qa015,
        "IMP-168b: 1% growth should pass QA-015"
    );

    // Leak: significant growth
    let leak = MemoryStabilityTest::evaluate(1000.0, 1200.0, 1000);
    assert!(
        !leak.passes_qa015,
        "IMP-168b: 20% growth should fail QA-015"
    );

    println!("\nIMP-168b: Memory Stability Test:");
    println!(
        "  Stable: {:.0}MB → {:.0}MB ({:.1}%), QA-015={}",
        stable.initial_memory_mb,
        stable.final_memory_mb,
        stable.growth_percent,
        stable.passes_qa015
    );
    println!(
        "  Leak: {:.0}MB → {:.0}MB ({:.1}%), QA-015={}",
        leak.initial_memory_mb, leak.final_memory_mb, leak.growth_percent, leak.passes_qa015
    );
}

/// IMP-168c: Memory fragmentation detection
#[derive(Debug, Clone)]
pub struct MemoryFragmentationAnalysis {
    /// Allocated memory (MB)
    pub allocated_mb: f64,
    /// Actual used memory (MB)
    pub used_mb: f64,
    /// Fragmentation ratio (allocated / used)
    pub fragmentation_ratio: f64,
    /// Whether fragmentation is acceptable (< 1.5x)
    pub acceptable: bool,
}

impl MemoryFragmentationAnalysis {
    pub fn analyze(allocated_mb: f64, used_mb: f64) -> Self {
        let ratio = if used_mb > 0.0 {
            allocated_mb / used_mb
        } else {
            1.0
        };
        Self {
            allocated_mb,
            used_mb,
            fragmentation_ratio: ratio,
            acceptable: ratio < 1.5,
        }
    }
}

/// IMP-168c: Test fragmentation detection
#[test]
fn test_imp_168c_fragmentation_detection() {
    // Low fragmentation
    let low = MemoryFragmentationAnalysis::analyze(1100.0, 1000.0);
    assert!(
        low.acceptable,
        "IMP-168c: 1.1x fragmentation should be acceptable"
    );

    // High fragmentation
    let high = MemoryFragmentationAnalysis::analyze(2000.0, 1000.0);
    assert!(
        !high.acceptable,
        "IMP-168c: 2.0x fragmentation should not be acceptable"
    );

    println!("\nIMP-168c: Memory Fragmentation:");
    println!(
        "  Low: allocated={:.0}MB, used={:.0}MB, ratio={:.2}x, acceptable={}",
        low.allocated_mb, low.used_mb, low.fragmentation_ratio, low.acceptable
    );
    println!(
        "  High: allocated={:.0}MB, used={:.0}MB, ratio={:.2}x, acceptable={}",
        high.allocated_mb, high.used_mb, high.fragmentation_ratio, high.acceptable
    );
}
