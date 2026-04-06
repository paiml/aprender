use crate::http_client::tests::part_02::LatencyPercentiles;
use crate::http_client::*;
// =========================================================================
// Per spec QA-016: Cold start latency < 5 seconds for 7B model
// Run with: cargo test test_imp_166 --lib --features bench-http

/// IMP-166a: Cold start measurement
#[derive(Debug, Clone)]
pub struct ColdStartMeasurement {
    /// Server name
    pub server_name: String,
    /// Model size category ("7B", "13B", "70B")
    pub model_size: String,
    /// Time to first token in milliseconds
    pub ttft_ms: f64,
    /// Target cold start (QA-016: 5s for 7B)
    pub target_ms: f64,
    /// Whether it meets the target
    pub meets_target: bool,
    /// Margin to target (positive = good, negative = exceeded)
    pub margin_ms: f64,
}

impl ColdStartMeasurement {
    pub fn new(server_name: &str, model_size: &str, ttft_ms: f64) -> Self {
        // QA-016 targets by model size
        let target_ms = match model_size {
            "7B" => 5000.0,
            "13B" => 10000.0,
            "70B" => 30000.0,
            _ => 5000.0,
        };

        let meets_target = ttft_ms < target_ms;
        let margin_ms = target_ms - ttft_ms;

        Self {
            server_name: server_name.to_string(),
            model_size: model_size.to_string(),
            ttft_ms,
            target_ms,
            meets_target,
            margin_ms,
        }
    }

    /// Get target description
    pub fn target_description(&self) -> String {
        format!("{}ms for {} model", self.target_ms, self.model_size)
    }
}

/// IMP-166a: Test cold start measurement
#[test]
fn test_imp_166a_cold_start_measurement() {
    // Fast cold start (meets QA-016)
    let fast = ColdStartMeasurement::new("FastServer", "7B", 2000.0);
    assert!(
        fast.meets_target,
        "IMP-166a: 2s cold start should meet 5s target"
    );
    assert!(fast.margin_ms > 2000.0, "IMP-166a: Should have 3s margin");

    // Slow cold start (fails QA-016)
    let slow = ColdStartMeasurement::new("SlowServer", "7B", 8000.0);
    assert!(
        !slow.meets_target,
        "IMP-166a: 8s cold start should fail 5s target"
    );
    assert!(
        slow.margin_ms < 0.0,
        "IMP-166a: Should have negative margin"
    );

    // Larger model has higher target
    let large = ColdStartMeasurement::new("LargeModelServer", "70B", 25000.0);
    assert!(
        large.meets_target,
        "IMP-166a: 25s cold start should meet 30s target for 70B"
    );

    println!("\nIMP-166a: Cold Start Measurement:");
    println!(
        "  Fast (7B): {:.0}ms, target={}, margin={:.0}ms",
        fast.ttft_ms,
        fast.target_description(),
        fast.margin_ms
    );
    println!(
        "  Slow (7B): {:.0}ms, target={}, margin={:.0}ms",
        slow.ttft_ms,
        slow.target_description(),
        slow.margin_ms
    );
    println!(
        "  Large (70B): {:.0}ms, target={}, margin={:.0}ms",
        large.ttft_ms,
        large.target_description(),
        large.margin_ms
    );
}

/// IMP-166b: Cold start breakdown analysis
#[derive(Debug, Clone)]
pub struct ColdStartBreakdown {
    /// Total time to first token
    pub total_ttft_ms: f64,
    /// Model loading time
    pub model_load_ms: f64,
    /// First inference time
    pub first_inference_ms: f64,
    /// Other overhead
    pub overhead_ms: f64,
    /// Bottleneck component
    pub bottleneck: String,
}

impl ColdStartBreakdown {
    pub fn analyze(model_load_ms: f64, first_inference_ms: f64, total_ttft_ms: f64) -> Self {
        let overhead_ms = (total_ttft_ms - model_load_ms - first_inference_ms).max(0.0);

        let bottleneck = if model_load_ms >= first_inference_ms && model_load_ms >= overhead_ms {
            "model_loading"
        } else if first_inference_ms >= model_load_ms && first_inference_ms >= overhead_ms {
            "first_inference"
        } else {
            "overhead"
        };

        Self {
            total_ttft_ms,
            model_load_ms,
            first_inference_ms,
            overhead_ms,
            bottleneck: bottleneck.to_string(),
        }
    }

    /// Get percentage breakdown
    pub fn percentage_breakdown(&self) -> (f64, f64, f64) {
        if self.total_ttft_ms > 0.0 {
            (
                self.model_load_ms / self.total_ttft_ms * 100.0,
                self.first_inference_ms / self.total_ttft_ms * 100.0,
                self.overhead_ms / self.total_ttft_ms * 100.0,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

/// IMP-166b: Test cold start breakdown
#[test]
fn test_imp_166b_cold_start_breakdown() {
    // Model loading dominated
    let load_heavy = ColdStartBreakdown::analyze(3000.0, 500.0, 4000.0);
    assert_eq!(
        load_heavy.bottleneck, "model_loading",
        "IMP-166b: Model loading should be bottleneck"
    );

    // Inference dominated (JIT warm-up)
    let inference_heavy = ColdStartBreakdown::analyze(500.0, 3000.0, 4000.0);
    assert_eq!(
        inference_heavy.bottleneck, "first_inference",
        "IMP-166b: First inference should be bottleneck"
    );

    let (load_pct, inf_pct, overhead_pct) = load_heavy.percentage_breakdown();

    println!("\nIMP-166b: Cold Start Breakdown:");
    println!(
        "  Load-heavy: model={:.0}ms, inference={:.0}ms, overhead={:.0}ms",
        load_heavy.model_load_ms, load_heavy.first_inference_ms, load_heavy.overhead_ms
    );
    println!(
        "  Percentages: {:.1}% load, {:.1}% inference, {:.1}% overhead",
        load_pct, inf_pct, overhead_pct
    );
    println!("  Bottleneck: {}", load_heavy.bottleneck);
}

/// IMP-166c: Cold vs warm latency comparison
#[derive(Debug, Clone)]
pub struct ColdWarmLatencyComparison {
    /// Cold start latency (first request)
    pub cold_ms: f64,
    /// Warm latency (subsequent average)
    pub warm_ms: f64,
    /// Cold start penalty (cold / warm)
    pub penalty_ratio: f64,
    /// Whether penalty is acceptable (< 10x)
    pub acceptable_penalty: bool,
}

impl ColdWarmLatencyComparison {
    pub fn analyze(latencies: &[f64]) -> Self {
        if latencies.is_empty() {
            return Self {
                cold_ms: 0.0,
                warm_ms: 0.0,
                penalty_ratio: 1.0,
                acceptable_penalty: true,
            };
        }

        let cold_ms = latencies[0];
        let warm_ms = if latencies.len() > 3 {
            // Skip first 3 for warm measurement
            latencies[3..].iter().sum::<f64>() / (latencies.len() - 3) as f64
        } else if latencies.len() > 1 {
            latencies[1..].iter().sum::<f64>() / (latencies.len() - 1) as f64
        } else {
            cold_ms
        };

        let penalty_ratio = if warm_ms > 0.0 {
            cold_ms / warm_ms
        } else {
            1.0
        };
        let acceptable_penalty = penalty_ratio < 10.0;

        Self {
            cold_ms,
            warm_ms,
            penalty_ratio,
            acceptable_penalty,
        }
    }
}

/// IMP-166c: Test cold/warm latency comparison
#[test]
fn test_imp_166c_cold_warm_comparison() {
    // Normal cold start penalty (5x)
    let normal = vec![
        500.0, 150.0, 105.0, 100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0,
    ];
    let normal_analysis = ColdWarmLatencyComparison::analyze(&normal);

    assert!(
        normal_analysis.penalty_ratio > 4.0 && normal_analysis.penalty_ratio < 6.0,
        "IMP-166c: Penalty should be ~5x, got {:.2}x",
        normal_analysis.penalty_ratio
    );
    assert!(
        normal_analysis.acceptable_penalty,
        "IMP-166c: 5x penalty should be acceptable"
    );

    // Extreme cold start penalty (20x)
    let extreme = vec![
        2000.0, 150.0, 105.0, 100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0,
    ];
    let extreme_analysis = ColdWarmLatencyComparison::analyze(&extreme);

    assert!(
        !extreme_analysis.acceptable_penalty,
        "IMP-166c: 20x penalty should not be acceptable"
    );

    println!("\nIMP-166c: Cold/Warm Latency Comparison:");
    println!(
        "  Normal: cold={:.0}ms, warm={:.0}ms, penalty={:.2}x, acceptable={}",
        normal_analysis.cold_ms,
        normal_analysis.warm_ms,
        normal_analysis.penalty_ratio,
        normal_analysis.acceptable_penalty
    );
    println!(
        "  Extreme: cold={:.0}ms, warm={:.0}ms, penalty={:.2}x, acceptable={}",
        extreme_analysis.cold_ms,
        extreme_analysis.warm_ms,
        extreme_analysis.penalty_ratio,
        extreme_analysis.acceptable_penalty
    );
}

/// IMP-166d: Real-world cold start measurement
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_166d_realworld_cold_start() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    // The server should be freshly started (cold) for accurate measurement
    let client = ModelHttpClient::with_timeout(120); // Long timeout for cold start
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 5,
        temperature: Some(0.0),
        stream: false,
    };

    // Measure cold start
    let cold_start = std::time::Instant::now();
    let cold_result = client.llamacpp_completion("http://127.0.0.1:8082", &request);
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

    if cold_result.is_err() {
        println!("IMP-166d: Cold start request failed");
        return;
    }

    // Collect warm measurements
    let mut latencies = vec![cold_ms];
    for _ in 0..9 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let cold_warm = ColdWarmLatencyComparison::analyze(&latencies);
    let measurement = ColdStartMeasurement::new("llama.cpp", "7B", cold_ms);

    println!("\nIMP-166d: Real-World Cold Start Measurement:");
    println!("  Cold start: {:.0}ms", cold_ms);
    println!("  Warm average: {:.0}ms", cold_warm.warm_ms);
    println!("  Penalty ratio: {:.2}x", cold_warm.penalty_ratio);
    println!(
        "  QA-016 (7B < 5s): {} (target: {:.0}ms, margin: {:.0}ms)",
        if measurement.meets_target {
            "PASS"
        } else {
            "FAIL"
        },
        measurement.target_ms,
        measurement.margin_ms
    );
}

// =========================================================================
// IMP-167: GPU Utilization Verification (QA-014, EXTREME TDD)
// =========================================================================
// Per spec QA-014: GPU utilization > 70% during inference
// Run with: cargo test test_imp_167 --lib --features bench-http

/// IMP-167a: GPU utilization measurement
#[derive(Debug, Clone)]
pub struct GpuUtilizationMeasurement {
    /// Server name
    pub server_name: String,
    /// Average GPU utilization percentage (0-100)
    pub avg_utilization_percent: f64,
    /// Peak GPU utilization
    pub peak_utilization_percent: f64,
    /// Minimum GPU utilization
    pub min_utilization_percent: f64,
    /// Target utilization (QA-014: 70%)
    pub target_percent: f64,
    /// Whether it meets QA-014
    pub meets_qa014: bool,
    /// Utilization samples
    pub samples: Vec<f64>,
}

impl GpuUtilizationMeasurement {
    pub fn from_samples(server_name: &str, samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                server_name: server_name.to_string(),
                avg_utilization_percent: 0.0,
                peak_utilization_percent: 0.0,
                min_utilization_percent: 0.0,
                target_percent: 70.0,
                meets_qa014: false,
                samples: Vec::new(),
            };
        }

        let avg = samples.iter().sum::<f64>() / samples.len() as f64;
        let peak = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            server_name: server_name.to_string(),
            avg_utilization_percent: avg,
            peak_utilization_percent: peak,
            min_utilization_percent: min,
            target_percent: 70.0,
            meets_qa014: avg >= 70.0,
            samples: samples.to_vec(),
        }
    }

    /// Calculate utilization efficiency (how close to peak we stay)
    pub fn utilization_efficiency(&self) -> f64 {
        if self.peak_utilization_percent > 0.0 {
            (self.avg_utilization_percent / self.peak_utilization_percent) * 100.0
        } else {
            0.0
        }
    }
}

/// IMP-167a: Test GPU utilization measurement
#[test]
fn test_imp_167a_gpu_utilization_measurement() {
    // High utilization (meets QA-014)
    let high_samples = vec![85.0, 90.0, 88.0, 92.0, 87.0, 89.0, 91.0, 86.0, 90.0, 88.0];
    let high = GpuUtilizationMeasurement::from_samples("HighUtilServer", &high_samples);

    assert!(high.meets_qa014, "IMP-167a: 88% avg should meet 70% target");
    assert!(
        high.avg_utilization_percent > 85.0,
        "IMP-167a: Average should be >85%, got {:.1}%",
        high.avg_utilization_percent
    );

    // Low utilization (fails QA-014)
    let low_samples = vec![45.0, 50.0, 48.0, 52.0, 47.0, 49.0, 51.0, 46.0, 50.0, 48.0];
    let low = GpuUtilizationMeasurement::from_samples("LowUtilServer", &low_samples);

    assert!(!low.meets_qa014, "IMP-167a: 48% avg should fail 70% target");

    println!("\nIMP-167a: GPU Utilization Measurement:");
    println!(
        "  High util: avg={:.1}%, peak={:.1}%, min={:.1}%, QA-014={}",
        high.avg_utilization_percent,
        high.peak_utilization_percent,
        high.min_utilization_percent,
        high.meets_qa014
    );
    println!(
        "  Low util: avg={:.1}%, peak={:.1}%, min={:.1}%, QA-014={}",
        low.avg_utilization_percent,
        low.peak_utilization_percent,
        low.min_utilization_percent,
        low.meets_qa014
    );
}

/// IMP-167b: GPU utilization comparison between servers
#[derive(Debug, Clone)]
pub struct GpuUtilizationComparison {
    /// Measurements for each server
    pub measurements: Vec<GpuUtilizationMeasurement>,
    /// Server with highest utilization
    pub most_efficient: String,
    /// Server with lowest utilization
    pub least_efficient: String,
    /// Average utilization across all servers
    pub overall_avg: f64,
}

include!("compare_gpu_utilization.rs");
include!("imp_168d.rs");
include!("imp_170a.rs");
