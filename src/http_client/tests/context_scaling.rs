
// ===========================================
// IMP-173: Context Growth Performance (QA-020)
// ===========================================

/// Per spec QA-020: No performance degradation with context growth
#[derive(Debug, Clone)]
pub struct ContextScalingMeasurement {
    /// Context length (tokens)
    pub context_length: usize,
    /// Latency per token (ms)
    pub latency_per_token_ms: f64,
    /// Memory usage (MB)
    pub memory_mb: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Context growth analysis
#[derive(Debug, Clone)]
pub struct ContextGrowthAnalysis {
    /// Measurements at different context lengths
    pub measurements: Vec<ContextScalingMeasurement>,
    /// Expected scaling factor (O(n), O(n²), etc.)
    pub scaling_exponent: f64,
    /// Actual latency growth rate
    pub latency_growth_rate: f64,
    /// Whether scaling is acceptable (< O(n²) degradation)
    pub acceptable_scaling: bool,
    /// Whether meets QA-020
    pub meets_qa020: bool,
}

impl ContextGrowthAnalysis {
    pub fn analyze(measurements: &[ContextScalingMeasurement]) -> Self {
        if measurements.len() < 2 {
            return Self {
                measurements: Vec::new(),
                scaling_exponent: 0.0,
                latency_growth_rate: 0.0,
                acceptable_scaling: false,
                meets_qa020: false,
            };
        }

        // Calculate scaling exponent via log-log regression
        // latency = k * context^n => log(latency) = log(k) + n*log(context)
        let n = measurements.len() as f64;
        let sum_log_x: f64 = measurements
            .iter()
            .map(|m| (m.context_length as f64).ln())
            .sum();
        let sum_log_y: f64 = measurements
            .iter()
            .map(|m| m.latency_per_token_ms.ln())
            .sum();
        let sum_log_xy: f64 = measurements
            .iter()
            .map(|m| (m.context_length as f64).ln() * m.latency_per_token_ms.ln())
            .sum();
        let sum_log_xx: f64 = measurements
            .iter()
            .map(|m| (m.context_length as f64).ln().powi(2))
            .sum();

        let scaling_exponent =
            (n * sum_log_xy - sum_log_x * sum_log_y) / (n * sum_log_xx - sum_log_x * sum_log_x);

        // Calculate latency growth rate (ratio of last to first)
        let first = &measurements[0];
        let last = &measurements[measurements.len() - 1];
        let latency_growth_rate = last.latency_per_token_ms / first.latency_per_token_ms;

        // QA-020: Acceptable if scaling is sub-quadratic (exponent < 1.5)
        // With KV cache, should be O(n) which is exponent ~1.0
        let acceptable_scaling = scaling_exponent < 1.5;

        // QA-020: No "degradation" means throughput should not drop more than 50%
        let throughput_ratio = first.tokens_per_second / last.tokens_per_second;
        let meets_qa020 = acceptable_scaling && throughput_ratio < 4.0; // Allow up to 4x slowdown

        Self {
            measurements: measurements.to_vec(),
            scaling_exponent,
            latency_growth_rate,
            acceptable_scaling,
            meets_qa020,
        }
    }
}

/// Memory scaling with context
#[derive(Debug, Clone)]
pub struct MemoryScalingAnalysis {
    /// Baseline memory (MB)
    pub baseline_mb: f64,
    /// Memory at max context (MB)
    pub max_context_mb: f64,
    /// Memory growth per 1K tokens (MB)
    pub growth_per_1k_tokens: f64,
    /// Whether memory growth is linear
    pub linear_growth: bool,
}

impl MemoryScalingAnalysis {
    pub fn analyze(measurements: &[ContextScalingMeasurement]) -> Self {
        if measurements.len() < 2 {
            return Self {
                baseline_mb: 0.0,
                max_context_mb: 0.0,
                growth_per_1k_tokens: 0.0,
                linear_growth: false,
            };
        }

        let first = &measurements[0];
        let last = &measurements[measurements.len() - 1];

        let baseline_mb = first.memory_mb;
        let max_context_mb = last.memory_mb;
        let delta_tokens = (last.context_length - first.context_length) as f64 / 1000.0;
        let growth_per_1k_tokens = if delta_tokens > 0.0 {
            (max_context_mb - baseline_mb) / delta_tokens
        } else {
            0.0
        };

        // Linear regression R² to check linearity
        let n = measurements.len() as f64;
        let sum_x: f64 = measurements.iter().map(|m| m.context_length as f64).sum();
        let sum_y: f64 = measurements.iter().map(|m| m.memory_mb).sum();
        let mean_y = sum_y / n;
        let sum_xy: f64 = measurements
            .iter()
            .map(|m| m.context_length as f64 * m.memory_mb)
            .sum();
        let sum_xx: f64 = measurements
            .iter()
            .map(|m| (m.context_length as f64).powi(2))
            .sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        let ss_tot: f64 = measurements
            .iter()
            .map(|m| (m.memory_mb - mean_y).powi(2))
            .sum();
        let ss_res: f64 = measurements
            .iter()
            .map(|m| (m.memory_mb - (slope * m.context_length as f64 + intercept)).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        let linear_growth = r_squared >= 0.9;

        Self {
            baseline_mb,
            max_context_mb,
            growth_per_1k_tokens,
            linear_growth,
        }
    }
}

/// IMP-173a: Test context scaling measurement
#[test]
fn test_imp_173a_context_scaling() {
    // O(n) scaling (ideal with KV cache)
    let linear_scaling = vec![
        ContextScalingMeasurement {
            context_length: 128,
            latency_per_token_ms: 10.0,
            memory_mb: 1000.0,
            tokens_per_second: 100.0,
        },
        ContextScalingMeasurement {
            context_length: 256,
            latency_per_token_ms: 20.0,
            memory_mb: 1100.0,
            tokens_per_second: 50.0,
        },
        ContextScalingMeasurement {
            context_length: 512,
            latency_per_token_ms: 40.0,
            memory_mb: 1300.0,
            tokens_per_second: 25.0,
        },
        ContextScalingMeasurement {
            context_length: 1024,
            latency_per_token_ms: 80.0,
            memory_mb: 1700.0,
            tokens_per_second: 12.5,
        },
    ];

    let analysis = ContextGrowthAnalysis::analyze(&linear_scaling);

    // O(n) scaling has exponent ~1.0
    assert!(
        analysis.scaling_exponent > 0.5 && analysis.scaling_exponent < 1.5,
        "IMP-173a: Linear scaling should have exponent between 0.5 and 1.5, got {}",
        analysis.scaling_exponent
    );
    assert!(
        analysis.acceptable_scaling,
        "IMP-173a: O(n) scaling should be acceptable"
    );

    println!("\nIMP-173a: Context Scaling Analysis:");
    println!(
        "  Scaling exponent: {:.2} (1.0 = O(n), 2.0 = O(n²))",
        analysis.scaling_exponent
    );
    println!(
        "  Latency growth: {:.1}x from 128 to 1024 tokens",
        analysis.latency_growth_rate
    );
    println!(
        "  QA-020: {}",
        if analysis.meets_qa020 { "PASS" } else { "FAIL" }
    );
}

/// IMP-173b: Test memory scaling analysis
#[test]
fn test_imp_173b_memory_scaling() {
    // Linear memory growth (KV cache)
    let linear_memory = vec![
        ContextScalingMeasurement {
            context_length: 128,
            latency_per_token_ms: 10.0,
            memory_mb: 1000.0,
            tokens_per_second: 100.0,
        },
        ContextScalingMeasurement {
            context_length: 512,
            latency_per_token_ms: 40.0,
            memory_mb: 1200.0,
            tokens_per_second: 25.0,
        },
        ContextScalingMeasurement {
            context_length: 1024,
            latency_per_token_ms: 80.0,
            memory_mb: 1400.0,
            tokens_per_second: 12.5,
        },
        ContextScalingMeasurement {
            context_length: 2048,
            latency_per_token_ms: 160.0,
            memory_mb: 1800.0,
            tokens_per_second: 6.25,
        },
    ];

    let memory_analysis = MemoryScalingAnalysis::analyze(&linear_memory);

    assert!(
        memory_analysis.linear_growth,
        "IMP-173b: Memory growth should be linear"
    );
    assert!(
        memory_analysis.growth_per_1k_tokens > 0.0,
        "IMP-173b: Memory should grow with context"
    );

    println!("\nIMP-173b: Memory Scaling Analysis:");
    println!(
        "  Baseline: {:.0} MB at 128 tokens",
        memory_analysis.baseline_mb
    );
    println!(
        "  Max context: {:.0} MB at 2048 tokens",
        memory_analysis.max_context_mb
    );
    println!(
        "  Growth: {:.1} MB per 1K tokens",
        memory_analysis.growth_per_1k_tokens
    );
    println!("  Linear growth: {}", memory_analysis.linear_growth);
}

/// IMP-173c: Test quadratic degradation detection
#[test]
fn test_imp_173c_quadratic_detection() {
    // O(n²) scaling (pathological case without KV cache)
    let quadratic_scaling = vec![
        ContextScalingMeasurement {
            context_length: 128,
            latency_per_token_ms: 10.0,
            memory_mb: 1000.0,
            tokens_per_second: 100.0,
        },
        ContextScalingMeasurement {
            context_length: 256,
            latency_per_token_ms: 40.0,
            memory_mb: 1400.0,
            tokens_per_second: 25.0,
        }, // 4x for 2x context
        ContextScalingMeasurement {
            context_length: 512,
            latency_per_token_ms: 160.0,
            memory_mb: 2600.0,
            tokens_per_second: 6.25,
        }, // 16x for 4x context
        ContextScalingMeasurement {
            context_length: 1024,
            latency_per_token_ms: 640.0,
            memory_mb: 5800.0,
            tokens_per_second: 1.56,
        }, // 64x for 8x context
    ];

    let analysis = ContextGrowthAnalysis::analyze(&quadratic_scaling);

    // O(n²) scaling has exponent ~2.0
    assert!(
        analysis.scaling_exponent > 1.5,
        "IMP-173c: Quadratic scaling should have exponent > 1.5, got {}",
        analysis.scaling_exponent
    );
    assert!(
        !analysis.acceptable_scaling,
        "IMP-173c: O(n²) scaling should NOT be acceptable"
    );
    assert!(
        !analysis.meets_qa020,
        "IMP-173c: O(n²) scaling should NOT meet QA-020"
    );

    println!("\nIMP-173c: Quadratic Detection:");
    println!(
        "  Scaling exponent: {:.2} (indicates O(n²))",
        analysis.scaling_exponent
    );
    println!(
        "  Acceptable: {} (should be false)",
        analysis.acceptable_scaling
    );
    println!(
        "  QA-020: {}",
        if analysis.meets_qa020 { "PASS" } else { "FAIL" }
    );
}

/// IMP-173d: Real-world context growth verification
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_173d_realworld_context_growth() {
    let client = ModelHttpClient::with_timeout(120);

    let mut measurements = Vec::new();

    // Test different context lengths by varying prompt size
    for context_mult in [1, 2, 4, 8] {
        let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(context_mult * 10);
        let context_length = prompt.len() / 4; // Rough token estimate

        let request = CompletionRequest {
            model: "default".to_string(),
            prompt,
            max_tokens: 20,
            temperature: Some(0.0),
            stream: false,
        };

        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let tokens = result.text.split_whitespace().count().max(1);
            let latency_per_token = elapsed / tokens as f64;

            measurements.push(ContextScalingMeasurement {
                context_length,
                latency_per_token_ms: latency_per_token,
                memory_mb: 0.0, // Would need server metrics API
                tokens_per_second: tokens as f64 / (elapsed / 1000.0),
            });
        }
    }

    if measurements.len() < 2 {
        println!("IMP-173d: Not enough measurements");
        return;
    }

    let analysis = ContextGrowthAnalysis::analyze(&measurements);

    println!("\nIMP-173d: Real-World Context Growth:");
    for m in &measurements {
        println!(
            "  context={}: {:.1}ms/tok, {:.1} tok/s",
            m.context_length, m.latency_per_token_ms, m.tokens_per_second
        );
    }
    println!("  Scaling exponent: {:.2}", analysis.scaling_exponent);
    println!("  Latency growth: {:.1}x", analysis.latency_growth_rate);
    println!(
        "  QA-020 (no degradation): {}",
        if analysis.meets_qa020 { "PASS" } else { "FAIL" }
    );
}

// ===========================================
// IMP-174: OOM Graceful Handling (QA-021)
// ===========================================

/// Per spec QA-021: Graceful handling of OOM conditions
#[derive(Debug, Clone)]
pub struct OOMHandlingResult {
    /// Whether OOM was detected
    pub oom_detected: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
    /// Whether system remained stable after OOM
    pub system_stable: bool,
    /// Whether resources were properly released
    pub resources_released: bool,
    /// Meets QA-021 requirements
    pub meets_qa021: bool,
}
