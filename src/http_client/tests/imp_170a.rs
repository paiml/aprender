
/// IMP-170a: Test token rate stability
#[test]
fn test_imp_170a_token_rate_stability() {
    // Stable rates (CV < 10%)
    let stable_rates = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 100.0, 102.0, 98.0,
    ];
    let stable = TokenRateStability::analyze(&stable_rates);

    assert!(stable.meets_qa019, "IMP-170a: Low CV should meet QA-019");
    assert!(
        stable.cv < 0.05,
        "IMP-170a: CV should be <5%, got {:.2}%",
        stable.cv * 100.0
    );

    // Unstable rates (CV > 10%)
    let unstable_rates = vec![
        80.0, 120.0, 70.0, 130.0, 75.0, 125.0, 85.0, 115.0, 90.0, 110.0,
    ];
    let unstable = TokenRateStability::analyze(&unstable_rates);

    assert!(
        !unstable.meets_qa019,
        "IMP-170a: High CV should fail QA-019"
    );

    println!("\nIMP-170a: Token Rate Stability:");
    println!(
        "  Stable: mean={:.1} tok/s, stddev={:.2}, CV={:.2}%, QA-019={}",
        stable.mean_rate,
        stable.stddev_rate,
        stable.cv * 100.0,
        stable.meets_qa019
    );
    println!(
        "  Unstable: mean={:.1} tok/s, stddev={:.2}, CV={:.2}%, QA-019={}",
        unstable.mean_rate,
        unstable.stddev_rate,
        unstable.cv * 100.0,
        unstable.meets_qa019
    );
}

/// IMP-170b: Inter-token latency (ITL) analysis
#[derive(Debug, Clone)]
pub struct InterTokenLatencyAnalysis {
    /// ITL samples (ms between tokens)
    pub itl_samples: Vec<f64>,
    /// Mean ITL
    pub mean_itl_ms: f64,
    /// ITL variance
    pub itl_variance: f64,
    /// ITL jitter (stddev)
    pub itl_jitter_ms: f64,
    /// Whether jitter is acceptable (< 20% of mean)
    pub jitter_acceptable: bool,
}

impl InterTokenLatencyAnalysis {
    pub fn analyze(itl_samples: &[f64]) -> Self {
        if itl_samples.is_empty() {
            return Self {
                itl_samples: Vec::new(),
                mean_itl_ms: 0.0,
                itl_variance: 0.0,
                itl_jitter_ms: 0.0,
                jitter_acceptable: true,
            };
        }

        let n = itl_samples.len();
        let mean = itl_samples.iter().sum::<f64>() / n as f64;
        let variance = itl_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let jitter = variance.sqrt();

        // Jitter acceptable if < 20% of mean
        let acceptable = mean > 0.0 && (jitter / mean) < 0.20;

        Self {
            itl_samples: itl_samples.to_vec(),
            mean_itl_ms: mean,
            itl_variance: variance,
            itl_jitter_ms: jitter,
            jitter_acceptable: acceptable,
        }
    }
}

/// IMP-170b: Test ITL analysis
#[test]
fn test_imp_170b_itl_analysis() {
    // Stable ITL
    let stable_itl = vec![10.0, 10.5, 9.8, 10.2, 9.9, 10.1, 10.3, 9.7, 10.0, 10.4];
    let stable = InterTokenLatencyAnalysis::analyze(&stable_itl);

    assert!(
        stable.jitter_acceptable,
        "IMP-170b: Low jitter should be acceptable"
    );

    // High jitter ITL
    let jittery_itl = vec![5.0, 15.0, 8.0, 12.0, 6.0, 14.0, 7.0, 13.0, 9.0, 11.0];
    let jittery = InterTokenLatencyAnalysis::analyze(&jittery_itl);

    assert!(
        !jittery.jitter_acceptable,
        "IMP-170b: High jitter should not be acceptable"
    );

    println!("\nIMP-170b: Inter-Token Latency Analysis:");
    println!(
        "  Stable: mean={:.2}ms, jitter={:.2}ms, acceptable={}",
        stable.mean_itl_ms, stable.itl_jitter_ms, stable.jitter_acceptable
    );
    println!(
        "  Jittery: mean={:.2}ms, jitter={:.2}ms, acceptable={}",
        jittery.mean_itl_ms, jittery.itl_jitter_ms, jittery.jitter_acceptable
    );
}

/// IMP-170c: Generation consistency check
#[derive(Debug, Clone)]
pub struct GenerationConsistency {
    /// Number of generations
    pub generation_count: usize,
    /// Generations with rate within 10% of mean
    pub consistent_count: usize,
    /// Consistency percentage
    pub consistency_percent: f64,
    /// Whether 95%+ generations are consistent
    pub highly_consistent: bool,
}

impl GenerationConsistency {
    pub fn analyze(rates: &[f64]) -> Self {
        if rates.is_empty() {
            return Self {
                generation_count: 0,
                consistent_count: 0,
                consistency_percent: 100.0,
                highly_consistent: true,
            };
        }

        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let consistent = rates
            .iter()
            .filter(|&&r| ((r - mean) / mean).abs() < 0.10)
            .count();

        let consistency = (consistent as f64 / rates.len() as f64) * 100.0;

        Self {
            generation_count: rates.len(),
            consistent_count: consistent,
            consistency_percent: consistency,
            highly_consistent: consistency >= 95.0,
        }
    }
}

/// IMP-170c: Test generation consistency
#[test]
fn test_imp_170c_generation_consistency() {
    // Highly consistent
    let consistent = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0,
    ];
    let consistent_analysis = GenerationConsistency::analyze(&consistent);

    assert!(
        consistent_analysis.highly_consistent,
        "IMP-170c: All within 10% should be highly consistent"
    );
    assert_eq!(consistent_analysis.consistent_count, 10);

    // Inconsistent (some outliers)
    let inconsistent = vec![
        100.0, 102.0, 50.0, 101.0, 150.0, 100.0, 103.0, 97.0, 100.0, 101.0,
    ];
    let inconsistent_analysis = GenerationConsistency::analyze(&inconsistent);

    assert!(
        !inconsistent_analysis.highly_consistent,
        "IMP-170c: With outliers should not be highly consistent"
    );

    println!("\nIMP-170c: Generation Consistency:");
    println!(
        "  Consistent: {}/{} ({:.1}%), highly_consistent={}",
        consistent_analysis.consistent_count,
        consistent_analysis.generation_count,
        consistent_analysis.consistency_percent,
        consistent_analysis.highly_consistent
    );
    println!(
        "  Inconsistent: {}/{} ({:.1}%), highly_consistent={}",
        inconsistent_analysis.consistent_count,
        inconsistent_analysis.generation_count,
        inconsistent_analysis.consistency_percent,
        inconsistent_analysis.highly_consistent
    );
}

/// IMP-170d: Real-world token rate stability
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_170d_realworld_token_rate_stability() {
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count from 1 to 20:".to_string(),
        max_tokens: 30,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect token rates from multiple generations
    let mut rates = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens = result.text.split_whitespace().count().max(1);
            rates.push(tokens as f64 / elapsed);
        }
    }

    if rates.len() < 5 {
        println!("IMP-170d: Not enough samples");
        return;
    }

    let stability = TokenRateStability::analyze(&rates);
    let consistency = GenerationConsistency::analyze(&rates);

    println!("\nIMP-170d: Real-World Token Rate Stability:");
    println!("  Samples: {}", rates.len());
    println!("  Mean rate: {:.1} tok/s", stability.mean_rate);
    println!("  CV: {:.2}%", stability.cv * 100.0);
    println!(
        "  QA-019 (CV < 10%): {}",
        if stability.meets_qa019 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("  Consistency: {:.1}%", consistency.consistency_percent);
}

// ===========================================
// IMP-171: Quantized vs F32 Quality Verification (QA-010)
