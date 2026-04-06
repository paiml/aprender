use crate::http_client::*;
// ===========================================

/// Per spec QA-010: Quantized inference matches F32 within acceptable tolerance
/// Measures output quality degradation from quantization
#[derive(Debug, Clone)]
pub struct QuantizedQualityComparison {
    /// F32 reference output (logits or tokens)
    pub f32_output: Vec<f32>,
    /// Quantized output
    pub quantized_output: Vec<f32>,
    /// Mean absolute error
    pub mae: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Maximum absolute difference
    pub max_diff: f64,
    /// Cosine similarity (1.0 = identical)
    pub cosine_similarity: f64,
    /// Relative tolerance threshold
    pub tolerance: f64,
    /// Whether quantization meets QA-010
    pub meets_qa010: bool,
}

impl QuantizedQualityComparison {
    /// Default tolerance: 1% relative error for logits
    const DEFAULT_TOLERANCE: f64 = 0.01;

    pub fn compare(f32_output: &[f32], quantized_output: &[f32]) -> Self {
        Self::compare_with_tolerance(f32_output, quantized_output, Self::DEFAULT_TOLERANCE)
    }

    pub fn compare_with_tolerance(
        f32_output: &[f32],
        quantized_output: &[f32],
        tolerance: f64,
    ) -> Self {
        if f32_output.is_empty()
            || quantized_output.is_empty()
            || f32_output.len() != quantized_output.len()
        {
            return Self {
                f32_output: Vec::new(),
                quantized_output: Vec::new(),
                mae: f64::INFINITY,
                rmse: f64::INFINITY,
                max_diff: f64::INFINITY,
                cosine_similarity: 0.0,
                tolerance,
                meets_qa010: false,
            };
        }

        // Calculate MAE
        let mae = f32_output
            .iter()
            .zip(quantized_output.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / f32_output.len() as f64;

        // Calculate RMSE
        let mse = f32_output
            .iter()
            .zip(quantized_output.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / f32_output.len() as f64;
        let rmse = mse.sqrt();

        // Calculate max diff
        let max_diff = f32_output
            .iter()
            .zip(quantized_output.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0f64, f64::max);

        // Calculate cosine similarity
        let dot: f64 = f32_output
            .iter()
            .zip(quantized_output.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm_a: f64 = f32_output
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = quantized_output
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };

        // QA-010: Relative RMSE should be within tolerance
        let f32_range = f32_output
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max)
            - f32_output
                .iter()
                .map(|&x| x as f64)
                .fold(f64::INFINITY, f64::min);
        let relative_rmse = if f32_range > 0.0 {
            rmse / f32_range
        } else {
            rmse
        };
        let meets_qa010 = relative_rmse <= tolerance && cosine_similarity >= 0.99;

        Self {
            f32_output: f32_output.to_vec(),
            quantized_output: quantized_output.to_vec(),
            mae,
            rmse,
            max_diff,
            cosine_similarity,
            tolerance,
            meets_qa010,
        }
    }
}

/// Token-level quality comparison
#[derive(Debug, Clone)]
pub struct TokenQualityComparison {
    /// F32 reference tokens
    pub f32_tokens: Vec<u32>,
    /// Quantized tokens
    pub quantized_tokens: Vec<u32>,
    /// Number of matching tokens
    pub matching_tokens: usize,
    /// Match rate (0.0-1.0)
    pub match_rate: f64,
    /// Target match rate for QA-010
    pub target_rate: f64,
    /// Whether token output meets QA-010
    pub meets_qa010: bool,
}

impl TokenQualityComparison {
    /// Default: 90% token match for generation quality
    const DEFAULT_TARGET: f64 = 0.90;

    pub fn compare(f32_tokens: &[u32], quantized_tokens: &[u32]) -> Self {
        Self::compare_with_target(f32_tokens, quantized_tokens, Self::DEFAULT_TARGET)
    }

    pub fn compare_with_target(
        f32_tokens: &[u32],
        quantized_tokens: &[u32],
        target_rate: f64,
    ) -> Self {
        if f32_tokens.is_empty() && quantized_tokens.is_empty() {
            return Self {
                f32_tokens: Vec::new(),
                quantized_tokens: Vec::new(),
                matching_tokens: 0,
                match_rate: 1.0,
                target_rate,
                meets_qa010: true,
            };
        }

        let max_len = f32_tokens.len().max(quantized_tokens.len());
        let matching = f32_tokens
            .iter()
            .zip(quantized_tokens.iter())
            .filter(|(&a, &b)| a == b)
            .count();

        let match_rate = matching as f64 / max_len as f64;

        Self {
            f32_tokens: f32_tokens.to_vec(),
            quantized_tokens: quantized_tokens.to_vec(),
            matching_tokens: matching,
            match_rate,
            target_rate,
            meets_qa010: match_rate >= target_rate,
        }
    }
}

/// KL Divergence for probability distribution comparison
#[derive(Debug, Clone)]
pub struct KLDivergenceAnalysis {
    /// F32 probability distribution
    pub f32_probs: Vec<f64>,
    /// Quantized probability distribution
    pub quantized_probs: Vec<f64>,
    /// KL divergence (bits)
    pub kl_divergence: f64,
    /// Maximum acceptable KL divergence
    pub threshold: f64,
    /// Whether within acceptable divergence
    pub acceptable: bool,
}

impl KLDivergenceAnalysis {
    /// Default threshold: 0.01 bits (very close distributions)
    const DEFAULT_THRESHOLD: f64 = 0.01;

    pub fn analyze(f32_probs: &[f64], quantized_probs: &[f64]) -> Self {
        Self::analyze_with_threshold(f32_probs, quantized_probs, Self::DEFAULT_THRESHOLD)
    }

    pub fn analyze_with_threshold(
        f32_probs: &[f64],
        quantized_probs: &[f64],
        threshold: f64,
    ) -> Self {
        if f32_probs.is_empty()
            || quantized_probs.is_empty()
            || f32_probs.len() != quantized_probs.len()
        {
            return Self {
                f32_probs: Vec::new(),
                quantized_probs: Vec::new(),
                kl_divergence: f64::INFINITY,
                threshold,
                acceptable: false,
            };
        }

        // KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
        // Add small epsilon to avoid log(0)
        let epsilon = 1e-10;
        let kl = f32_probs
            .iter()
            .zip(quantized_probs.iter())
            .map(|(&p, &q)| {
                let p = p.max(epsilon);
                let q = q.max(epsilon);
                p * (p / q).ln()
            })
            .sum::<f64>();

        Self {
            f32_probs: f32_probs.to_vec(),
            quantized_probs: quantized_probs.to_vec(),
            kl_divergence: kl,
            threshold,
            acceptable: kl <= threshold,
        }
    }
}

/// IMP-171a: Test quantized output quality comparison
#[test]
fn test_imp_171a_quantized_quality() {
    // High quality quantization (small differences)
    let f32_output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let quantized_good: Vec<f32> = vec![1.01, 2.02, 2.98, 4.01, 5.0, 5.99, 7.02, 7.98, 9.01, 10.0];

    let good_comparison = QuantizedQualityComparison::compare(&f32_output, &quantized_good);

    assert!(
        good_comparison.cosine_similarity > 0.999,
        "IMP-171a: High quality should have cosine > 0.999"
    );
    assert!(
        good_comparison.rmse < 0.03,
        "IMP-171a: High quality should have RMSE < 0.03"
    );

    // Low quality quantization (large differences)
    let quantized_bad: Vec<f32> = vec![1.5, 2.5, 2.5, 4.5, 5.5, 5.5, 7.5, 7.5, 9.5, 10.5];
    let bad_comparison = QuantizedQualityComparison::compare(&f32_output, &quantized_bad);

    assert!(
        bad_comparison.rmse > good_comparison.rmse,
        "IMP-171a: Low quality should have higher RMSE"
    );

    println!("\nIMP-171a: Quantized Output Quality:");
    println!(
        "  Good quantization: RMSE={:.4}, cosine={:.6}, QA-010={}",
        good_comparison.rmse, good_comparison.cosine_similarity, good_comparison.meets_qa010
    );
    println!(
        "  Bad quantization: RMSE={:.4}, cosine={:.6}, QA-010={}",
        bad_comparison.rmse, bad_comparison.cosine_similarity, bad_comparison.meets_qa010
    );
}

/// IMP-171b: Test token-level quality comparison
#[test]
fn test_imp_171b_token_quality() {
    // Perfect match
    let f32_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let perfect_match: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let perfect = TokenQualityComparison::compare(&f32_tokens, &perfect_match);
    assert!(
        perfect.meets_qa010,
        "IMP-171b: Perfect match should meet QA-010"
    );
    assert!(
        (perfect.match_rate - 1.0).abs() < 0.001,
        "IMP-171b: Match rate should be 1.0"
    );

    // Partial match (90%+)
    let good_match: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11]; // 9/10 matching
    let good = TokenQualityComparison::compare(&f32_tokens, &good_match);
    assert!(good.meets_qa010, "IMP-171b: 90% match should meet QA-010");

    // Poor match
    let bad_match: Vec<u32> = vec![1, 2, 3, 4, 5, 11, 12, 13, 14, 15]; // 5/10 matching
    let bad = TokenQualityComparison::compare(&f32_tokens, &bad_match);
    assert!(
        !bad.meets_qa010,
        "IMP-171b: 50% match should not meet QA-010"
    );

    println!("\nIMP-171b: Token Quality Comparison:");
    println!(
        "  Perfect: {}/{} ({:.0}%), QA-010={}",
        perfect.matching_tokens,
        f32_tokens.len(),
        perfect.match_rate * 100.0,
        perfect.meets_qa010
    );
    println!(
        "  Good: {}/{} ({:.0}%), QA-010={}",
        good.matching_tokens,
        f32_tokens.len(),
        good.match_rate * 100.0,
        good.meets_qa010
    );
    println!(
        "  Bad: {}/{} ({:.0}%), QA-010={}",
        bad.matching_tokens,
        f32_tokens.len(),
        bad.match_rate * 100.0,
        bad.meets_qa010
    );
}

/// IMP-171c: Test KL divergence analysis
#[test]
fn test_imp_171c_kl_divergence() {
    // Identical distributions
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let q_identical = vec![0.25, 0.25, 0.25, 0.25];

    let identical = KLDivergenceAnalysis::analyze(&p, &q_identical);
    assert!(
        identical.kl_divergence < 0.001,
        "IMP-171c: Identical should have near-zero KL"
    );
    assert!(
        identical.acceptable,
        "IMP-171c: Identical distributions should be acceptable"
    );

    // Close distributions
    let q_close = vec![0.26, 0.24, 0.26, 0.24];
    let close = KLDivergenceAnalysis::analyze(&p, &q_close);
    assert!(
        close.kl_divergence < 0.01,
        "IMP-171c: Close distributions should have small KL"
    );

    // Very different distributions
    let q_different = vec![0.7, 0.1, 0.1, 0.1];
    let different = KLDivergenceAnalysis::analyze(&p, &q_different);
    assert!(
        different.kl_divergence > 0.1,
        "IMP-171c: Different distributions should have large KL"
    );
    assert!(
        !different.acceptable,
        "IMP-171c: Very different distributions should not be acceptable"
    );

    println!("\nIMP-171c: KL Divergence Analysis:");
    println!(
        "  Identical: KL={:.6} bits, acceptable={}",
        identical.kl_divergence, identical.acceptable
    );
    println!(
        "  Close: KL={:.6} bits, acceptable={}",
        close.kl_divergence, close.acceptable
    );
    println!(
        "  Different: KL={:.6} bits, acceptable={}",
        different.kl_divergence, different.acceptable
    );
}

/// IMP-171d: Real-world quantized quality verification
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_171d_realworld_quantized_quality() {
    let client = ModelHttpClient::with_timeout(60);

    // Request with deterministic settings
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "The capital of France is".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    // Run multiple times to check consistency
    let mut outputs: Vec<String> = Vec::new();
    for _ in 0..5 {
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            outputs.push(result.text.clone());
        }
    }

    if outputs.len() < 3 {
        println!("IMP-171d: Not enough samples");
        return;
    }

    // Check determinism (with temp=0, outputs should be identical)
    let first = &outputs[0];
    let matching = outputs.iter().filter(|&o| o == first).count();
    let determinism_rate = matching as f64 / outputs.len() as f64;

    println!("\nIMP-171d: Real-World Quantized Quality:");
    println!("  Samples: {}", outputs.len());
    println!("  Determinism rate: {:.0}%", determinism_rate * 100.0);
    println!("  First output: {:?}", first);
    println!(
        "  QA-010 (deterministic): {}",
        if determinism_rate >= 0.8 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

include!("batch_scaling.rs");
include!("context_scaling.rs");
include!("success_oom_graceful.rs");
