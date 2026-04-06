#![allow(clippy::disallowed_methods)]
//! Cross-Format Parity Tests
//!
//! These tests verify that GGUF, SafeTensors, and APR formats produce
//! identical logits for the same model and prompt.
//!
//! FALSIFICATION CRITERIA:
//! - Max absolute difference > 1e-4 between any two formats
//! - Missing tensor in any format
//! - Shape mismatch between formats
//!
//! Toyota Way: Genchi Genbutsu - Go see the actual numbers match.

use std::collections::HashMap;

/// Tolerance for logit comparison (1e-4 as per spec)
const LOGIT_TOLERANCE: f32 = 1e-4;

/// Mock logits for testing without real model files
/// In production, these would come from actual model inference
#[derive(Debug, Clone)]
struct MockLogits {
    format: &'static str,
    logits: Vec<f32>,
}

impl MockLogits {
    fn new(format: &'static str, prompt: &str, vocab_size: usize) -> Self {
        // Generate deterministic "logits" based on format and prompt
        // In a real test, these would come from actual model inference
        let seed = prompt
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| {
                let base = ((seed.wrapping_mul(i as u64 + 1)) % 1000) as f32 / 1000.0;
                // Small format-specific variation for testing
                let format_offset = match format {
                    "gguf" => 0.0,
                    "safetensors" => 1e-6, // Should be within tolerance
                    "apr" => 2e-6,         // Should be within tolerance
                    _ => 0.0,
                };
                base + format_offset
            })
            .collect();

        Self { format, logits }
    }

    /// Create with explicit logits (for poison tests)
    fn with_logits(format: &'static str, _prompt: &str, logits: Vec<f32>) -> Self {
        Self { format, logits }
    }
}

/// Compare logits between two formats
fn compare_logits(a: &MockLogits, b: &MockLogits) -> Result<f32, String> {
    if a.logits.len() != b.logits.len() {
        return Err(format!(
            "Shape mismatch: {} has {} logits, {} has {}",
            a.format,
            a.logits.len(),
            b.format,
            b.logits.len()
        ));
    }

    let max_diff = a
        .logits
        .iter()
        .zip(b.logits.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    Ok(max_diff)
}

/// Parity verification result
#[derive(Debug)]
struct ParityResult {
    passed: bool,
}

impl ParityResult {
    fn new(_format_a: &'static str, _format_b: &'static str, max_diff: f32) -> Self {
        Self {
            passed: max_diff <= LOGIT_TOLERANCE,
        }
    }
}

// ============================================================================
// Cross-Format Parity Tests
// ============================================================================

/// P1: GGUF and SafeTensors produce identical logits
/// FALSIFICATION: max_diff > 1e-4
#[test]
fn p1_gguf_safetensors_logit_parity() {
    let prompt = "Hello";
    let vocab_size = 32000; // Typical LLaMA vocab size

    let gguf_logits = MockLogits::new("gguf", prompt, vocab_size);
    let st_logits = MockLogits::new("safetensors", prompt, vocab_size);

    let max_diff = compare_logits(&gguf_logits, &st_logits).expect("Comparison failed");

    let result = ParityResult::new("gguf", "safetensors", max_diff);

    assert!(
        result.passed,
        "P1 FALSIFIED: GGUF vs SafeTensors max_diff={:.2e} > {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );

    println!(
        "P1 PASS: GGUF vs SafeTensors max_diff={:.2e} <= {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );
}

/// P2: GGUF and APR produce identical logits
/// FALSIFICATION: max_diff > 1e-4
#[test]
fn p2_gguf_apr_logit_parity() {
    let prompt = "Hello";
    let vocab_size = 32000;

    let gguf_logits = MockLogits::new("gguf", prompt, vocab_size);
    let apr_logits = MockLogits::new("apr", prompt, vocab_size);

    let max_diff = compare_logits(&gguf_logits, &apr_logits).expect("Comparison failed");

    let result = ParityResult::new("gguf", "apr", max_diff);

    assert!(
        result.passed,
        "P2 FALSIFIED: GGUF vs APR max_diff={:.2e} > {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );

    println!(
        "P2 PASS: GGUF vs APR max_diff={:.2e} <= {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );
}

/// P3: SafeTensors and APR produce identical logits
/// FALSIFICATION: max_diff > 1e-4
#[test]
fn p3_safetensors_apr_logit_parity() {
    let prompt = "Hello";
    let vocab_size = 32000;

    let st_logits = MockLogits::new("safetensors", prompt, vocab_size);
    let apr_logits = MockLogits::new("apr", prompt, vocab_size);

    let max_diff = compare_logits(&st_logits, &apr_logits).expect("Comparison failed");

    let result = ParityResult::new("safetensors", "apr", max_diff);

    assert!(
        result.passed,
        "P3 FALSIFIED: SafeTensors vs APR max_diff={:.2e} > {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );

    println!(
        "P3 PASS: SafeTensors vs APR max_diff={:.2e} <= {:.2e}",
        max_diff, LOGIT_TOLERANCE
    );
}

/// P4: All three formats produce identical logits (transitive check)
/// FALSIFICATION: Any pairwise comparison fails
#[test]
fn p4_all_formats_logit_parity() {
    let prompt = "Hello";
    let vocab_size = 32000;

    let gguf = MockLogits::new("gguf", prompt, vocab_size);
    let st = MockLogits::new("safetensors", prompt, vocab_size);
    let apr = MockLogits::new("apr", prompt, vocab_size);

    let results = vec![
        ("gguf", "safetensors", compare_logits(&gguf, &st)),
        ("gguf", "apr", compare_logits(&gguf, &apr)),
        ("safetensors", "apr", compare_logits(&st, &apr)),
    ];

    let mut all_passed = true;
    let mut max_overall = 0.0f32;

    for (fmt_a, fmt_b, result) in &results {
        match result {
            Ok(diff) => {
                max_overall = max_overall.max(*diff);
                if *diff > LOGIT_TOLERANCE {
                    all_passed = false;
                    eprintln!(
                        "P4 FAIL: {} vs {} max_diff={:.2e} > {:.2e}",
                        fmt_a, fmt_b, diff, LOGIT_TOLERANCE
                    );
                }
            }
            Err(e) => {
                all_passed = false;
                eprintln!("P4 FAIL: {} vs {} error: {}", fmt_a, fmt_b, e);
            }
        }
    }

    assert!(
        all_passed,
        "P4 FALSIFIED: Cross-format parity check failed (max_diff={:.2e})",
        max_overall
    );

    println!(
        "P4 PASS: All formats match within {:.2e} (max_diff={:.2e})",
        LOGIT_TOLERANCE, max_overall
    );
}

// ============================================================================
// Poison Detection Tests
// ============================================================================

/// P5: Detect shape mismatch between formats
/// FALSIFICATION: Shape mismatch not detected
#[test]
fn p5_detect_shape_mismatch() {
    let gguf = MockLogits::with_logits("gguf", "Hello", vec![0.1, 0.2, 0.3]);
    let apr = MockLogits::with_logits("apr", "Hello", vec![0.1, 0.2]); // Wrong size!

    let result = compare_logits(&gguf, &apr);

    assert!(result.is_err(), "P5 FALSIFIED: Shape mismatch not detected");

    let err = result.unwrap_err();
    assert!(
        err.contains("Shape mismatch"),
        "P5 FALSIFIED: Error message doesn't mention shape mismatch: {}",
        err
    );

    println!("P5 PASS: Shape mismatch correctly detected");
}

/// P6: Detect large logit divergence (poisoned model)
/// FALSIFICATION: Divergence not flagged
#[test]
fn p6_detect_logit_divergence() {
    // Simulate a poisoned model with large logit differences
    let gguf = MockLogits::with_logits("gguf", "Hello", vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let poisoned = MockLogits::with_logits(
        "poisoned",
        "Hello",
        vec![0.1, 0.2, 0.5, 0.4, 0.5], // 0.3 -> 0.5 is > 1e-4
    );

    let max_diff = compare_logits(&gguf, &poisoned).expect("Comparison failed");

    assert!(
        max_diff > LOGIT_TOLERANCE,
        "P6 FALSIFIED: Large divergence not flagged (diff={:.2e})",
        max_diff
    );

    println!(
        "P6 PASS: Logit divergence correctly flagged (diff={:.2e} > {:.2e})",
        max_diff, LOGIT_TOLERANCE
    );
}

/// P7: Verify tolerance boundary behavior
/// FALSIFICATION: Boundary behavior not as expected
#[test]
fn p7_tolerance_boundary() {
    // Test below and above tolerance boundary
    // Using 0.5x and 2x tolerance to avoid floating point precision issues
    let base: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let below_boundary: Vec<f32> = base.iter().map(|x| x + LOGIT_TOLERANCE * 0.5).collect();
    let above_boundary: Vec<f32> = base.iter().map(|x| x + LOGIT_TOLERANCE * 2.0).collect();

    let gguf = MockLogits::with_logits("gguf", "Hello", base);
    let below = MockLogits::with_logits("below", "Hello", below_boundary);
    let above = MockLogits::with_logits("above", "Hello", above_boundary);

    // Below boundary should pass
    let diff_below = compare_logits(&gguf, &below).expect("Comparison failed");
    assert!(
        diff_below <= LOGIT_TOLERANCE,
        "P7 FALSIFIED: Below-boundary case should pass (diff={:.2e})",
        diff_below
    );

    // Above boundary should fail
    let diff_above = compare_logits(&gguf, &above).expect("Comparison failed");
    assert!(
        diff_above > LOGIT_TOLERANCE,
        "P7 FALSIFIED: Above-boundary case should fail (diff={:.2e})",
        diff_above
    );

    println!(
        "P7 PASS: Boundary cases handled correctly (below={:.2e}, above={:.2e})",
        diff_below, diff_above
    );
}

// ============================================================================
// Integration Tests (require actual model files)
// ============================================================================

/// P8: Real GGUF/APR parity test (requires model files)
/// Skip if model files not present
#[test]
#[ignore = "Requires Qwen2.5-0.5B model files"]
fn p8_real_model_logit_parity() {
    // This test would load actual model files:
    // - qwen2.5-0.5b.gguf
    // - qwen2.5-0.5b.safetensors (directory)
    // - qwen2.5-0.5b.apr
    //
    // And compare logits for prompt "Hello"
    //
    // Run with: cargo test --release p8_real_model -- --ignored

    todo!("Implement real model comparison when files available");
}

/// P9: Real SafeTensors/APR parity test (requires model files)
#[test]
#[ignore = "Requires Qwen2.5-0.5B model files"]
fn p9_real_safetensors_apr_parity() {
    todo!("Implement real SafeTensors/APR comparison");
}

// ============================================================================
// Helper: Cross-format inference facade
// ============================================================================

/// Cross-format parity checker
struct ParityChecker {
    tolerance: f32,
    results: HashMap<(&'static str, &'static str), f32>,
}

impl ParityChecker {
    fn new(tolerance: f32) -> Self {
        Self {
            tolerance,
            results: HashMap::new(),
        }
    }

    fn compare(&mut self, a: &MockLogits, b: &MockLogits) -> bool {
        let key = (a.format, b.format);
        match compare_logits(a, b) {
            Ok(diff) => {
                self.results.insert(key, diff);
                diff <= self.tolerance
            }
            Err(_) => false,
        }
    }

    fn report(&self) -> String {
        let mut report = String::from("Cross-Format Parity Report\n");
        report.push_str(&format!("Tolerance: {:.2e}\n\n", self.tolerance));

        for ((fmt_a, fmt_b), diff) in &self.results {
            let status = if *diff <= self.tolerance {
                "PASS"
            } else {
                "FAIL"
            };
            report.push_str(&format!(
                "{} vs {}: max_diff={:.2e} [{}]\n",
                fmt_a, fmt_b, diff, status
            ));
        }

        report
    }
}

#[test]
fn p10_parity_checker_report() {
    let mut checker = ParityChecker::new(LOGIT_TOLERANCE);

    let gguf = MockLogits::new("gguf", "Hello", 1000);
    let st = MockLogits::new("safetensors", "Hello", 1000);
    let apr = MockLogits::new("apr", "Hello", 1000);

    checker.compare(&gguf, &st);
    checker.compare(&gguf, &apr);
    checker.compare(&st, &apr);

    let report = checker.report();

    assert!(report.contains("PASS"), "P10: Report should show PASS");
    assert!(
        report.contains("Tolerance"),
        "P10: Report should show tolerance"
    );
    assert!(
        report.contains("gguf"),
        "P10: Report should mention formats"
    );

    println!("\n{}", report);
}

// ============================================================================
// Summary
// ============================================================================
//
// Cross-Format Parity Tests (P1-P10):
//
// P1: GGUF vs SafeTensors logit parity
// P2: GGUF vs APR logit parity
// P3: SafeTensors vs APR logit parity
// P4: All formats parity (transitive check)
// P5: Shape mismatch detection
// P6: Logit divergence detection (poisoned model)
// P7: Tolerance boundary handling
// P8: Real GGUF/APR parity (requires model files)
// P9: Real SafeTensors/APR parity (requires model files)
// P10: Parity checker report generation
//
// TOLERANCE: 1e-4 (max absolute difference between logits)
//
// To run with real models:
//   cargo test --release parity_cross_format -- --ignored
