use crate::http_client::*;
// =========================================================================

/// IMP-158a: Benchmark result schema
#[derive(Debug, Clone)]
pub struct BenchmarkResultSchema {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

impl BenchmarkResultSchema {
    pub fn standard() -> Self {
        Self {
            required_fields: vec![
                "throughput_tps".to_string(),
                "iterations".to_string(),
                "cv_achieved".to_string(),
                "timestamp".to_string(),
            ],
            optional_fields: vec![
                "latency_p50_ms".to_string(),
                "latency_p95_ms".to_string(),
                "latency_p99_ms".to_string(),
                "model_path".to_string(),
                "environment".to_string(),
            ],
        }
    }

    pub fn validate(&self, json: &str) -> std::result::Result<(), Vec<String>> {
        let parsed: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => return Err(vec![format!("Invalid JSON: {}", e)]),
        };

        let mut missing = Vec::new();
        for field in &self.required_fields {
            if !Self::has_field(&parsed, field) {
                missing.push(format!("Missing required field: {}", field));
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    fn has_field(value: &serde_json::Value, field: &str) -> bool {
        // Check top-level and nested in "results"
        if value.get(field).is_some() {
            return true;
        }
        if let Some(results) = value.get("results") {
            if results.get(field).is_some() {
                return true;
            }
        }
        false
    }
}

/// IMP-158a: Test schema validation
#[test]
fn test_imp_158a_schema_validation() {
    let schema = BenchmarkResultSchema::standard();

    // Valid result
    let valid_json = r#"{
        "throughput_tps": 150.0,
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    assert!(
        schema.validate(valid_json).is_ok(),
        "IMP-158a: Valid JSON should pass validation"
    );

    // Invalid result (missing throughput)
    let invalid_json = r#"{
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    let errors = schema.validate(invalid_json).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("throughput_tps")),
        "IMP-158a: Should report missing throughput_tps"
    );

    println!("\nIMP-158a: Schema Validation:");
    println!("  Required fields: {:?}", schema.required_fields);
    println!("  Valid JSON: PASS");
    println!("  Invalid JSON errors: {:?}", errors);
}

/// IMP-158b: Result range validation
#[derive(Debug, Clone)]
pub struct RangeValidator {
    pub field: String,
    pub min: f64,
    pub max: f64,
}

impl RangeValidator {
    pub fn new(field: &str, min: f64, max: f64) -> Self {
        Self {
            field: field.to_string(),
            min,
            max,
        }
    }

    pub fn validate(&self, value: f64) -> std::result::Result<(), String> {
        if value < self.min {
            Err(format!(
                "{} = {} is below minimum {}",
                self.field, value, self.min
            ))
        } else if value > self.max {
            Err(format!(
                "{} = {} exceeds maximum {}",
                self.field, value, self.max
            ))
        } else {
            Ok(())
        }
    }
}

/// IMP-158b: Test range validation
#[test]
fn test_imp_158b_range_validation() {
    // Throughput: 0-10000 tok/s reasonable
    let throughput_validator = RangeValidator::new("throughput_tps", 0.0, 10000.0);
    assert!(
        throughput_validator.validate(150.0).is_ok(),
        "IMP-158b: 150 in range"
    );
    assert!(
        throughput_validator.validate(-1.0).is_err(),
        "IMP-158b: Negative should fail"
    );
    assert!(
        throughput_validator.validate(50000.0).is_err(),
        "IMP-158b: 50000 too high"
    );

    // CV: 0-1.0
    let cv_validator = RangeValidator::new("cv_achieved", 0.0, 1.0);
    assert!(
        cv_validator.validate(0.08).is_ok(),
        "IMP-158b: 0.08 CV in range"
    );
    assert!(
        cv_validator.validate(1.5).is_err(),
        "IMP-158b: 1.5 CV too high"
    );

    println!("\nIMP-158b: Range Validation:");
    println!(
        "  throughput_tps: {:.0}-{:.0}",
        throughput_validator.min, throughput_validator.max
    );
    println!(
        "  cv_achieved: {:.2}-{:.2}",
        cv_validator.min, cv_validator.max
    );
}

/// IMP-158c: Complete result validation
pub struct CompleteValidator {
    pub schema: BenchmarkResultSchema,
    pub range_validators: Vec<RangeValidator>,
}

impl CompleteValidator {
    pub fn standard() -> Self {
        Self {
            schema: BenchmarkResultSchema::standard(),
            range_validators: vec![
                RangeValidator::new("throughput_tps", 0.0, 10000.0),
                RangeValidator::new("cv_achieved", 0.0, 1.0),
                RangeValidator::new("iterations", 1.0, 1000.0),
            ],
        }
    }

    pub fn validate_json(&self, json: &str) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Schema validation
        if let Err(schema_errors) = self.schema.validate(json) {
            errors.extend(schema_errors);
        }

        // Parse for range validation
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json) {
            for validator in &self.range_validators {
                if let Some(value) = Self::get_field_value(&parsed, &validator.field) {
                    if let Err(e) = validator.validate(value) {
                        errors.push(e);
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn get_field_value(value: &serde_json::Value, field: &str) -> Option<f64> {
        value
            .get(field)
            .and_then(serde_json::Value::as_f64)
            .or_else(|| {
                value
                    .get("results")
                    .and_then(|r| r.get(field))
                    .and_then(serde_json::Value::as_f64)
            })
    }
}

/// IMP-158c: Test complete validation
#[test]
fn test_imp_158c_complete_validation() {
    let validator = CompleteValidator::standard();

    let valid = r#"{
        "throughput_tps": 150.0,
        "iterations": 25,
        "cv_achieved": 0.08,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    assert!(
        validator.validate_json(valid).is_ok(),
        "IMP-158c: Valid result should pass"
    );

    let invalid = r#"{
        "throughput_tps": -50.0,
        "iterations": 25,
        "cv_achieved": 2.0,
        "timestamp": "2025-12-13T10:00:00Z"
    }"#;

    let errors = validator.validate_json(invalid).unwrap_err();
    assert!(errors.len() >= 2, "IMP-158c: Should have multiple errors");

    println!("\nIMP-158c: Complete Validation:");
    println!("  Valid JSON: PASS");
    println!("  Invalid JSON errors: {:?}", errors);
}

/// IMP-158d: Comparison result validation
#[derive(Debug, Clone)]
pub struct ComparisonResultValidator;

impl ComparisonResultValidator {
    pub fn validate_comparison(
        realizar_tps: f64,
        reference_tps: f64,
    ) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if realizar_tps <= 0.0 {
            errors.push("Realizar throughput must be positive".to_string());
        }
        if reference_tps <= 0.0 {
            errors.push("Reference throughput must be positive".to_string());
        }

        // Sanity check: throughput shouldn't differ by more than 1000x
        if realizar_tps > 0.0 && reference_tps > 0.0 {
            let ratio = (realizar_tps / reference_tps).max(reference_tps / realizar_tps);
            if ratio > 1000.0 {
                errors.push(format!("Throughput ratio {}x seems unreasonable", ratio));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// IMP-158d: Test comparison validation
#[test]
fn test_imp_158d_comparison_validation() {
    assert!(
        ComparisonResultValidator::validate_comparison(150.0, 256.0).is_ok(),
        "IMP-158d: Valid comparison should pass"
    );

    let errors = ComparisonResultValidator::validate_comparison(-10.0, 256.0).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("positive")),
        "IMP-158d: Should reject negative throughput"
    );

    let ratio_errors = ComparisonResultValidator::validate_comparison(1.0, 100000.0).unwrap_err();
    assert!(
        ratio_errors.iter().any(|e| e.contains("unreasonable")),
        "IMP-158d: Should flag extreme ratios"
    );

    println!("\nIMP-158d: Comparison Validation:");
    println!("  Valid (150 vs 256): PASS");
    println!("  Negative value: {:?}", errors);
    println!("  Extreme ratio: {:?}", ratio_errors);
}

// =========================================================================
// IMP-159: Throughput Variance Tracking (QA-036, EXTREME TDD)
// =========================================================================
// Per spec QA-036: Track throughput variance for statistical confidence.
// CV-based stopping criterion per Hoefler & Belli SC'15.
// Run with: cargo test test_imp_159 --lib --features bench-http

/// IMP-159a: Throughput measurement with variance tracking
#[derive(Debug, Clone)]
pub struct ThroughputWithVariance {
    /// Mean throughput in tokens/second
    pub mean_tps: f64,
    /// Standard deviation of throughput
    pub stddev_tps: f64,
    /// Coefficient of variation (CV = stddev/mean)
    pub cv: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Individual samples for analysis
    pub samples: Vec<f64>,
    /// 95% confidence interval (mean ± margin)
    pub ci_95_margin: f64,
}

impl ThroughputWithVariance {
    /// Create from a vector of throughput samples
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                mean_tps: 0.0,
                stddev_tps: 0.0,
                cv: 0.0,
                sample_count: 0,
                samples: Vec::new(),
                ci_95_margin: 0.0,
            };
        }

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

        // 95% CI margin: t * (stddev / sqrt(n)), using t ≈ 1.96 for large n
        let t_value = if n >= 30 { 1.96 } else { 2.0 };
        let ci_margin = t_value * stddev / (n as f64).sqrt();

        Self {
            mean_tps: mean,
            stddev_tps: stddev,
            cv,
            sample_count: n,
            samples: samples.to_vec(),
            ci_95_margin: ci_margin,
        }
    }

    /// Check if measurement meets CV threshold for reliability
    pub fn meets_cv_threshold(&self, threshold: f64) -> bool {
        self.cv <= threshold && self.sample_count >= 5
    }

    /// Get 95% confidence interval as (lower, upper)
    pub fn confidence_interval(&self) -> (f64, f64) {
        (
            self.mean_tps - self.ci_95_margin,
            self.mean_tps + self.ci_95_margin,
        )
    }
}

/// IMP-159a: Test throughput variance calculation
#[test]
fn test_imp_159a_throughput_variance_calculation() {
    // Stable throughput samples (low variance)
    let stable_samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0,
    ];
    let stable = ThroughputWithVariance::from_samples(&stable_samples);

    // IMP-159a: Mean should be ~100
    assert!(
        (stable.mean_tps - 100.1).abs() < 0.5,
        "IMP-159a: Mean should be ~100, got {:.2}",
        stable.mean_tps
    );

    // IMP-159a: CV should be low (< 5%)
    assert!(
        stable.cv < 0.05,
        "IMP-159a: CV for stable samples should be < 5%, got {:.4}",
        stable.cv
    );

    // IMP-159a: Should meet CV threshold
    assert!(
        stable.meets_cv_threshold(0.10),
        "IMP-159a: Stable samples should meet 10% CV threshold"
    );

    println!("\nIMP-159a: Throughput Variance Calculation:");
    println!("  Samples: {:?}", stable_samples);
    println!("  Mean: {:.2} tok/s", stable.mean_tps);
    println!("  Stddev: {:.2} tok/s", stable.stddev_tps);
    println!("  CV: {:.4} ({:.1}%)", stable.cv, stable.cv * 100.0);
    println!(
        "  95% CI: ({:.2}, {:.2})",
        stable.confidence_interval().0,
        stable.confidence_interval().1
    );
}

include!("compare_variance_aware.rs");
include!("compare_multi_run.rs");
include!("analyze_jit_aware.rs");
