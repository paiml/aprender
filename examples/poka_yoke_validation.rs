//! APR-POKA-001: Poka-Yoke Validation Example
//!
//! Demonstrates the Poka-yoke (mistake-proofing) validation system for ML models.
//! Poka-yoke is a Toyota Way concept: build quality in at the source, not at inspection.
//!
//! # Key Concepts
//!
//! - **Gate**: A single validation check with pass/fail and actionable error
//! - **PokaYokeResult**: Collection of gates with score (0-100) and grade (A+ to F)
//! - **PokaYoke trait**: Implement for your model type to define validation rules
//! - **Jidoka**: Save is REFUSED if quality_score=0 (stop the line)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example poka_yoke_validation
//! ```

use aprender::format::validation::{fail_no_validation_rules, Gate, PokaYoke, PokaYokeResult};
use aprender::format::{save, ModelType, SaveOptions};
use serde::{Deserialize, Serialize};
use std::fs;

/// Example ML model with validation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioModel {
    /// Model name
    name: String,
    /// Whether filterbank is embedded
    has_filterbank: bool,
    /// Filterbank max value (should be <0.1 for Slaney normalization)
    filterbank_max: f32,
    /// Number of encoder layers (should be ≥4)
    encoder_layers: usize,
    /// Vocabulary size (should be >0)
    vocab_size: usize,
}

impl AudioModel {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            has_filterbank: false,
            filterbank_max: 1.0,
            encoder_layers: 0,
            vocab_size: 0,
        }
    }

    fn with_filterbank(mut self, max_value: f32) -> Self {
        self.has_filterbank = true;
        self.filterbank_max = max_value;
        self
    }

    fn with_encoder_layers(mut self, count: usize) -> Self {
        self.encoder_layers = count;
        self
    }

    fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }
}

/// Implement PokaYoke for AudioModel
/// Each gate has actionable error messages telling exactly how to fix issues
impl PokaYoke for AudioModel {
    fn poka_yoke_validate(&self) -> PokaYokeResult {
        let mut result = PokaYokeResult::new();

        // Gate 1: Filterbank must be embedded (20 points)
        if self.has_filterbank {
            result.add_gate(Gate::pass("filterbank_present", 20));
        } else {
            result.add_gate(Gate::fail(
                "filterbank_present",
                20,
                "Fix: Embed Slaney-normalized filterbank via MelFilterbankData::mel_80()",
            ));
        }

        // Gate 2: Filterbank must be Slaney-normalized (30 points)
        validate_filterbank_normalization(&mut result, self.has_filterbank, self.filterbank_max);

        // Gate 3: Sufficient encoder layers (25 points)
        validate_encoder_layers(&mut result, self.encoder_layers);

        // Gate 4: Vocabulary size (25 points)
        validate_vocabulary(&mut result, self.vocab_size);

        result
    }
}

/// Validate filterbank normalization (Slaney: max < 0.1)
fn validate_filterbank_normalization(result: &mut PokaYokeResult, has_fb: bool, max: f32) {
    if !has_fb {
        result.add_gate(Gate::fail(
            "filterbank_normalized",
            30,
            "Fix: Add filterbank first (see filterbank_present gate)",
        ));
    } else if max < 0.1 {
        result.add_gate(Gate::pass("filterbank_normalized", 30));
    } else {
        result.add_gate(Gate::fail(
            "filterbank_normalized",
            30,
            format!("Fix: Apply 2.0/bandwidth normalization (max={max:.4}, expected <0.1)"),
        ));
    }
}

/// Validate encoder layer count
fn validate_encoder_layers(result: &mut PokaYokeResult, layers: usize) {
    if layers >= 4 {
        result.add_gate(Gate::pass("encoder_layers", 25));
    } else {
        result.add_gate(Gate::fail(
            "encoder_layers",
            25,
            format!("Fix: Model needs ≥4 encoder layers (has {layers})"),
        ));
    }
}

/// Validate vocabulary size
fn validate_vocabulary(result: &mut PokaYokeResult, vocab: usize) {
    if vocab > 0 {
        result.add_gate(Gate::pass("vocabulary_size", 25));
    } else {
        result.add_gate(Gate::fail(
            "vocabulary_size",
            25,
            "Fix: Set vocabulary size > 0 for tokenization",
        ));
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("              APR-POKA-001: Poka-Yoke Validation Demo");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Demo 1: Perfect model (A+ grade)
    demo_perfect_model();

    // Demo 2: Partially valid model (C grade)
    demo_partial_model();

    // Demo 3: Failing model (F grade) - demonstrates Jidoka
    demo_failing_model();

    // Demo 4: Gate inspection
    demo_gate_inspection();

    // Demo 5: Bulk construction with from_gates()
    demo_from_gates();

    // Demo 6: Helper function for unvalidated models
    demo_fail_no_validation_rules();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                     Demo Complete");
    println!("═══════════════════════════════════════════════════════════════════");
}

/// Demo 1: A perfectly configured model
fn demo_perfect_model() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Perfect Model (A+ Grade)                                │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let model = AudioModel::new("whisper-tiny-perfect")
        .with_filterbank(0.05) // Slaney-normalized (max < 0.1)
        .with_encoder_layers(4)
        .with_vocab_size(51865);

    let result = model.poka_yoke_validate();
    print_validation_result(&result);

    // Save with quality score
    let temp_path = "/tmp/poka_yoke_demo_perfect.apr";
    let options = SaveOptions::new()
        .with_name("whisper-tiny-perfect")
        .with_poka_yoke_result(&result);

    match save(&model, ModelType::LinearRegression, temp_path, options) {
        Ok(()) => {
            println!("  ✅ Model saved successfully to {temp_path}");
            println!(
                "     Quality score embedded in header byte 22: {}",
                result.score
            );
            let _ = fs::remove_file(temp_path);
        }
        Err(e) => println!("  ❌ Save failed: {e}"),
    }
    println!();
}

/// Demo 2: A model with some issues (passing but not perfect)
fn demo_partial_model() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: Partial Model (C Grade - Passing)                       │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let model = AudioModel::new("whisper-tiny-partial")
        .with_filterbank(0.5) // NOT Slaney-normalized (max >= 0.1)
        .with_encoder_layers(4)
        .with_vocab_size(51865);

    let result = model.poka_yoke_validate();
    print_validation_result(&result);

    // Show error summary for failed gates
    if !result.failed_gates().is_empty() {
        println!("  📋 Error Summary:");
        println!("{}", indent_lines(&result.error_summary(), "     "));
    }

    // Save still allowed (score >= 60)
    let temp_path = "/tmp/poka_yoke_demo_partial.apr";
    let options = SaveOptions::new()
        .with_name("whisper-tiny-partial")
        .with_poka_yoke_result(&result);

    match save(&model, ModelType::LinearRegression, temp_path, options) {
        Ok(()) => {
            println!("  ✅ Model saved (with warnings) to {temp_path}");
            let _ = fs::remove_file(temp_path);
        }
        Err(e) => println!("  ❌ Save failed: {e}"),
    }
    println!();
}

/// Demo 3: A completely misconfigured model (Jidoka - stop the line)
fn demo_failing_model() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: Failing Model (F Grade - Jidoka Triggered)              │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Model with all defaults - fails all gates
    let model = AudioModel::new("whisper-broken");

    let result = model.poka_yoke_validate();
    print_validation_result(&result);

    // Show all errors
    println!("  📋 Error Summary (Actionable Fixes):");
    println!("{}", indent_lines(&result.error_summary(), "     "));

    // Jidoka: Save REFUSED because score = 0
    let temp_path = "/tmp/poka_yoke_demo_fail.apr";
    let options = SaveOptions::new()
        .with_name("whisper-broken")
        .with_poka_yoke_result(&result);

    println!("\n  🛑 Attempting to save model with quality_score=0...");
    match save(&model, ModelType::LinearRegression, temp_path, options) {
        Ok(()) => println!("  ✅ Model saved (unexpected!)"),
        Err(e) => {
            println!("  ❌ JIDOKA: Save refused!");
            println!("     Error: {e}");
            println!("     This is intentional - Poka-yoke prevents shipping broken models.");
        }
    }
    println!();
}

/// Demo 4: Inspect individual gates
fn demo_gate_inspection() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 4: Gate Inspection                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let model = AudioModel::new("inspection-demo")
        .with_filterbank(0.05)
        .with_encoder_layers(2) // Insufficient
        .with_vocab_size(0); // Missing

    let result = model.poka_yoke_validate();

    println!("  Gate Details:");
    println!("  {:─<65}", "");
    println!(
        "  {:<20} {:<6} {:<6} {:<6} Error",
        "Gate Name", "Passed", "Points", "Max"
    );
    println!("  {:─<65}", "");

    for gate in &result.gates {
        let status = if gate.passed { "✅" } else { "❌" };
        let error = gate.error.as_deref().unwrap_or("-");
        println!(
            "  {:20} {:6} {:6} {:6} {}",
            gate.name,
            status,
            gate.points,
            gate.max_points,
            truncate_error(error, 30)
        );
    }

    println!("  {:─<65}", "");
    println!(
        "  Total Score: {}/{}  Grade: {}",
        result.score,
        result.max_score,
        result.grade()
    );
    println!();
}

/// Demo 5: Bulk construction with from_gates()
fn demo_from_gates() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 5: Bulk Construction with from_gates()                     │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Create gates as a vector
    let gates = vec![
        Gate::pass("data_integrity", 25),
        Gate::pass("model_architecture", 25),
        Gate::fail("hyperparameters", 25, "Fix: learning_rate must be > 0"),
        Gate::pass("output_shape", 25),
    ];

    // Bulk construct the result
    let result = PokaYokeResult::from_gates(gates);

    println!(
        "  Created PokaYokeResult from {} gates in one call",
        result.gates.len()
    );
    print_validation_result(&result);
    println!();
}

/// Demo 6: Helper for models without PokaYoke implementation
fn demo_fail_no_validation_rules() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 6: fail_no_validation_rules() Helper                       │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Use when saving models that don't implement PokaYoke trait
    let result = fail_no_validation_rules();

    println!("  Use this helper when model doesn't implement PokaYoke trait:");
    print_validation_result(&result);

    println!("\n  📋 Error Summary:");
    println!("{}", indent_lines(&result.error_summary(), "     "));
    println!();
}

/// Print validation result summary
fn print_validation_result(result: &PokaYokeResult) {
    let status = if result.passed() {
        "✅ PASSED"
    } else {
        "❌ FAILED"
    };
    println!("  Validation: {status}");
    println!("  Score: {}/100 (Grade: {})", result.score, result.grade());
    println!(
        "  Gates: {} passed, {} failed",
        result.gates.iter().filter(|g| g.passed).count(),
        result.failed_gates().len()
    );
}

/// Indent each line of text
fn indent_lines(text: &str, indent: &str) -> String {
    text.lines()
        .map(|line| format!("{indent}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncate error message for display
fn truncate_error(error: &str, max_len: usize) -> String {
    if error.len() <= max_len {
        error.to_string()
    } else {
        format!("{}...", &error[..max_len - 3])
    }
}
