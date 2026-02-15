
/// Run the rosetta verify subcommand
pub fn run_verify(source: &Path, intermediate: &str, tolerance: f32, json: bool) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    let intermediate_format = match intermediate.to_lowercase().as_str() {
        "gguf" => FormatType::Gguf,
        "safetensors" | "st" => FormatType::SafeTensors,
        "apr" => FormatType::Apr,
        other => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown format: {other}. Supported: gguf, safetensors, apr"
            )));
        }
    };

    let rosetta = RosettaStone::new();

    if !json {
        println!(
            "{}",
            "=== Rosetta Stone Round-Trip Verification ==="
                .cyan()
                .bold()
        );
        println!();
        println!("Source: {}", source.display());
        println!("Intermediate: {intermediate_format}");
        println!("Tolerance: {tolerance}");
        println!();
        println!("{}", "Verifying round-trip...".yellow());
    }

    let report = rosetta
        .verify_roundtrip(source, intermediate_format)
        .map_err(|e| CliError::ValidationFailed(format!("Verification failed: {e}")))?;

    if json {
        print_verification_json(&report);
    } else {
        println!();
        println!("{}", "=== Verification Report ===".cyan().bold());
        println!("Equivalent: {}", report.is_equivalent);
        println!("Max Diff: {:.2e}", report.max_diff);
        println!("Mean Diff: {:.2e}", report.mean_diff);

        if !report.failed_tensors.is_empty() {
            println!();
            println!("{}", "Failed tensors:".red());
            for t in &report.failed_tensors {
                println!("  - {t}");
            }
        }

        println!();
        if report.passes_with_tolerance(tolerance) {
            println!("{}", "Round-trip verification PASSED".green().bold());
        } else {
            println!("{}", "Round-trip verification FAILED".red().bold());
        }
    }

    Ok(())
}

/// Print the header box for inference comparison report
fn print_compare_header(model_a: &Path, model_b: &Path, prompt: &str) {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║                     INFERENCE COMPARISON REPORT (PMAT-114)                   ║".cyan()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
    println!(
        "║ Model A: {:<66} ║",
        truncate_path(model_a.display().to_string(), 66)
    );
    println!(
        "║ Model B: {:<66} ║",
        truncate_path(model_b.display().to_string(), 66)
    );
    println!(
        "║ Prompt: {:?}{} ║",
        prompt,
        " ".repeat(59_usize.saturating_sub(prompt.len()))
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
}

/// Print JSON output for inference comparison
#[allow(clippy::too_many_arguments)]
fn print_compare_json(
    model_a: &Path,
    model_b: &Path,
    prompt: &str,
    total_tokens: usize,
    mismatches: usize,
    tolerance: f32,
    text_a: &str,
    text_b: &str,
) {
    let match_rate = if total_tokens > 0 {
        1.0 - (mismatches as f64 / total_tokens as f64)
    } else {
        0.0
    };

    println!("{{");
    println!("  \"model_a\": \"{}\",", model_a.display());
    println!("  \"model_b\": \"{}\",", model_b.display());
    println!("  \"prompt\": {:?},", prompt);
    println!("  \"total_tokens\": {},", total_tokens);
    println!("  \"mismatches\": {},", mismatches);
    println!("  \"match_rate\": {:.4},", match_rate);
    println!("  \"text_a\": {:?},", text_a);
    println!("  \"text_b\": {:?},", text_b);
    println!(
        "  \"passed\": {}",
        mismatches == 0 || (1.0 - match_rate as f32) <= tolerance
    );
    println!("}}");
}

/// Validate that tokens were captured from both models (GH-188)
fn validate_captured_tokens(text_a: &str, text_b: &str) -> Result<()> {
    let a_empty = text_a.is_empty() || text_a.contains("tok/s");
    let b_empty = text_b.is_empty() || text_b.contains("tok/s");

    if a_empty && b_empty {
        return Err(CliError::ValidationFailed(
            "TRACING BROKEN: No tokens captured from either model. Check APR_TRACE_LOGITS parsing."
                .to_string(),
        ));
    } else if a_empty {
        return Err(CliError::ValidationFailed(format!(
            "Model A produced no output. Model B: {:?}",
            text_b
        )));
    } else if b_empty {
        return Err(CliError::ValidationFailed(format!(
            "Model B produced no output. Model A: {:?}",
            text_a
        )));
    }
    Ok(())
}

/// Run the rosetta compare-inference subcommand (PMAT-114)
///
/// Compare inference outputs between two models to debug parity issues.
/// Runs the same prompt through both models and compares logits/outputs.
pub fn run_compare_inference(
    model_a: &Path,
    model_b: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    tolerance: f32,
    json: bool,
) -> Result<()> {
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    if !json {
        print_compare_header(model_a, model_b, prompt);
    }

    // F-GT-002: Check for mixed quantization levels (R3 violation)
    if let Some(warning) = check_mixed_quant_warning(model_a, model_b) {
        if !json {
            println!("{}", warning.yellow());
            println!();
        }
    }

    // Run Model A with APR_TRACE_LOGITS to capture logit data
    if !json {
        println!("{}", "Running Model A...".yellow());
    }

    let result_a = run_model_with_logits(model_a, prompt, max_tokens, temperature)?;

    if !json {
        println!("{}", "Running Model B...".yellow());
    }

    let result_b = run_model_with_logits(model_b, prompt, max_tokens, temperature)?;

    let total_tokens = result_a.tokens.len().min(result_b.tokens.len());
    let mismatches = if json {
        count_token_mismatches(&result_a, &result_b, total_tokens)
    } else {
        print_token_comparison_table(&result_a, &result_b, total_tokens)
    };

    if json {
        print_compare_json(
            model_a,
            model_b,
            prompt,
            total_tokens,
            mismatches,
            tolerance,
            &result_a.output_text,
            &result_b.output_text,
        );
    } else {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        print_inference_diagnosis(
            total_tokens,
            mismatches,
            tolerance,
            &result_a.output_text,
            &result_b.output_text,
        );
    }

    if total_tokens == 0 {
        validate_captured_tokens(&result_a.output_text, &result_b.output_text)?;
    }

    validate_match_tolerance(mismatches, total_tokens, tolerance)
}

/// Count token mismatches between two inference results (no output).
fn count_token_mismatches(
    result_a: &InferenceResult,
    result_b: &InferenceResult,
    total_tokens: usize,
) -> usize {
    (0..total_tokens)
        .filter(|&i| {
            result_a.tokens.get(i).copied().unwrap_or(0)
                != result_b.tokens.get(i).copied().unwrap_or(0)
        })
        .count()
}

/// Print token-by-token comparison table and return mismatch count.
fn print_token_comparison_table(
    result_a: &InferenceResult,
    result_b: &InferenceResult,
    total_tokens: usize,
) -> usize {
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
    println!(
        "{}",
        "║                           TOKEN-BY-TOKEN COMPARISON                           ║".cyan()
    );
    println!(
        "{}",
        "╠───────┬─────────────────────────────────┬────────────────────────────────┬───╣".cyan()
    );
    println!(
        "{}",
        "║ Pos   │ Model A (top-1)                 │ Model B (top-1)                │ Δ ║".cyan()
    );
    println!(
        "{}",
        "╠───────┼─────────────────────────────────┼────────────────────────────────┼───╣".cyan()
    );

    let mut mismatches = 0;
    for i in 0..total_tokens {
        let token_a = result_a.tokens.get(i).copied().unwrap_or(0);
        let token_b = result_b.tokens.get(i).copied().unwrap_or(0);
        let logit_a = result_a.logits.get(i).copied().unwrap_or(0.0);
        let logit_b = result_b.logits.get(i).copied().unwrap_or(0.0);

        let matches = token_a == token_b;
        if !matches {
            mismatches += 1;
        }

        let status_colored = if matches { "✓".green() } else { "✗".red() };

        println!(
            "║ {:<5} │ token={:<5} logit={:<12.2} │ token={:<5} logit={:<11.2} │ {} ║",
            i, token_a, logit_a, token_b, logit_b, status_colored
        );

        if !matches {
            if let Some(top5_a) = result_a.top5.get(i) {
                println!(
                    "║       │ Top-5: {:<24} │{:<32} │   ║",
                    format!("{:?}", top5_a),
                    ""
                );
            }
            if let Some(top5_b) = result_b.top5.get(i) {
                println!(
                    "║       │{:<33} │ Top-5: {:<23} │   ║",
                    "",
                    format!("{:?}", top5_b)
                );
            }
        }
    }
    mismatches
}

/// Validate that match rate meets tolerance threshold.
fn validate_match_tolerance(mismatches: usize, total_tokens: usize, tolerance: f32) -> Result<()> {
    if mismatches > 0 {
        let match_rate = 1.0 - (mismatches as f32 / total_tokens.max(1) as f32);
        if match_rate < (1.0 - tolerance) {
            return Err(CliError::ValidationFailed(format!(
                "Inference mismatch: {}/{} tokens differ ({:.0}% match rate, need {:.0}%)",
                mismatches,
                total_tokens,
                match_rate * 100.0,
                (1.0 - tolerance) * 100.0
            )));
        }
    }
    Ok(())
}
