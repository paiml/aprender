
/// Print diagnosis section for inference comparison (extracted for complexity reduction).
fn print_inference_diagnosis(
    total_tokens: usize,
    mismatches: usize,
    tolerance: f32,
    text_a: &str,
    text_b: &str,
) {
    println!(
        "{}",
        "║                           DIAGNOSIS                                           ║".cyan()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );

    // GH-188 FIX: Detect when tracing captured nothing (inference failure)
    if total_tokens == 0 {
        println!(
            "║ {:<76} ║",
            "⚠️  NO TOKENS CAPTURED - INFERENCE MAY HAVE FAILED!"
                .red()
                .bold()
        );
        println!(
            "║ {:<76} ║",
            "Check that APR_TRACE_LOGITS output is being parsed correctly."
        );
        println!(
            "║ {:<76} ║",
            "Model A output: see below. Model B output: see below."
        );
    } else if mismatches == 0 {
        println!(
            "║ {:<76} ║",
            "All tokens match - models produce identical output"
                .green()
                .bold()
        );
    } else {
        println!(
            "║ {:<76} ║",
            format!(
                "{}/{} tokens differ - see possible causes below",
                mismatches, total_tokens
            )
            .yellow()
        );
        println!(
            "║ Possible causes:                                                              ║"
        );
        println!(
            "║   1. Precision difference (F32 vs Q4K): logit variance ~0.5                  ║"
        );
        println!(
            "║   2. RoPE type mismatch: Qwen2 needs rope_type=2 (NEOX style)                ║"
        );
        println!(
            "║   3. Missing QKV bias: check APR has qkv_proj.bias tensors                   ║"
        );
        println!(
            "║   4. LayerNorm epsilon: check rms_norm_eps matches (1e-6 for Qwen2)          ║"
        );
    }

    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );

    // Result
    let match_rate = if total_tokens > 0 {
        1.0 - (mismatches as f32 / total_tokens as f32)
    } else {
        0.0
    };

    let result_text = if mismatches == 0 {
        "RESULT: INFERENCE MATCH (100%)".green().bold().to_string()
    } else if match_rate >= (1.0 - tolerance) {
        format!(
            "RESULT: PARTIAL MATCH ({:.0}% within tolerance {:.0}%)",
            match_rate * 100.0,
            tolerance * 100.0
        )
        .yellow()
        .bold()
        .to_string()
    } else {
        format!(
            "RESULT: INFERENCE MISMATCH ({}/{} tokens = {:.0}%)",
            mismatches,
            total_tokens,
            match_rate * 100.0
        )
        .red()
        .bold()
        .to_string()
    };
    println!("║ {:<76} ║", result_text);
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );

    // Show actual outputs
    println!();
    println!("{}", "=== Generated Text ===".cyan().bold());
    println!("Model A: {:?}", text_a);
    println!("Model B: {:?}", text_b);

    // GH-188: Direct text comparison for quick diagnosis
    let text_a_clean = text_a.trim();
    let text_b_clean = text_b.trim();
    let text_a_has_content = !text_a_clean.is_empty() && !text_a_clean.contains("tok/s");
    let text_b_has_content = !text_b_clean.is_empty() && !text_b_clean.contains("tok/s");

    if text_a_has_content != text_b_has_content {
        println!();
        println!("{}", "⚠️  TEXT OUTPUT MISMATCH DETECTED:".red().bold());
        if text_a_has_content && !text_b_has_content {
            println!("   Model A produced text, Model B produced nothing/garbage.");
            println!("   → Model B likely has inference bug (layout, kernel, or load issue).");
        } else {
            println!("   Model B produced text, Model A produced nothing/garbage.");
            println!("   → Model A likely has inference bug (layout, kernel, or load issue).");
        }
    } else if text_a_has_content && text_b_has_content && text_a_clean != text_b_clean {
        println!();
        println!("{}", "⚠️  TEXT CONTENT DIFFERS:".yellow().bold());
        println!("   Models produced different outputs (may be precision-related).");
    }
}

/// F-GT-002: Detect quantization level from file path.
///
/// Returns a normalized quantization level string for R3 comparison.
/// SafeTensors are unquantized (BF16/F16/F32), GGUF files contain quant level
/// in their filename (e.g. Q4_K_M, Q6_K), APR files may have been quantized at import.
/// Match a filename against a list of quantization patterns.
fn match_quant_pattern(name: &str, patterns: &[&str]) -> Option<String> {
    patterns
        .iter()
        .find(|q| name.contains(**q))
        .map(|q| q.to_uppercase())
}

fn detect_quant_level_from_path(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let ext = path.extension().map(std::ffi::OsStr::to_ascii_lowercase);
    let ext_str = ext.as_deref().and_then(|e| e.to_str()).unwrap_or("");

    match ext_str {
        "safetensors" => "unquantized (BF16/F16/F32)".to_string(),
        "gguf" => {
            let patterns = &[
                "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k",
                "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q5_k", "q6_k", "q8_0", "f16", "f32",
            ];
            match_quant_pattern(&name, patterns)
                .unwrap_or_else(|| "GGUF (quant unknown)".to_string())
        }
        "apr" => {
            let patterns = &["q4k", "q6k", "q4_k", "q6_k", "q8_0", "f16", "f32"];
            match_quant_pattern(&name, patterns)
                .unwrap_or_else(|| "APR (quant unknown)".to_string())
        }
        _ => "unknown".to_string(),
    }
}

/// F-GT-002: Check for R3 mixed quantization level violation.
///
/// Returns a warning string if the two models have different quantization levels,
/// which violates the R3 ground truth comparison rule.
pub(crate) fn check_mixed_quant_warning(model_a: &Path, model_b: &Path) -> Option<String> {
    let quant_a = detect_quant_level_from_path(model_a);
    let quant_b = detect_quant_level_from_path(model_b);

    if quant_a == quant_b {
        None
    } else {
        Some(format!(
            "F-GT-002 WARNING: Mixed quantization levels detected (R3 violation)\n  \
             Model A: {} ({})\n  \
             Model B: {} ({})\n  \
             Comparing models at different quantization levels may produce \
             misleading differences.\n  \
             For valid comparison, use the same quantization level (R3 rule).",
            model_a.display(),
            quant_a,
            model_b.display(),
            quant_b,
        ))
    }
}

/// Run the rosetta diff-tensors subcommand (GH-188)
///
/// Compares tensor dimensions between two models to detect layout mismatches.
/// GGML stores weights as [in_dim, out_dim] but most ML code expects [out_dim, in_dim].
/// This mismatch causes garbage output (PAD token floods).
/// Print the diff report header box
fn print_diff_header(model_a: &Path, model_b: &Path, count_a: usize, count_b: usize) {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║               TENSOR DIFF REPORT (GH-188: Layout Mismatch Detection)        ║".cyan()
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
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );

    let count_match = count_a == count_b;
    let count_status = if count_match {
        "✓".green()
    } else {
        "✗".red()
    };
    println!(
        "║ {} Tensor Count: A={:<5} B={:<5} {}║",
        count_status,
        count_a,
        count_b,
        if count_match {
            "                                  ".to_string()
        } else {
            format!(
                "MISSING {} TENSORS!",
                (count_a as i64 - count_b as i64).abs()
            )
            .red()
            .bold()
            .to_string()
        }
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
    println!(
        "{}",
        "║ GGML Convention: [in_dim, out_dim] - needs transpose for standard matmul     ║".yellow()
    );
    println!(
        "{}",
        "║ Standard Conv:   [out_dim, in_dim] - expected by most ML code                ║".yellow()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
}

/// Print diff summary in JSON format
fn print_diff_json_summary(
    model_a: &Path,
    model_b: &Path,
    tensors_a_len: usize,
    tensors_b_len: usize,
    layout_mismatches: &[(String, Vec<usize>, Vec<usize>)],
    missing_in_a: &[(String, Vec<usize>)],
    missing_in_b: &[(String, Vec<usize>)],
) {
    println!("{{");
    println!("  \"model_a\": \"{}\",", model_a.display());
    println!("  \"model_b\": \"{}\",", model_b.display());
    println!("  \"tensors_a\": {},", tensors_a_len);
    println!("  \"tensors_b\": {},", tensors_b_len);
    println!("  \"layout_mismatches\": {},", layout_mismatches.len());
    println!("  \"missing_in_a\": {},", missing_in_a.len());
    println!("  \"missing_in_b\": {},", missing_in_b.len());
    if !layout_mismatches.is_empty() {
        println!("  \"mismatched_tensors\": [");
        for (i, (name, shape_a, shape_b)) in layout_mismatches.iter().enumerate() {
            let comma = if i < layout_mismatches.len() - 1 {
                ","
            } else {
                ""
            };
            println!(
                "    {{\"name\": \"{}\", \"shape_a\": {:?}, \"shape_b\": {:?}}}{}",
                name, shape_a, shape_b, comma
            );
        }
        println!("  ],");
    }
    println!(
        "  \"diagnosis\": \"{}\"",
        if layout_mismatches.is_empty() {
            "No layout mismatches detected"
        } else {
            "LAYOUT MISMATCH: Some tensors have transposed dimensions (GGML convention)"
        }
    );
    println!("}}");
}

/// Print diff summary in text format with diagnosis
fn print_diff_text_summary(
    tensors_a_len: usize,
    tensors_b_len: usize,
    layout_mismatches: &[(String, Vec<usize>, Vec<usize>)],
    missing_in_a: &[(String, Vec<usize>)],
    missing_in_b: &[(String, Vec<usize>)],
) {
    println!(
        "{}",
        "║                                 SUMMARY                                       ║".cyan()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
    println!("║ Tensors in A: {:<62} ║", tensors_a_len);
    println!("║ Tensors in B: {:<62} ║", tensors_b_len);
    println!(
        "║ Layout mismatches: {:<56} ║",
        format!("{}", layout_mismatches.len()).red().bold()
    );
    println!("║ Missing in A: {:<62} ║", missing_in_a.len());
    println!("║ Missing in B: {:<62} ║", missing_in_b.len());
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );

    if layout_mismatches.is_empty() {
        println!("║ {} ║", "No layout mismatches detected".green().bold());
    } else {
        println!(
            "{}",
            "║                              DIAGNOSIS                                       ║"
                .red()
                .bold()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!("║ {} ║", "LAYOUT MISMATCH DETECTED!".red().bold());
        println!("║ Tensors with transposed dimensions found. This causes garbage output. ║");
        println!("║  ║");
        println!("║ Root Cause: GGML stores weights as [in_dim, out_dim] ║");
        println!("║              Standard ML expects [out_dim, in_dim] ║");
        println!("║  ║");
        println!("║ {} ║", "Fix Options:".yellow().bold());
        println!("║   1. Transpose tensor data during APR load ║");
        println!("║   2. Use row-major kernels that expect GGML layout ║");
        println!("║   3. Store layout convention in APR metadata ║");
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Mismatched tensors:                                                          ║"
        );
        for (name, shape_a, shape_b) in layout_mismatches {
            println!("║   {} ║", name);
            println!("║     A: {:?} → B: {:?} ║", shape_a, shape_b);
        }
    }

    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );
}

/// Compare a single tensor pair and accumulate mismatches.
#[allow(clippy::too_many_arguments)]
/// Print the text comparison for two tensors that are both present.
fn print_both_present(
    name: &str,
    a: &TensorInfo,
    b: &TensorInfo,
    dims_match: bool,
    is_transposed: bool,
) {
    let separator =
        "╠──────────────────────────────────────────────────────────────────────────────╣".cyan();
    let status = if dims_match {
        "✓".green()
    } else if is_transposed {
        "⚠️".yellow()
    } else {
        "✗".red()
    };
    println!("║ {} {:<72} ║", status, name);
    println!(
        "║   A: {:?} {:>20} {:>15} bytes    ║",
        a.shape, a.dtype, a.size_bytes
    );
    println!(
        "║   B: {:?} {:>20} {:>15} bytes    ║",
        b.shape, b.dtype, b.size_bytes
    );
    if is_transposed {
        println!(
            "║   {} ║",
            "LAYOUT MISMATCH: Dimensions are transposed! Likely GGML convention."
                .red()
                .bold()
        );
        println!(
            "║   {} ║",
            "FIX: Transpose this tensor during load OR use row-major kernel".yellow()
        );
    }
    println!("{separator}");
}
