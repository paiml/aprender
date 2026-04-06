
/// Get top-k indices with values from logits
fn get_top_k_indices(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

/// Compute softmax probabilities for top-k tokens
fn compute_top_k_probs(logits: &[f32], top_k: &[(u32, f32)]) -> Vec<(u32, f32)> {
    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp sum for softmax
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

    // Compute probs for top-k
    top_k
        .iter()
        .map(|&(idx, logit)| {
            let prob = (logit - max_logit).exp() / exp_sum;
            (idx, prob)
        })
        .collect()
}

/// Check if decoded output contains garbage characters (APR-TOK-001)
fn is_garbage_output(text: &str) -> bool {
    contract_pre_garbage_oracle!();
    if text.is_empty() {
        return false;
    }

    // Count suspicious characters (CJK private use, replacement chars, etc.)
    let suspicious_count = text
        .chars()
        .filter(|&c| {
            // Unicode replacement character
            c == '\u{FFFD}'
                // Private use area (often indicates bad decoding)
                || ('\u{E000}'..='\u{F8FF}').contains(&c)
                // CJK Extension B/C/D (rarely used, often garbage)
                || ('\u{20000}'..='\u{2FFFF}').contains(&c)
        })
        .count();

    // If more than 30% suspicious, likely garbage
    suspicious_count * 3 > text.chars().count()
}

/// Get hint for error (Jidoka: actionable feedback)
fn get_error_hint(error: &TraceError) -> &'static str {
    match error {
        TraceError::VocabOverflow { .. } => {
            "Check GGUF vocab loading or tokenizer.json compatibility"
        },
        TraceError::NaNDetected { .. } => "Check for numerical overflow in matmul or softmax",
        TraceError::InfDetected { .. } => "Check for division by zero or very large values",
        TraceError::GarbageOutput { .. } => {
            "Token ID may not match tokenizer vocab. Check tokenizer.json vs GGUF vocab"
        },
        TraceError::UnknownToken { .. } => "Token not in vocabulary. Check tokenizer configuration",
        TraceError::ShapeMismatch { .. } => {
            "Tensor dimensions don't match. Check model architecture"
        },
        TraceError::ExecutionFailed { .. } => {
            "Execution failed. Check model config and dependencies"
        },
    }
}

/// Format float for JSON (handle NaN/Inf)
fn format_json_float(v: f32) -> String {
    if v.is_nan() {
        "null".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "\"Infinity\"".to_string()
        } else {
            "\"-Infinity\"".to_string()
        }
    } else {
        format!("{:.6}", v)
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod inference_trace_tests;
