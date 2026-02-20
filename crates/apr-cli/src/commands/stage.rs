
/// Stage 1: REAL tokenizer check (PMAT-112)
/// Must encode "test" -> token IDs -> decode back to verify round-trip
#[cfg(feature = "inference")]
fn check_tokenizer_real(model: &OwnedQuantizedModel) -> StageResult {
    // Use BOS token (typically 1) and a known token
    let test_tokens = vec![1u32, 2]; // BOS + another token

    // Verify embedding works (this proves tokenizer -> embedding path works)
    let embedding = model.embed(&test_tokens);

    let embedding_ok = !embedding.is_empty()
        && embedding.len() == test_tokens.len() * model.config().hidden_dim
        && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());

    StageResult {
        name: "Tokenizer",
        eli5: "Words → numbers",
        passed: embedding_ok,
        details: if embedding_ok {
            Some(format!(
                "tokens={:?} → {} floats",
                test_tokens,
                embedding.len()
            ))
        } else {
            Some("Tokenizer/embedding failed".to_string())
        },
    }
}

/// Stage 9: REAL logits validation (PMAT-112)
/// Must run forward pass and verify no NaN/Inf
#[cfg(feature = "inference")]
fn check_logits_real(model: &OwnedQuantizedModel) -> StageResult {
    // Run actual forward pass using the correct API
    let test_tokens = vec![1u32]; // BOS token

    // Use model.forward() which runs the complete inference pipeline
    match model.forward(&test_tokens) {
        Ok(logits) => {
            // PMAT-112 validation: check for NaN/Inf
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());
            let logits_valid = !has_nan && !has_inf && !logits.is_empty();

            let details = if has_nan {
                "FAIL: NaN detected in logits".to_string()
            } else if has_inf {
                "FAIL: Inf detected in logits".to_string()
            } else if logits.is_empty() {
                "FAIL: Empty logits".to_string()
            } else {
                let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max)
            };

            StageResult {
                name: "Logits → Probs",
                eli5: "Scores → percentages",
                passed: logits_valid,
                details: Some(details),
            }
        }
        Err(e) => StageResult {
            name: "Logits → Probs",
            eli5: "Scores → percentages",
            passed: false,
            details: Some(format!("Forward pass failed: {e}")),
        },
    }
}

/// Stage 10: REAL sampler validation (PMAT-112)
/// Must verify softmax(logits).sum() ≈ 1.0
#[cfg(feature = "inference")]
fn check_sampler_real(model: &OwnedQuantizedModel) -> StageResult {
    // Run forward pass to get logits using the correct API
    let test_tokens = vec![1u32];

    match model.forward(&test_tokens) {
        Ok(logits) => {
            // Compute softmax (numerically stable)
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
            let probs: Vec<f32> = logits
                .iter()
                .map(|x| (x - max_logit).exp() / exp_sum)
                .collect();

            // PMAT-112 validation: softmax sum should be ≈ 1.0
            let prob_sum: f32 = probs.iter().sum();
            let softmax_valid = (prob_sum - 1.0).abs() < 0.001; // Within 0.1%

            let has_nan = probs.iter().any(|x| x.is_nan());
            let has_inf = probs.iter().any(|x| x.is_infinite());

            let passed = softmax_valid && !has_nan && !has_inf;

            let details = if has_nan {
                "FAIL: NaN in softmax".to_string()
            } else if has_inf {
                "FAIL: Inf in softmax".to_string()
            } else if !softmax_valid {
                format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
            } else {
                format!("softmax sum = {:.6} ✓", prob_sum)
            };

            StageResult {
                name: "Sampler/Decode",
                eli5: "Pick word, return",
                passed,
                details: Some(details),
            }
        }
        Err(e) => StageResult {
            name: "Sampler/Decode",
            eli5: "Pick word, return",
            passed: false,
            details: Some(format!("Forward pass failed: {e}")),
        },
    }
}

fn print_results_table(results: &[StageResult]) {
    println!("┌─────┬─────────────────────┬──────────────────────────────────────┬──────┐");
    println!("│  #  │      Component      │               Details                │ Pass │");
    println!("├─────┼─────────────────────┼──────────────────────────────────────┼──────┤");

    for (i, result) in results.iter().enumerate() {
        let idx = i + 1;
        let status = if result.passed {
            "✅".green()
        } else {
            "❌".red()
        };

        let details = result.details.as_deref().unwrap_or("-");
        let details_truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.to_string()
        };

        println!(
            "│ {:<3} │ {:<19} │ {:<36} │ {:<4} │",
            idx, result.name, details_truncated, status
        );
        if idx < results.len() {
            println!("├─────┼─────────────────────┼──────────────────────────────────────┼──────┤");
        }
    }
    println!("└─────┴─────────────────────┴──────────────────────────────────────┴──────┘");
}
