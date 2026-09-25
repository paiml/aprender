
/// GH-242: Tokenize text for evaluation
#[cfg(feature = "inference")]
fn tokenize_for_eval(model_path: &Path, text: &str) -> Result<Vec<u32>> {
    use realizar::apr::AprV2Model;

    // Try tokenizer.json in same directory (uses realizar's BpeTokenizer)
    if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        let tokens = tokenizer.encode(text);
        if !tokens.is_empty() {
            return Ok(tokens);
        }
    }

    // Try encode_text which searches additional paths
    if let Some(tokens) = AprV2Model::encode_text(model_path, text) {
        if !tokens.is_empty() {
            return Ok(tokens);
        }
    }

    Err(CliError::ValidationFailed(
        "No tokenizer found. Place tokenizer.json next to model file.".to_string(),
    ))
}

fn validate_token_count(tokens: &[u32]) -> Result<()> {
    if tokens.len() < 2 {
        return Err(CliError::ValidationFailed(
            "Need at least 2 tokens for perplexity calculation".to_string(),
        ));
    }
    Ok(())
}

/// GH-242: Calculate perplexity using AprTransformer forward pass
#[cfg(feature = "inference")]
fn calculate_apr_perplexity(
    transformer: &realizar::apr_transformer::AprTransformer,
    cache: &mut realizar::apr_transformer::AprKVCache,
    tokens: &[u32],
    vocab_size: usize,
) -> Result<(f32, f32)> {
    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;

    for (pos, window) in tokens.windows(2).enumerate() {
        let input_token = window[0];
        let target_token = window[1];

        let logits = transformer
            .forward_with_cache(input_token, cache, pos)
            .map_err(|e| CliError::ValidationFailed(format!("Forward pass failed: {e}")))?;

        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp: f64 = logits
            .iter()
            .map(|&l| ((l - max_logit) as f64).exp())
            .sum::<f64>()
            .ln()
            + max_logit as f64;

        let target_idx = target_token as usize;
        if target_idx < vocab_size {
            let log_prob = logits[target_idx] as f64 - log_sum_exp;
            total_log_prob += log_prob;
            count += 1;
        }
    }

    if count == 0 {
        return Err(CliError::ValidationFailed(
            "No valid tokens for perplexity calculation".to_string(),
        ));
    }

    let cross_entropy = (-total_log_prob / count as f64) as f32;
    let perplexity = cross_entropy.exp();
    Ok((perplexity, cross_entropy))
}

/// #4270: teacher-forced perplexity through any one-engine `Session` — the logits
/// after position `pos` score `tokens[pos + 1]`. Every architecture that implements
/// `ArchForward` gets the same arithmetic as `calculate_gguf_perplexity`.
#[cfg(feature = "inference")]
pub(crate) fn session_perplexity<F: realizar::session::ArchForward>(
    session: &mut realizar::session::Session<F>,
    tokens: &[u32],
) -> Result<(f32, f32)> {
    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;
    session
        .score(tokens, &mut |pos, logits| {
            let Some(&target) = tokens.get(pos + 1) else {
                return false;
            };
            if let Some(lp) = log_softmax_at(logits, target as usize) {
                total_log_prob += lp;
                count += 1;
            }
            true
        })
        .map_err(|e| CliError::ValidationFailed(format!("Forward pass failed: {e}")))?;
    if count == 0 {
        return Err(CliError::ValidationFailed(
            "No tokens scored for perplexity".to_string(),
        ));
    }
    let cross_entropy = (-total_log_prob / count as f64) as f32;
    Ok((cross_entropy.exp(), cross_entropy))
}

/// `log softmax(logits)[target]`, or `None` when `target` is outside the vocab.
#[cfg(feature = "inference")]
fn log_softmax_at(logits: &[f32], target: usize) -> Option<f64> {
    let target_logit = *logits.get(target)?;
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let log_sum_exp = logits
        .iter()
        .map(|&l| f64::from(l - max_logit).exp())
        .sum::<f64>()
        .ln()
        + f64::from(max_logit);
    Some(f64::from(target_logit) - log_sum_exp)
}

/// PMAT-128: Calculate perplexity using realizar's GGUF inference
#[cfg(feature = "inference")]
fn calculate_gguf_perplexity(
    model: &realizar::gguf::OwnedQuantizedModel,
    tokens: &[u32],
) -> Result<(f32, f32)> {
    use realizar::gguf::OwnedQuantizedKVCache;

    let vocab_size = model.config().vocab_size;
    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;

    // Create KV cache for efficient inference
    let mut cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().hidden_dim,
        tokens.len() + 1, // max_seq_len = num tokens being evaluated
    );

    // Process each token and calculate log probability of next token
    for (pos, window) in tokens.windows(2).enumerate() {
        let input_token = window[0];
        let target_token = window[1];

        // Forward pass to get logits
        let logits = model
            .forward_single_with_cache(input_token, &mut cache, pos)
            .map_err(|e| CliError::ValidationFailed(format!("Forward pass failed: {e}")))?;

        // Compute log softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp: f64 = logits
            .iter()
            .map(|&l| ((l - max_logit) as f64).exp())
            .sum::<f64>()
            .ln()
            + max_logit as f64;

        // Get log probability of target token
        let target_idx = target_token as usize;
        if target_idx < vocab_size {
            let log_prob = logits[target_idx] as f64 - log_sum_exp;
            total_log_prob += log_prob;
            count += 1;
        }
    }

    if count == 0 {
        return Err(CliError::ValidationFailed(
            "No valid tokens for perplexity calculation".to_string(),
        ));
    }

    let cross_entropy = (-total_log_prob / count as f64) as f32;
    let perplexity = cross_entropy.exp();

    Ok((perplexity, cross_entropy))
}

fn print_results(result: &EvalResult) {
    output::section("Results");
    println!();

    // Perplexity (the key metric)
    let ppl_str = format!("{:.2}", result.perplexity);
    if result.passed {
        println!(
            "{} {} {}",
            "Perplexity:".white().bold(),
            ppl_str.green().bold(),
            format!("(PASS: <= {:.1})", result.threshold).green()
        );
    } else {
        println!(
            "{} {} {}",
            "Perplexity:".white().bold(),
            ppl_str.red().bold(),
            format!("(FAIL: > {:.1})", result.threshold).red()
        );
    }

    println!();
    output::kv("Cross-entropy", format!("{:.4}", result.cross_entropy));
    output::kv("Tokens evaluated", result.tokens_evaluated);
    output::kv("Eval time", format!("{:.2}s", result.eval_time_secs));
    println!();

    // GH-599: Quality interpretation — suppress misleading labels for small samples.
    // PPL on <100 tokens is statistically unreliable; don't label a valid model "Garbage".
    let quality = if result.tokens_evaluated < 100 {
        format!("PPL {:.1} (insufficient tokens for quality assessment)", result.perplexity).yellow()
    } else if result.perplexity < 10.0 {
        "Excellent (competitive with SotA)".green()
    } else if result.perplexity < 15.0 {
        "Good (usable quality)".green()
    } else if result.perplexity < 20.0 {
        "Acceptable (minimum threshold)".yellow()
    } else if result.perplexity < 50.0 {
        "Poor (likely undertrained)".red()
    } else {
        "Garbage (model broken)".red().bold()
    };
    output::kv("Quality", quality);
}

// Sample WikiText-2 style text for testing
const SAMPLE_WIKITEXT: &str = r#"
The tower is 324 metres tall, about the same height as an 81-storey building,
and the tallest structure in Paris. Its base is square, measuring 125 metres
on each side. During its construction, the Eiffel Tower surpassed the Washington
Monument to become the tallest man-made structure in the world, a title it held
for 41 years until the Chrysler Building in New York City was finished in 1930.
Due to the addition of a broadcasting aerial at the top of the tower in 1957,
it is now taller than the Chrysler Building by 5.2 metres. Excluding transmitters,
the Eiffel Tower is the second tallest free-standing structure in France after
the Millau Viaduct.
"#;

// Sample LAMBADA style text for testing
const SAMPLE_LAMBADA: &str = r#"
She walked into the room and saw her old friend sitting by the window. After
all these years, she finally understood why he had left. The answer was simple:
he had been afraid of what might happen if he stayed. But now, looking at him,
she realized that fear had cost them both dearly. The time they had lost could
never be recovered. All she could do was sit beside him and hope that somehow,
they could find a way to start again.
"#;

#[cfg(test)]
#[path = "eval_tests.rs"]
mod tests;
