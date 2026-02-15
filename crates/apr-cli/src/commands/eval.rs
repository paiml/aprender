//! Eval Command Implementation
//!
//! Implements spec §H13: Perplexity evaluation on standard datasets.
//!
//! # Usage
//!
//! ```bash
//! apr eval model.gguf --dataset wikitext-2   # Evaluate GGUF on WikiText-2
//! apr eval model.apr --dataset lambada       # Evaluate APR on LAMBADA
//! apr eval model.safetensors --text "Hello"  # Evaluate SafeTensors on custom text
//! ```
//!
//! Toyota Way: Jidoka - fail fast if perplexity exceeds threshold.
//!
//! # PMAT-128 Fix: GGUF Weight Loading
//!
//! This module now uses realizar's GGUF inference engine for GGUF models,
//! fixing the F-EVAL bug where GGUF models showed PPL ~1000 due to
//! uninitialized weights.

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use std::path::Path;
use std::time::Instant;

/// Supported evaluation datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dataset {
    /// WikiText-2 test set (standard LM benchmark)
    WikiText2,
    /// LAMBADA (last word prediction)
    Lambada,
    /// Custom text input
    Custom,
}

impl std::str::FromStr for Dataset {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wikitext-2" | "wikitext2" => Ok(Self::WikiText2),
            "lambada" => Ok(Self::Lambada),
            "custom" => Ok(Self::Custom),
            _ => Err(format!(
                "Unknown dataset: {s}. Use: wikitext-2, lambada, or custom"
            )),
        }
    }
}

/// Evaluation configuration
struct EvalConfig {
    /// Dataset to evaluate on
    dataset: Dataset,
    /// Custom text (if dataset is Custom)
    text: Option<String>,
    /// Maximum tokens to evaluate
    max_tokens: usize,
    /// Perplexity threshold for pass/fail
    threshold: f32,
}

/// Evaluation results
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EvalResult {
    /// Perplexity score (lower is better)
    pub perplexity: f32,
    /// Cross-entropy loss
    pub cross_entropy: f32,
    /// Number of tokens evaluated
    pub tokens_evaluated: usize,
    /// Evaluation time
    pub eval_time_secs: f32,
    /// Whether perplexity is below threshold
    pub passed: bool,
    /// Threshold used
    pub threshold: f32,
}

/// Run the eval command
pub(crate) fn run(
    path: &Path,
    dataset: &str,
    text: Option<&str>,
    max_tokens: Option<usize>,
    threshold: Option<f32>,
    json: bool,
) -> Result<()> {
    let dataset_enum: Dataset = dataset
        .parse()
        .map_err(|e: String| CliError::ValidationFailed(e))?;

    let config = EvalConfig {
        dataset: dataset_enum,
        text: text.map(String::from),
        max_tokens: max_tokens.unwrap_or(512),
        threshold: threshold.unwrap_or(20.0), // Per spec H13: PPL > 20 indicates garbage
    };

    if !json {
        print_header(path, &config);
    }

    // Run evaluation
    let result = run_evaluation(path, &config, json)?;

    // GH-248: JSON output mode
    if json {
        return print_json_results(path, &config, &result);
    }

    // Print results
    print_results(&result);

    // Return error if threshold exceeded
    if !result.passed {
        return Err(CliError::ValidationFailed(format!(
            "Perplexity {:.2} exceeds threshold {:.2} (spec H13)",
            result.perplexity, result.threshold
        )));
    }

    Ok(())
}

/// GH-248: JSON output mode for eval results
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_json_results(path: &Path, config: &EvalConfig, result: &EvalResult) -> Result<()> {
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "dataset": format!("{:?}", config.dataset),
        "perplexity": result.perplexity,
        "cross_entropy": result.cross_entropy,
        "tokens_evaluated": result.tokens_evaluated,
        "eval_time_secs": result.eval_time_secs,
        "threshold": result.threshold,
        "passed": result.passed,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !result.passed {
        return Err(CliError::ValidationFailed(format!(
            "Perplexity {:.2} exceeds threshold {:.2} (spec H13)",
            result.perplexity, result.threshold
        )));
    }
    Ok(())
}

fn print_header(path: &Path, config: &EvalConfig) {
    output::section("APR Evaluation");
    println!();
    output::kv("Model", path.display());
    output::kv("Dataset", format!("{:?}", config.dataset));
    output::kv("Max tokens", config.max_tokens);
    output::kv("PPL threshold", config.threshold);
    println!();
}

fn run_evaluation(path: &Path, config: &EvalConfig, json: bool) -> Result<EvalResult> {
    // Detect format
    let is_safetensors = path.extension().is_some_and(|e| e == "safetensors");
    let is_apr = path.extension().is_some_and(|e| e == "apr");
    let is_gguf = path.extension().is_some_and(|e| e == "gguf");

    // GH-242: All 3 formats supported via realizar inference engine
    if is_gguf {
        return run_gguf_evaluation(path, config, json);
    }
    if is_apr {
        return run_apr_evaluation(path, config, json);
    }
    if is_safetensors {
        return run_safetensors_evaluation(path, config, json);
    }

    Err(CliError::ValidationFailed(format!(
        "Unsupported format for eval: {}. Supported: .gguf, .apr, .safetensors",
        path.display()
    )))
}

/// Get evaluation text based on dataset
fn get_eval_text(config: &EvalConfig) -> Result<String> {
    match config.dataset {
        Dataset::WikiText2 => {
            // Sample WikiText-2 style text for testing
            // In production, would load actual WikiText-2 test set
            Ok(SAMPLE_WIKITEXT.to_string())
        }
        Dataset::Lambada => {
            // Sample LAMBADA style text
            Ok(SAMPLE_LAMBADA.to_string())
        }
        Dataset::Custom => config.text.clone().ok_or_else(|| {
            CliError::ValidationFailed("Custom dataset requires --text argument".to_string())
        }),
    }
}

/// PMAT-128: Run GGUF evaluation using realizar's inference engine
///
/// This fixes the F-EVAL bug where GGUF models showed PPL ~1000 due to
/// uninitialized weights. Now uses realizar's `OwnedQuantizedModel` which
/// properly loads GGUF weights.
#[cfg(feature = "inference")]
fn run_gguf_evaluation(path: &Path, config: &EvalConfig, json: bool) -> Result<EvalResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    // GH-257: Progress to stderr when --json, so stdout is clean JSON
    macro_rules! progress {
        ($($arg:tt)*) => {
            if json { eprintln!($($arg)*); } else { println!($($arg)*); }
        };
    }

    progress!("{}", "Loading GGUF model (realizar)...".yellow());
    let start = Instant::now();

    // Load GGUF via mmap
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    // Create quantized model
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    let load_time = start.elapsed();
    progress!(
        "{} in {:.2}s ({} layers, vocab_size={})",
        "Model ready".green(),
        load_time.as_secs_f32(),
        model.config.num_layers,
        model.config.vocab_size
    );
    progress!();

    // Get evaluation text
    let eval_text = get_eval_text(config)?;
    progress!(
        "{}",
        format!("Evaluating on {} characters...", eval_text.len()).yellow()
    );

    // Tokenize using GGUF's embedded tokenizer
    let tokens = mapped
        .model
        .encode(&eval_text)
        .ok_or_else(|| CliError::ValidationFailed("GGUF model has no tokenizer".to_string()))?;

    // Limit tokens
    let tokens: Vec<u32> = if tokens.len() > config.max_tokens {
        tokens[..config.max_tokens].to_vec()
    } else {
        tokens
    };

    if tokens.len() < 2 {
        return Err(CliError::ValidationFailed(
            "Need at least 2 tokens for perplexity calculation".to_string(),
        ));
    }

    progress!(
        "{}",
        format!("Calculating perplexity on {} tokens...", tokens.len()).yellow()
    );

    // Calculate perplexity using realizar's forward pass
    let eval_start = Instant::now();
    let (perplexity, cross_entropy) = calculate_gguf_perplexity(&model, &tokens)?;
    let eval_time = eval_start.elapsed();

    let passed = perplexity <= config.threshold;

    Ok(EvalResult {
        perplexity,
        cross_entropy,
        tokens_evaluated: tokens.len(),
        eval_time_secs: eval_time.as_secs_f32(),
        passed,
        threshold: config.threshold,
    })
}

/// PMAT-128: Fallback for non-inference builds
#[cfg(not(feature = "inference"))]
fn run_gguf_evaluation(_path: &Path, _config: &EvalConfig, _json: bool) -> Result<EvalResult> {
    Err(CliError::ValidationFailed(
        "Evaluation requires 'inference' feature. Rebuild with: \
         cargo install --path crates/apr-cli --features inference"
            .to_string(),
    ))
}

/// GH-242: APR evaluation using realizar's AprTransformer
#[cfg(feature = "inference")]
fn run_apr_evaluation(path: &Path, config: &EvalConfig, json: bool) -> Result<EvalResult> {
    use realizar::apr_transformer::{AprKVCache, AprTransformer};

    macro_rules! progress {
        ($($arg:tt)*) => {
            if json { eprintln!($($arg)*); } else { println!($($arg)*); }
        };
    }

    progress!("{}", "Loading APR model (realizar)...".yellow());
    let start = Instant::now();

    let transformer = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let load_time = start.elapsed();
    progress!(
        "{} in {:.2}s ({} layers, vocab_size={})",
        "Model ready".green(),
        load_time.as_secs_f32(),
        transformer.config.num_layers,
        transformer.config.vocab_size
    );
    progress!();

    let eval_text = get_eval_text(config)?;
    let tokens = tokenize_for_eval(path, &eval_text)?;
    let tokens: Vec<u32> = if tokens.len() > config.max_tokens {
        tokens[..config.max_tokens].to_vec()
    } else {
        tokens
    };
    validate_token_count(&tokens)?;

    progress!(
        "{}",
        format!("Calculating perplexity on {} tokens...", tokens.len()).yellow()
    );

    let eval_start = Instant::now();
    let vocab_size = transformer.config.vocab_size;
    let mut cache = AprKVCache::new(&transformer.config);
    let (perplexity, cross_entropy) =
        calculate_apr_perplexity(&transformer, &mut cache, &tokens, vocab_size)?;
    let eval_time = eval_start.elapsed();

    let passed = perplexity <= config.threshold;
    Ok(EvalResult {
        perplexity,
        cross_entropy,
        tokens_evaluated: tokens.len(),
        eval_time_secs: eval_time.as_secs_f32(),
        passed,
        threshold: config.threshold,
    })
}

#[cfg(not(feature = "inference"))]
fn run_apr_evaluation(_path: &Path, _config: &EvalConfig, _json: bool) -> Result<EvalResult> {
    Err(CliError::ValidationFailed(
        "Evaluation requires 'inference' feature. Rebuild with: \
         cargo install --path crates/apr-cli --features inference"
            .to_string(),
    ))
}

/// GH-242: SafeTensors evaluation using realizar's SafeTensors→AprTransformer path
#[cfg(feature = "inference")]
fn run_safetensors_evaluation(path: &Path, config: &EvalConfig, json: bool) -> Result<EvalResult> {
    use realizar::apr_transformer::AprKVCache;
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    macro_rules! progress {
        ($($arg:tt)*) => {
            if json { eprintln!($($arg)*); } else { println!($($arg)*); }
        };
    }

    progress!("{}", "Loading SafeTensors model (realizar)...".yellow());
    let start = Instant::now();

    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors: {e}")))?;

    let load_time = start.elapsed();
    progress!(
        "{} in {:.2}s ({} layers, vocab_size={})",
        "Model ready".green(),
        load_time.as_secs_f32(),
        transformer.config.num_layers,
        transformer.config.vocab_size
    );
    progress!();

    let eval_text = get_eval_text(config)?;
    let tokens = tokenize_for_eval(path, &eval_text)?;
    let tokens: Vec<u32> = if tokens.len() > config.max_tokens {
        tokens[..config.max_tokens].to_vec()
    } else {
        tokens
    };
    validate_token_count(&tokens)?;

    progress!(
        "{}",
        format!("Calculating perplexity on {} tokens...", tokens.len()).yellow()
    );

    let eval_start = Instant::now();
    let vocab_size = transformer.config.vocab_size;
    let mut cache = AprKVCache::new(&transformer.config);
    let (perplexity, cross_entropy) =
        calculate_apr_perplexity(&transformer, &mut cache, &tokens, vocab_size)?;
    let eval_time = eval_start.elapsed();

    let passed = perplexity <= config.threshold;
    Ok(EvalResult {
        perplexity,
        cross_entropy,
        tokens_evaluated: tokens.len(),
        eval_time_secs: eval_time.as_secs_f32(),
        passed,
        threshold: config.threshold,
    })
}

#[cfg(not(feature = "inference"))]
fn run_safetensors_evaluation(
    _path: &Path,
    _config: &EvalConfig,
    _json: bool,
) -> Result<EvalResult> {
    Err(CliError::ValidationFailed(
        "Evaluation requires 'inference' feature. Rebuild with: \
         cargo install --path crates/apr-cli --features inference"
            .to_string(),
    ))
}

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

/// PMAT-128: Calculate perplexity using realizar's GGUF inference
#[cfg(feature = "inference")]
fn calculate_gguf_perplexity(
    model: &realizar::gguf::OwnedQuantizedModel,
    tokens: &[u32],
) -> Result<(f32, f32)> {
    use realizar::gguf::OwnedQuantizedKVCache;

    let vocab_size = model.config.vocab_size;
    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;

    // Create KV cache for efficient inference
    let mut cache = OwnedQuantizedKVCache::new(
        model.config.num_layers,
        model.config.hidden_dim,
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

    // Quality interpretation
    let quality = if result.perplexity < 10.0 {
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
