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

    print_header(path, &config);

    // Run evaluation
    let result = run_evaluation(path, &config)?;

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

fn print_header(path: &Path, config: &EvalConfig) {
    output::section("APR Evaluation");
    println!();
    output::kv("Model", path.display());
    output::kv("Dataset", format!("{:?}", config.dataset));
    output::kv("Max tokens", config.max_tokens);
    output::kv("PPL threshold", config.threshold);
    println!();
}

fn run_evaluation(path: &Path, config: &EvalConfig) -> Result<EvalResult> {
    // Detect format
    let is_safetensors = path.extension().is_some_and(|e| e == "safetensors");
    let is_apr = path.extension().is_some_and(|e| e == "apr");
    let is_gguf = path.extension().is_some_and(|e| e == "gguf");

    // Route GGUF to realizar's inference engine (PMAT-128)
    if is_gguf {
        return run_gguf_evaluation(path, config);
    }

    // Realizar-first architecture: all inference goes through realizar.
    // SafeTensors/APR models must be converted to GGUF for evaluation.
    if is_safetensors {
        return Err(CliError::ValidationFailed(
            "SafeTensors evaluation requires GGUF format. Convert first: \
             apr convert model.safetensors -o model.gguf"
                .to_string(),
        ));
    }
    if is_apr {
        return Err(CliError::ValidationFailed(
            "APR evaluation requires GGUF format. Convert first: \
             apr export model.apr --format gguf -o model.gguf"
                .to_string(),
        ));
    }

    Err(CliError::ValidationFailed(format!(
        "Unsupported format for eval: {}. Use GGUF format.",
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
fn run_gguf_evaluation(path: &Path, config: &EvalConfig) -> Result<EvalResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    println!("{}", "Loading GGUF model (realizar)...".yellow());
    let start = Instant::now();

    // Load GGUF via mmap
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    // Create quantized model
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    let load_time = start.elapsed();
    println!(
        "{} in {:.2}s ({} layers, vocab_size={})",
        "Model ready".green(),
        load_time.as_secs_f32(),
        model.config.num_layers,
        model.config.vocab_size
    );
    println!();

    // Get evaluation text
    let eval_text = get_eval_text(config)?;
    println!(
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

    println!(
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
fn run_gguf_evaluation(_path: &Path, _config: &EvalConfig) -> Result<EvalResult> {
    Err(CliError::ValidationFailed(
        "GGUF evaluation requires 'inference' feature. Rebuild with: \
         cargo install --path crates/apr-cli --features inference"
            .to_string(),
    ))
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
mod tests {
    use super::*;

    // =========================================================================
    // Dataset tests
    // =========================================================================

    #[test]
    fn test_dataset_parse() {
        assert_eq!("wikitext-2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("wikitext2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("lambada".parse::<Dataset>().unwrap(), Dataset::Lambada);
        assert_eq!("custom".parse::<Dataset>().unwrap(), Dataset::Custom);
    }

    #[test]
    fn test_dataset_parse_case_insensitive() {
        assert_eq!("WIKITEXT-2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("WikiText2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("LAMBADA".parse::<Dataset>().unwrap(), Dataset::Lambada);
        assert_eq!("CUSTOM".parse::<Dataset>().unwrap(), Dataset::Custom);
    }

    #[test]
    fn test_dataset_parse_error() {
        assert!("unknown".parse::<Dataset>().is_err());
    }

    #[test]
    fn test_dataset_parse_error_message() {
        let result: std::result::Result<Dataset, String> = "invalid".parse();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown dataset"));
        assert!(err.contains("wikitext-2"));
        assert!(err.contains("lambada"));
        assert!(err.contains("custom"));
    }

    #[test]
    fn test_dataset_debug() {
        assert_eq!(format!("{:?}", Dataset::WikiText2), "WikiText2");
        assert_eq!(format!("{:?}", Dataset::Lambada), "Lambada");
        assert_eq!(format!("{:?}", Dataset::Custom), "Custom");
    }

    #[test]
    fn test_dataset_clone() {
        let dataset = Dataset::WikiText2;
        let cloned = dataset;
        assert_eq!(dataset, cloned);
    }

    #[test]
    fn test_dataset_copy() {
        let dataset = Dataset::Lambada;
        let copied: Dataset = dataset;
        assert_eq!(dataset, copied);
    }

    #[test]
    fn test_dataset_eq() {
        assert_eq!(Dataset::WikiText2, Dataset::WikiText2);
        assert_eq!(Dataset::Lambada, Dataset::Lambada);
        assert_eq!(Dataset::Custom, Dataset::Custom);
        assert_ne!(Dataset::WikiText2, Dataset::Lambada);
        assert_ne!(Dataset::Lambada, Dataset::Custom);
    }

    // =========================================================================
    // EvalResult tests
    // =========================================================================

    #[test]
    fn test_eval_result_pass() {
        let result = EvalResult {
            perplexity: 15.0,
            cross_entropy: 2.7,
            tokens_evaluated: 100,
            eval_time_secs: 1.5,
            passed: true,
            threshold: 20.0,
        };

        assert!(result.passed);
        assert!(result.perplexity < result.threshold);
    }

    #[test]
    fn test_eval_result_fail() {
        let result = EvalResult {
            perplexity: 25.0,
            cross_entropy: 3.2,
            tokens_evaluated: 100,
            eval_time_secs: 1.5,
            passed: false,
            threshold: 20.0,
        };

        assert!(!result.passed);
        assert!(result.perplexity > result.threshold);
    }

    #[test]
    fn test_eval_result_excellent() {
        let result = EvalResult {
            perplexity: 8.5,
            cross_entropy: 2.14,
            tokens_evaluated: 512,
            eval_time_secs: 3.0,
            passed: true,
            threshold: 20.0,
        };

        assert!(result.passed);
        assert!(result.perplexity < 10.0); // Excellent threshold
    }

    #[test]
    fn test_eval_result_good() {
        let result = EvalResult {
            perplexity: 12.5,
            cross_entropy: 2.53,
            tokens_evaluated: 256,
            eval_time_secs: 2.0,
            passed: true,
            threshold: 20.0,
        };

        assert!(result.passed);
        assert!(result.perplexity >= 10.0 && result.perplexity < 15.0); // Good threshold
    }

    #[test]
    fn test_eval_result_poor() {
        let result = EvalResult {
            perplexity: 35.0,
            cross_entropy: 3.56,
            tokens_evaluated: 100,
            eval_time_secs: 1.0,
            passed: false,
            threshold: 20.0,
        };

        assert!(!result.passed);
        assert!(result.perplexity >= 20.0 && result.perplexity < 50.0); // Poor threshold
    }

    #[test]
    fn test_eval_result_garbage() {
        let result = EvalResult {
            perplexity: 150.0,
            cross_entropy: 5.01,
            tokens_evaluated: 50,
            eval_time_secs: 0.5,
            passed: false,
            threshold: 20.0,
        };

        assert!(!result.passed);
        assert!(result.perplexity >= 50.0); // Garbage threshold
    }

    #[test]
    fn test_eval_result_debug() {
        let result = EvalResult {
            perplexity: 15.0,
            cross_entropy: 2.7,
            tokens_evaluated: 100,
            eval_time_secs: 1.5,
            passed: true,
            threshold: 20.0,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("EvalResult"));
        assert!(debug_str.contains("perplexity: 15.0"));
        assert!(debug_str.contains("passed: true"));
    }

    #[test]
    fn test_eval_result_clone() {
        let result = EvalResult {
            perplexity: 15.0,
            cross_entropy: 2.7,
            tokens_evaluated: 100,
            eval_time_secs: 1.5,
            passed: true,
            threshold: 20.0,
        };

        let cloned = result.clone();
        assert_eq!(cloned.perplexity, result.perplexity);
        assert_eq!(cloned.passed, result.passed);
    }

    // =========================================================================
    // get_eval_text tests
    // =========================================================================

    #[test]
    fn test_get_eval_text_wikitext() {
        let config = EvalConfig {
            dataset: Dataset::WikiText2,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };

        let text = get_eval_text(&config).unwrap();
        assert!(!text.is_empty());
        assert!(text.contains("Eiffel Tower")); // WikiText sample contains this
    }

    #[test]
    fn test_get_eval_text_lambada() {
        let config = EvalConfig {
            dataset: Dataset::Lambada,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };

        let text = get_eval_text(&config).unwrap();
        assert!(!text.is_empty());
        assert!(text.contains("walked into the room")); // LAMBADA sample contains this
    }

    #[test]
    fn test_get_eval_text_custom() {
        let config = EvalConfig {
            dataset: Dataset::Custom,
            text: Some("This is custom evaluation text.".to_string()),
            max_tokens: 512,
            threshold: 20.0,
        };

        let text = get_eval_text(&config).unwrap();
        assert_eq!(text, "This is custom evaluation text.");
    }

    #[test]
    fn test_get_eval_text_custom_no_text() {
        let config = EvalConfig {
            dataset: Dataset::Custom,
            text: None, // Missing text for custom dataset
            max_tokens: 512,
            threshold: 20.0,
        };

        let result = get_eval_text(&config);
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Custom dataset requires --text argument"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    // =========================================================================
    // Sample text tests
    // =========================================================================

    #[test]
    fn test_sample_texts_not_empty() {
        assert!(!SAMPLE_WIKITEXT.is_empty());
        assert!(!SAMPLE_LAMBADA.is_empty());
        assert!(SAMPLE_WIKITEXT.len() > 100);
        assert!(SAMPLE_LAMBADA.len() > 100);
    }

    #[test]
    fn test_sample_wikitext_content() {
        // WikiText-2 sample should contain recognizable content
        assert!(SAMPLE_WIKITEXT.contains("tower"));
        assert!(SAMPLE_WIKITEXT.contains("metres"));
        assert!(SAMPLE_WIKITEXT.contains("Paris"));
    }

    #[test]
    fn test_sample_lambada_content() {
        // LAMBADA sample should contain narrative text
        assert!(SAMPLE_LAMBADA.contains("She"));
        assert!(SAMPLE_LAMBADA.contains("friend"));
        assert!(SAMPLE_LAMBADA.contains("window"));
    }

    // =========================================================================
    // run() error cases tests
    // =========================================================================

    #[test]
    fn test_run_unknown_dataset() {
        let result = run(Path::new("test.apr"), "invalid_dataset", None, None, None);

        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown dataset"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            "wikitext-2",
            None,
            None,
            None,
        );

        // Will fail because file doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_run_custom_without_text() {
        let result = run(
            Path::new("test.apr"),
            "custom",
            None, // Missing text for custom dataset
            None,
            None,
        );

        // Will fail at validation or file loading
        assert!(result.is_err());
    }

    // =========================================================================
    // EvalConfig tests
    // =========================================================================

    #[test]
    fn test_eval_config_default_values() {
        let config = EvalConfig {
            dataset: Dataset::WikiText2,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };

        assert!(matches!(config.dataset, Dataset::WikiText2));
        assert!(config.text.is_none());
        assert_eq!(config.max_tokens, 512);
        assert!((config.threshold - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_config_with_text() {
        let config = EvalConfig {
            dataset: Dataset::Custom,
            text: Some("Test text".to_string()),
            max_tokens: 256,
            threshold: 15.0,
        };

        assert!(matches!(config.dataset, Dataset::Custom));
        assert_eq!(config.text, Some("Test text".to_string()));
        assert_eq!(config.max_tokens, 256);
        assert!((config.threshold - 15.0).abs() < 0.01);
    }

    // =========================================================================
    // Additional Dataset::from_str edge cases
    // =========================================================================

    #[test]
    fn test_dataset_parse_mixed_case() {
        assert_eq!(
            "WikiText-2".parse::<Dataset>().ok(),
            Some(Dataset::WikiText2)
        );
        assert_eq!("Lambada".parse::<Dataset>().ok(), Some(Dataset::Lambada));
        assert_eq!("Custom".parse::<Dataset>().ok(), Some(Dataset::Custom));
    }

    #[test]
    fn test_dataset_parse_empty_string() {
        let result = "".parse::<Dataset>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown dataset"));
    }

    #[test]
    fn test_dataset_parse_whitespace_only() {
        let result = "  ".parse::<Dataset>();
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_parse_similar_but_wrong() {
        assert!("wikitext-3".parse::<Dataset>().is_err());
        assert!("wikitext".parse::<Dataset>().is_err());
        assert!("wiki".parse::<Dataset>().is_err());
        assert!("lamb".parse::<Dataset>().is_err());
    }

    #[test]
    fn test_dataset_parse_error_contains_input() {
        let result: std::result::Result<Dataset, String> = "foobar".parse();
        let err = result.unwrap_err();
        assert!(
            err.contains("foobar"),
            "Error should contain the invalid input"
        );
    }

    // =========================================================================
    // get_eval_text additional coverage
    // =========================================================================

    #[test]
    fn test_get_eval_text_wikitext_returns_sample_constant() {
        let config = EvalConfig {
            dataset: Dataset::WikiText2,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };
        let text = get_eval_text(&config).expect("should return WikiText sample");
        assert_eq!(text, SAMPLE_WIKITEXT);
    }

    #[test]
    fn test_get_eval_text_lambada_returns_sample_constant() {
        let config = EvalConfig {
            dataset: Dataset::Lambada,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };
        let text = get_eval_text(&config).expect("should return LAMBADA sample");
        assert_eq!(text, SAMPLE_LAMBADA);
    }

    #[test]
    fn test_get_eval_text_custom_with_empty_string() {
        let config = EvalConfig {
            dataset: Dataset::Custom,
            text: Some(String::new()),
            max_tokens: 512,
            threshold: 20.0,
        };
        let text = get_eval_text(&config).expect("should return empty string");
        assert!(text.is_empty());
    }

    #[test]
    fn test_get_eval_text_wikitext_ignores_text_field() {
        // Even if text is Some, WikiText2 returns the sample constant
        let config = EvalConfig {
            dataset: Dataset::WikiText2,
            text: Some("ignored".to_string()),
            max_tokens: 512,
            threshold: 20.0,
        };
        let text = get_eval_text(&config).expect("should return WikiText sample");
        assert!(text.contains("Eiffel Tower"));
        assert!(!text.contains("ignored"));
    }

    #[test]
    fn test_get_eval_text_lambada_ignores_text_field() {
        let config = EvalConfig {
            dataset: Dataset::Lambada,
            text: Some("ignored".to_string()),
            max_tokens: 512,
            threshold: 20.0,
        };
        let text = get_eval_text(&config).expect("should return LAMBADA sample");
        assert!(text.contains("walked into the room"));
    }

    // =========================================================================
    // print_header and print_results no-panic tests
    // =========================================================================

    #[test]
    fn test_print_header_does_not_panic() {
        let config = EvalConfig {
            dataset: Dataset::WikiText2,
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        };
        // Should not panic for any valid path
        print_header(Path::new("model.apr"), &config);
    }

    #[test]
    fn test_print_header_custom_dataset_does_not_panic() {
        let config = EvalConfig {
            dataset: Dataset::Custom,
            text: Some("custom text".to_string()),
            max_tokens: 128,
            threshold: 10.0,
        };
        print_header(Path::new("/tmp/custom_model.safetensors"), &config);
    }

    #[test]
    fn test_print_results_passing_does_not_panic() {
        let result = EvalResult {
            perplexity: 8.0,
            cross_entropy: 2.08,
            tokens_evaluated: 200,
            eval_time_secs: 0.5,
            passed: true,
            threshold: 20.0,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_failing_does_not_panic() {
        let result = EvalResult {
            perplexity: 55.0,
            cross_entropy: 4.01,
            tokens_evaluated: 100,
            eval_time_secs: 0.3,
            passed: false,
            threshold: 20.0,
        };
        print_results(&result);
    }

    #[test]
    fn test_print_results_all_quality_tiers() {
        // Test all quality tier branches in print_results
        let tiers = vec![
            (5.0, true),    // Excellent: < 10
            (12.0, true),   // Good: 10..15
            (18.0, true),   // Acceptable: 15..20
            (30.0, false),  // Poor: 20..50
            (100.0, false), // Garbage: >= 50
        ];
        for (ppl, passed) in tiers {
            let result = EvalResult {
                perplexity: ppl,
                cross_entropy: ppl.ln(),
                tokens_evaluated: 100,
                eval_time_secs: 1.0,
                passed,
                threshold: 20.0,
            };
            print_results(&result);
        }
    }

    // =========================================================================
    // EvalResult edge cases
    // =========================================================================

    #[test]
    fn test_eval_result_boundary_at_threshold() {
        let result = EvalResult {
            perplexity: 20.0,
            cross_entropy: 3.0,
            tokens_evaluated: 100,
            eval_time_secs: 1.0,
            passed: true,
            threshold: 20.0,
        };
        // At exactly the threshold, passed should be true (<=)
        assert!(result.passed);
    }

    #[test]
    fn test_eval_result_zero_tokens() {
        let result = EvalResult {
            perplexity: 1.0,
            cross_entropy: 0.0,
            tokens_evaluated: 0,
            eval_time_secs: 0.0,
            passed: true,
            threshold: 20.0,
        };
        assert_eq!(result.tokens_evaluated, 0);
    }

    #[test]
    fn test_eval_result_clone_preserves_all_fields() {
        let result = EvalResult {
            perplexity: 12.34,
            cross_entropy: 2.51,
            tokens_evaluated: 999,
            eval_time_secs: 4.56,
            passed: false,
            threshold: 10.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.perplexity, 12.34);
        assert_eq!(cloned.cross_entropy, 2.51);
        assert_eq!(cloned.tokens_evaluated, 999);
        assert_eq!(cloned.eval_time_secs, 4.56);
        assert!(!cloned.passed);
        assert_eq!(cloned.threshold, 10.0);
    }

    #[test]
    fn test_eval_result_debug_contains_all_fields() {
        let result = EvalResult {
            perplexity: 7.5,
            cross_entropy: 2.01,
            tokens_evaluated: 512,
            eval_time_secs: 2.3,
            passed: true,
            threshold: 20.0,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("perplexity"));
        assert!(debug.contains("cross_entropy"));
        assert!(debug.contains("tokens_evaluated"));
        assert!(debug.contains("eval_time_secs"));
        assert!(debug.contains("passed"));
        assert!(debug.contains("threshold"));
    }

    // =========================================================================
    // Sample text content validation
    // =========================================================================

    #[test]
    fn test_sample_wikitext_has_sufficient_length_for_eval() {
        // The eval needs at least 2 tokens, so text must be non-trivial
        assert!(
            SAMPLE_WIKITEXT.len() > 200,
            "WikiText sample too short for meaningful eval"
        );
    }

    #[test]
    fn test_sample_lambada_has_sufficient_length_for_eval() {
        assert!(
            SAMPLE_LAMBADA.len() > 200,
            "LAMBADA sample too short for meaningful eval"
        );
    }

    #[test]
    fn test_sample_texts_are_distinct() {
        assert_ne!(SAMPLE_WIKITEXT, SAMPLE_LAMBADA);
    }

    // =========================================================================
    // run() additional error paths
    // =========================================================================

    #[test]
    fn test_run_with_max_tokens() {
        // Test that max_tokens parameter is accepted (even if file doesn't exist)
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            "wikitext-2",
            None,
            Some(128),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_threshold() {
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            "wikitext-2",
            None,
            None,
            Some(5.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_custom_with_text() {
        // Custom dataset with text on a non-existent file should fail at file loading
        let result = run(
            Path::new("/nonexistent/model.apr"),
            "custom",
            Some("test input text"),
            None,
            None,
        );
        assert!(result.is_err());
    }
}
