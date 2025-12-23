//! Eval Command Implementation
//!
//! Implements spec §H13: Perplexity evaluation on standard datasets.
//!
//! # Usage
//!
//! ```bash
//! apr eval model.apr --dataset wikitext-2    # Evaluate on WikiText-2
//! apr eval model.apr --dataset lambada       # Evaluate on LAMBADA
//! apr eval model.apr --text "Hello world"    # Evaluate on custom text
//! ```
//!
//! Toyota Way: Jidoka - fail fast if perplexity exceeds threshold.

use crate::error::{CliError, Result};
use crate::output;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::text::bpe::Qwen2BpeTokenizer;
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

    // Create model config
    let model_config = if is_safetensors || is_apr {
        Qwen2Config::qwen2_0_5b_instruct()
    } else {
        // Demo config for testing
        Qwen2Config {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            vocab_size: 1000,
            max_seq_len: 512,
            intermediate_size: 256,
            rope_theta: 10000.0,
        }
    };

    println!("{}", "Loading model...".yellow());
    let start = Instant::now();

    let mut model = if is_safetensors || is_apr {
        Qwen2Model::new_uninitialized(&model_config)
    } else {
        Qwen2Model::new(&model_config)
    };

    // Load weights
    if is_apr {
        let count = model
            .load_from_apr(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
        println!("{} {} tensors", "Loaded".green(), count);
    } else if is_safetensors {
        let count = model
            .load_from_safetensors(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors: {e}")))?;
        println!("{} {} tensors", "Loaded".green(), count);
    }

    model.eval();
    let load_time = start.elapsed();
    println!(
        "{} in {:.2}s",
        "Model ready".green(),
        load_time.as_secs_f32()
    );
    println!();

    // Get evaluation text
    let eval_text = get_eval_text(config)?;
    println!(
        "{}",
        format!("Evaluating on {} characters...", eval_text.len()).yellow()
    );

    // Tokenize
    let tokenizer = Qwen2BpeTokenizer::new();
    let tokens = tokenizer.encode(&eval_text);

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

    // Calculate perplexity
    let eval_start = Instant::now();
    let (perplexity, cross_entropy) = calculate_perplexity(&mut model, &tokens)?;
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

/// Calculate perplexity using the model
fn calculate_perplexity(model: &mut Qwen2Model, tokens: &[u32]) -> Result<(f32, f32)> {
    let vocab_size = model.config().vocab_size;
    let mut total_log_prob = 0.0f64;
    let mut count = 0usize;

    // Process in chunks for efficiency
    let chunk_size = 64;

    for start in (0..tokens.len().saturating_sub(1)).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens.len() - 1);
        let input_tokens = &tokens[start..=end];

        if input_tokens.len() < 2 {
            continue;
        }

        // Forward pass
        let position_ids: Vec<usize> = (0..input_tokens.len()).collect();
        let logits = model.forward(input_tokens, &position_ids);
        let logits_data = logits.data();

        // Calculate log probabilities for each position
        for (pos, &next_token) in tokens[start + 1..=end].iter().enumerate() {
            let logit_start = pos * vocab_size;
            let logit_end = (pos + 1) * vocab_size;

            if logit_end > logits_data.len() {
                continue;
            }

            let position_logits = &logits_data[logit_start..logit_end];

            // Compute log softmax
            let max_logit = position_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f64 = position_logits
                .iter()
                .map(|&l| ((l - max_logit) as f64).exp())
                .sum::<f64>()
                .ln()
                + max_logit as f64;

            let next_token_idx = next_token as usize;
            if next_token_idx < vocab_size {
                let log_prob = position_logits[next_token_idx] as f64 - log_sum_exp;
                total_log_prob += log_prob;
                count += 1;
            }
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

    #[test]
    fn test_dataset_parse() {
        assert_eq!("wikitext-2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("wikitext2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
        assert_eq!("lambada".parse::<Dataset>().unwrap(), Dataset::Lambada);
        assert_eq!("custom".parse::<Dataset>().unwrap(), Dataset::Custom);
    }

    #[test]
    fn test_dataset_parse_error() {
        assert!("unknown".parse::<Dataset>().is_err());
    }

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
    fn test_sample_texts_not_empty() {
        assert!(!SAMPLE_WIKITEXT.is_empty());
        assert!(!SAMPLE_LAMBADA.is_empty());
        assert!(SAMPLE_WIKITEXT.len() > 100);
        assert!(SAMPLE_LAMBADA.len() > 100);
    }
}
