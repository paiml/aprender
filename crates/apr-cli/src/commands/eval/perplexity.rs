//! Perplexity evaluation for GGUF, APR, and SafeTensors models.
//!
//! Implements spec H13: Perplexity evaluation on standard datasets.

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::Path;
use std::time::Instant;

use super::{get_eval_text, EvalConfig, EvalResult};

pub(crate) fn run_evaluation(path: &Path, config: &EvalConfig, json: bool) -> Result<EvalResult> {
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
        model.config().num_layers,
        model.config().vocab_size
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
    let (perplexity, cross_entropy) = super::calculate_gguf_perplexity(&model, &tokens)?;
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
    let tokens = super::tokenize_for_eval(path, &eval_text)?;
    let tokens: Vec<u32> = if tokens.len() > config.max_tokens {
        tokens[..config.max_tokens].to_vec()
    } else {
        tokens
    };
    super::validate_token_count(&tokens)?;

    progress!(
        "{}",
        format!("Calculating perplexity on {} tokens...", tokens.len()).yellow()
    );

    let eval_start = Instant::now();
    let vocab_size = transformer.config.vocab_size;
    let mut cache = AprKVCache::new(&transformer.config);
    let (perplexity, cross_entropy) =
        super::calculate_apr_perplexity(&transformer, &mut cache, &tokens, vocab_size)?;
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

/// GH-242: SafeTensors evaluation using realizar's SafeTensors->AprTransformer path
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
    let tokens = super::tokenize_for_eval(path, &eval_text)?;
    let tokens: Vec<u32> = if tokens.len() > config.max_tokens {
        tokens[..config.max_tokens].to_vec()
    } else {
        tokens
    };
    super::validate_token_count(&tokens)?;

    progress!(
        "{}",
        format!("Calculating perplexity on {} tokens...", tokens.len()).yellow()
    );

    let eval_start = Instant::now();
    let vocab_size = transformer.config.vocab_size;
    let mut cache = AprKVCache::new(&transformer.config);
    let (perplexity, cross_entropy) =
        super::calculate_apr_perplexity(&transformer, &mut cache, &tokens, vocab_size)?;
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
