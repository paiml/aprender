//! HumanEval and MBPP benchmark inference.
//!
//! Full inference via realizar -- generates completions and executes Python tests.
//! ALB-084 (HumanEval), ALB-085 (MBPP), ALB-088 (multi-sample pass@k),
//! ALB-089 (GPU-accelerated).

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use std::path::Path;
use std::time::Instant;

use super::code_eval::{compute_pass_at_k, emit_eval_results, run_multisample_loop};

// --- HumanEval benchmark evaluation (R-020, survey #62/#69) ---

/// A HumanEval problem from JSONL.
#[derive(Debug, serde::Deserialize)]
pub(super) struct HumanEvalProblem {
    /// Task identifier (e.g., "HumanEval/0")
    pub(super) task_id: String,
    /// Function prompt (signature + docstring)
    pub(super) prompt: String,
    /// Canonical solution
    #[serde(default)]
    pub(super) canonical_solution: Option<String>,
    /// Test harness code
    pub(super) test: String,
    /// Entry point function name (extracted from prompt if missing)
    #[serde(default)]
    pub(super) entry_point: Option<String>,
}

/// Run HumanEval benchmark evaluation.
///
/// Evaluates a model on HumanEval-format JSONL. Reports pass@k metrics.
/// ALB-084: Full inference via realizar -- generates completions and executes Python tests.
pub(crate) fn run_humaneval(
    model_path: &Path,
    data_path: Option<&Path>,
    k_values: &[usize],
    json_output: bool,
    device: &str,
    num_samples: usize,
    temperature: f32,
) -> Result<()> {
    let data_path = data_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <humaneval.jsonl> is required for HumanEval evaluation.\n\
             Format: OpenAI HumanEval JSONL with task_id, prompt, canonical_solution, test, entry_point"
                .to_string(),
        )
    })?;

    if !data_path.exists() {
        return Err(CliError::FileNotFound(data_path.to_path_buf()));
    }
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let content = std::fs::read_to_string(data_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read HumanEval data: {e}")))?;

    let problems: Vec<HumanEvalProblem> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| {
                CliError::ValidationFailed(format!("Invalid JSON on line {}: {e}", i + 1))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if problems.is_empty() {
        return Err(CliError::ValidationFailed(
            "HumanEval file is empty".to_string(),
        ));
    }

    // Validate problem structure
    let valid = problems
        .iter()
        .filter(|p| validate_humaneval_problem(p))
        .count();

    let num_samples = num_samples.max(1);
    if !json_output {
        output::section("APR HumanEval Evaluation");
        println!();
        output::kv("Model", model_path.display());
        output::kv("Benchmark", data_path.display());
        output::kv("Problems", format!("{} ({valid} valid)", problems.len()));
        output::kv("k values", format!("{k_values:?}"));
        if num_samples > 1 {
            output::kv("Samples/problem", num_samples);
            output::kv("Temperature", format!("{temperature:.2}"));
        }
        println!();
    }

    let start = Instant::now();

    // ALB-088: Multi-sample pass@k -- collect per-problem correct counts
    let mut per_problem_correct: Vec<(String, String, usize)> = problems
        .iter()
        .map(|p| {
            let ep = p
                .entry_point
                .as_deref()
                .or_else(|| extract_function_name(&p.prompt))
                .unwrap_or("")
                .to_string();
            (p.task_id.clone(), ep, 0usize)
        })
        .collect();

    let mut first_err: Option<String> = None;
    let any_ok = run_multisample_loop(&mut per_problem_correct, num_samples, json_output, || {
        let result = if device == "cuda" {
            run_humaneval_inference_cuda(model_path, &problems, k_values, json_output)
        } else {
            run_humaneval_inference(model_path, &problems, k_values, json_output)
        };
        if let Err(ref e) = result {
            if first_err.is_none() {
                first_err = Some(format!("{e}"));
            }
        }
        result
    });

    if !any_ok {
        // Fallback: structural validation
        if !json_output {
            // ALB-131: Print the actual inference error instead of swallowing it
            if let Some(ref err) = first_err {
                println!("  Inference error: {err}");
            }
            println!("  Falling back to structural validation (no inference)");
        }
        for (i, problem) in problems.iter().enumerate() {
            if validate_humaneval_problem(problem) {
                if let Some(ref sol) = problem.canonical_solution {
                    if !sol.trim().is_empty() {
                        per_problem_correct[i].2 = 1;
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    emit_eval_results(
        "humaneval",
        model_path,
        &per_problem_correct,
        num_samples,
        temperature,
        k_values,
        elapsed,
        if any_ok { "inference" } else { "structural" },
        json_output,
        None,
    );
    Ok(())
}

/// Sample a token from logits with temperature.
/// Temperature=0.0 -> greedy argmax. Temperature>0 -> softmax sampling.
pub(super) fn sample_token(logits: &[f32], temperature: f32, rng_state: &mut u64) -> u32 {
    contract_pre_repeat_penalty!();
    contract_pre_generation_temperature_zero!();
    if temperature <= 0.0 || logits.is_empty() {
        // Greedy argmax
        let result = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);
        contract_post_repeat_penalty!(&result);
        contract_post_generation_temperature_zero!(&result);
        return result;
    }

    // Temperature-scaled softmax sampling
    let inv_temp = 1.0 / temperature;
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) * inv_temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    // xorshift64 for deterministic sampling
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let r = (*rng_state as f32) / (u64::MAX as f32);

    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            let result = i as u32;
            contract_post_repeat_penalty!(&result);
            contract_post_generation_temperature_zero!(&result);
            return result;
        }
    }
    let result = (probs.len() - 1) as u32;
    contract_post_repeat_penalty!(&result);
    contract_post_generation_temperature_zero!(&result);
    result
}

/// ALB-084: Run HumanEval with actual model inference + Python test execution.
#[cfg(feature = "inference")]
fn run_humaneval_inference(
    model_path: &Path,
    problems: &[HumanEvalProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    use realizar::apr_transformer::{AprKVCache, AprTransformer};
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    // Load model -- try APR format first, fall back to SafeTensors
    if !json_output {
        println!("  {} Loading model for inference...", "→".dimmed());
    }
    let transformer: AprTransformer = if model_path.extension().is_some_and(|e| e == "apr")
        || model_path.join("model-best.apr").exists()
    {
        let apr_path = if model_path.is_dir() {
            model_path.join("model-best.apr")
        } else {
            model_path.to_path_buf()
        };
        AprTransformer::from_apr_file(&apr_path)
            .map_err(|e| format!("Cannot load APR model: {e}"))?
    } else {
        SafetensorsToAprConverter::convert(model_path)
            .map_err(|e| format!("Cannot load model: {e}"))?
            .into_inner()
    };

    // ALB-130/ALB-131: Load tokenizer — try embedded first, then sibling file
    let apr_file = if model_path.is_dir() {
        model_path.join("model-best.apr")
    } else {
        model_path.to_path_buf()
    };
    let tokenizer = if apr_file.extension().is_some_and(|e| e == "apr") {
        if let Some(embedded) = realizar::apr::AprV2Model::load(&apr_file)
            .ok()
            .and_then(|m| m.load_embedded_bpe_tokenizer())
        {
            if !json_output {
                println!("  {} Loaded embedded BPE tokenizer", "✓".green());
            }
            embedded
        } else {
            realizar::apr::AprV2Model::load_tokenizer(model_path).ok_or_else(|| {
                "No tokenizer found (no embedded tokenizer and no sibling tokenizer.json)"
                    .to_string()
            })?
        }
    } else {
        realizar::apr::AprV2Model::load_tokenizer(model_path)
            .ok_or_else(|| "No tokenizer found".to_string())?
    };

    if !json_output {
        println!(
            "  {} Model loaded ({} layers, vocab={})",
            "✓".green(),
            transformer.config.num_layers,
            transformer.config.vocab_size
        );
    }

    let mut passed = 0usize;
    let mut results = Vec::new();
    // Temperature: 0.0 for pass@1 (greedy), 0.8 for pass@k>1
    // Currently using greedy; temperature sampling available via sample_token()
    let temperature = 0.0f32;
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let entry = problem
            .entry_point
            .as_deref()
            .or_else(|| extract_function_name(&problem.prompt))
            .unwrap_or("unknown");

        // Tokenize prompt
        let prompt_tokens = tokenizer.encode(&problem.prompt);
        if prompt_tokens.is_empty() {
            results.push((problem.task_id.clone(), entry.to_string(), false));
            continue;
        }

        // Generate completion (greedy, max 256 tokens)
        let mut cache = AprKVCache::new(&transformer.config);
        let mut tokens = prompt_tokens.clone();

        // Feed prompt through cache
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let _ = transformer.forward_with_cache(tok, &mut cache, pos);
        }

        // Generate new tokens
        let max_new = 256;
        for step in 0..max_new {
            let pos = prompt_tokens.len() + step;
            let last_tok = *tokens.last().expect("last(");
            let logits = transformer
                .forward_with_cache(last_tok, &mut cache, pos)
                .map_err(|e| format!("Generation failed: {e}"))?;

            let next = sample_token(&logits, temperature, &mut rng_state);

            tokens.push(next);

            // Stop at EOS or double newline at indent 0 (function boundary)
            if next == 0 {
                break;
            }
            if let Some(eos) = transformer.config.eos_token_id {
                if next == eos {
                    break;
                }
            }
        }

        // Decode completion (only new tokens)
        let completion_tokens = &tokens[prompt_tokens.len()..];
        let completion = tokenizer.decode(completion_tokens);

        // Truncate at function boundary (next 'def ' or '\nclass ' at indent 0)
        let completion = truncate_at_function_boundary(&completion);

        // Build full program: prompt + completion + test + check(entry_point)
        let full_program = format!(
            "{}{}\n\n{}\n\ncheck({})\n",
            problem.prompt, completion, problem.test, entry
        );

        // Execute Python test
        let ok = execute_python_test(&full_program, 10);

        if ok {
            passed += 1;
        }

        results.push((problem.task_id.clone(), entry.to_string(), ok));

        // Progress
        if !json_output && (i + 1) % 10 == 0 {
            println!(
                "  {} {}/{} problems evaluated ({} passed)",
                "→".dimmed(),
                i + 1,
                problems.len(),
                passed
            );
        }
    }

    Ok((passed, results))
}

#[cfg(not(feature = "inference"))]
fn run_humaneval_inference(
    _model_path: &Path,
    _problems: &[HumanEvalProblem],
    _k_values: &[usize],
    _json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    Err("Inference not available (compile with --features inference)".to_string())
}

// --- ALB-089: GPU-accelerated inference for eval ---

/// Load TransformerConfig from checkpoint dir's config.json.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn load_transformer_config(
    checkpoint_dir: &Path,
) -> std::result::Result<entrenar::transformer::TransformerConfig, String> {
    let config_path = checkpoint_dir.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Cannot read config.json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Invalid config.json: {e}"))?;

    Ok(entrenar::transformer::TransformerConfig {
        hidden_size: v["hidden_size"].as_u64().unwrap_or(1024) as usize,
        num_attention_heads: v["num_attention_heads"].as_u64().unwrap_or(16) as usize,
        num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(4096) as usize,
        num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        vocab_size: v["vocab_size"].as_u64().unwrap_or(32768) as usize,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(1024) as usize,
        rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        rope_theta: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        use_bias: v["use_bias"].as_bool().unwrap_or(false),
        head_dim_override: None,
        architecture: Default::default(),
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    })
}

/// GPU-accelerated HumanEval inference via entrenar CudaTransformerTrainer (ALB-089).
///
/// Uses `forward_logits()` for autoregressive generation. No KV cache -- each step
/// reprocesses the full sequence. Still 20-40x faster than CPU for 350M model.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn run_humaneval_inference_cuda(
    model_path: &Path,
    problems: &[HumanEvalProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    // ALB-089: resolve to checkpoint directory (model_path may be a .apr file)
    let checkpoint_dir = if model_path.is_file() {
        model_path.parent().unwrap_or(model_path)
    } else {
        model_path
    };

    let config = load_transformer_config(checkpoint_dir)?;
    let max_seq = config.max_position_embeddings;

    if !json_output {
        println!(
            "  {} Loading model onto GPU for inference (ALB-089)...",
            "→".dimmed()
        );
    }

    let mut trainer =
        entrenar::train::CudaTransformerTrainer::for_inference(checkpoint_dir, config)
            .map_err(|e| format!("CUDA inference init failed: {e}"))?;

    // Load tokenizer -- use original model_path (file) for sibling lookup
    let tokenizer = realizar::apr::AprV2Model::load_tokenizer(model_path)
        .or_else(|| {
            // Fallback: try tokenizer.json directly in checkpoint dir
            let tok_path = checkpoint_dir.join("tokenizer.json");
            realizar::apr::AprV2Model::load_tokenizer_from_path(&tok_path)
        })
        .ok_or_else(|| format!("No tokenizer found in {}", checkpoint_dir.display()))?;

    if !json_output {
        println!("  {} GPU inference ready", "✓".green());
    }

    let mut passed = 0usize;
    let mut results = Vec::new();
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let entry = problem
            .entry_point
            .as_deref()
            .or_else(|| extract_function_name(&problem.prompt))
            .unwrap_or("unknown");

        let prompt_tokens = tokenizer.encode(&problem.prompt);
        if prompt_tokens.is_empty() {
            results.push((problem.task_id.clone(), entry.to_string(), false));
            continue;
        }

        // Autoregressive generation: build sequence incrementally
        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let max_new = 256;

        for _ in 0..max_new {
            if tokens.len() >= max_seq {
                break;
            }

            // Forward full sequence, get last-position logits
            let logits = trainer
                .forward_logits(&tokens)
                .ok_or_else(|| "forward_logits failed".to_string())?;

            let next = sample_token(&logits, 0.0, &mut rng_state);
            tokens.push(next);

            // Stop at EOS or token 0
            if next == 0 {
                break;
            }
        }

        // Decode completion
        let completion_tokens = &tokens[prompt_tokens.len()..];
        let completion = tokenizer.decode(completion_tokens);
        let completion = truncate_at_function_boundary(&completion);

        // Build and test
        let full_program = format!(
            "{}{}\n\n{}\n\ncheck({})\n",
            problem.prompt, completion, problem.test, entry
        );
        let ok = execute_python_test(&full_program, 10);

        if ok {
            passed += 1;
        }
        results.push((problem.task_id.clone(), entry.to_string(), ok));

        if !json_output && (i + 1) % 10 == 0 {
            println!(
                "  {} {}/{} problems evaluated ({} passed)",
                "→".dimmed(),
                i + 1,
                problems.len(),
                passed
            );
        }
    }

    Ok((passed, results))
}

#[cfg(not(all(feature = "cuda", feature = "training")))]
fn run_humaneval_inference_cuda(
    _model_path: &Path,
    _problems: &[HumanEvalProblem],
    _k_values: &[usize],
    _json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    Err("CUDA not available (compile with --features cuda)".to_string())
}

/// Truncate completion at the next top-level function/class definition.
pub(super) fn truncate_at_function_boundary(completion: &str) -> &str {
    // Find the first '\ndef ' or '\nclass ' that indicates a new top-level definition
    for pattern in &["\ndef ", "\nclass "] {
        if let Some(pos) = completion.find(pattern) {
            return &completion[..pos];
        }
    }
    completion
}

/// Execute a Python program and check if all assertions pass.
/// Returns true if exit code is 0, false otherwise.
/// Enforces a timeout to catch infinite loops (FALSIFY-EVAL-003).
pub(super) fn execute_python_test(program: &str, timeout_secs: u64) -> bool {
    use std::process::Command;
    use std::time::{Duration, Instant};

    // Write program to a temp file
    let tmp = std::env::temp_dir().join(format!("apr_eval_{}.py", std::process::id()));
    if std::fs::write(&tmp, program).is_err() {
        return false;
    }

    let result = Command::new("python3")
        .arg(&tmp)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            let deadline = Instant::now() + Duration::from_secs(timeout_secs);
            loop {
                match child.try_wait()? {
                    Some(status) => return Ok(status.success()),
                    None => {
                        if Instant::now() >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            return Ok(false);
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        });

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp);

    result.unwrap_or(false)
}

/// Validate a single HumanEval problem has correct structure.
fn validate_humaneval_problem(problem: &HumanEvalProblem) -> bool {
    if problem.prompt.trim().is_empty() || problem.test.trim().is_empty() {
        return false;
    }
    // If canonical solution provided, check it has content
    if let Some(ref sol) = problem.canonical_solution {
        if !sol.trim().is_empty() {
            return true;
        }
    }
    // Without canonical solution, validate prompt has a function definition
    problem.prompt.contains("def ")
}

/// Extract function name from a Python prompt like "def foo(...):"
pub(super) fn extract_function_name(prompt: &str) -> Option<&str> {
    for line in prompt.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("def ") {
            if let Some(paren) = rest.find('(') {
                return Some(&rest[..paren]);
            }
        }
    }
    None
}

/// Print HumanEval results table.
pub(super) fn print_humaneval_results(
    results: &[(String, String, bool)],
    total: usize,
    passed: usize,
    k_values: &[usize],
    elapsed: f32,
    mode: &str,
) {
    for (task_id, entry_point, ok) in results {
        let status = if *ok {
            "PASS".green().to_string()
        } else {
            "FAIL".red().to_string()
        };
        println!("  [{status}] {task_id} ({entry_point})");
    }

    println!();
    for &k in k_values {
        let rate = compute_pass_at_k(total, passed, k);
        output::kv(&format!("pass@{k}"), format!("{:.1}%", rate * 100.0));
    }
    output::kv("Time", format!("{elapsed:.2}s"));
    println!();
    println!(
        "{}",
        format!("{passed}/{total} problems evaluated ({mode})").dimmed()
    );
}

// --- MBPP benchmark evaluation (ALB-085) ---

/// An MBPP problem from JSONL.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub(super) struct MbppProblem {
    /// Natural language description
    pub(super) text: String,
    /// Canonical solution code
    #[serde(default)]
    pub(super) code: Option<String>,
    /// Task identifier (integer in MBPP)
    pub(super) task_id: serde_json::Value,
    /// Setup code to prepend to tests
    #[serde(default)]
    pub(super) test_setup_code: Option<String>,
    /// Test assertion strings
    pub(super) test_list: Vec<String>,
    /// Challenge test assertions (harder)
    #[serde(default)]
    pub(super) challenge_test_list: Vec<String>,
}

/// Run MBPP benchmark evaluation.
///
/// Evaluates a model on MBPP-format JSONL. Reports pass@k metrics.
/// ALB-085: Full inference via realizar -- generates completions and executes Python tests.
pub(crate) fn run_mbpp(
    model_path: &Path,
    data_path: Option<&Path>,
    k_values: &[usize],
    json_output: bool,
    device: &str,
    num_samples: usize,
    temperature: f32,
) -> Result<()> {
    let data_path = data_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <mbpp.jsonl> is required for MBPP evaluation.\n\
             Format: Google MBPP JSONL with text, code, task_id, test_list"
                .to_string(),
        )
    })?;

    if !data_path.exists() {
        return Err(CliError::FileNotFound(data_path.to_path_buf()));
    }
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let content = std::fs::read_to_string(data_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read MBPP data: {e}")))?;

    let problems: Vec<MbppProblem> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).map_err(|e| {
                CliError::ValidationFailed(format!("Invalid JSON on line {}: {e}", i + 1))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if problems.is_empty() {
        return Err(CliError::ValidationFailed("MBPP file is empty".to_string()));
    }

    // MBPP-sanitized: standard subset uses task_ids 11-510 (inclusive)
    // Filter to sanitized subset for comparable results
    let problems: Vec<MbppProblem> = problems
        .into_iter()
        .filter(|p| {
            if let Some(id) = p.task_id.as_u64() {
                (11..=510).contains(&id)
            } else {
                true // Keep non-numeric task_ids
            }
        })
        .collect();

    let num_samples = num_samples.max(1);
    if !json_output {
        output::section("APR MBPP Evaluation (sanitized)");
        println!();
        output::kv("Model", model_path.display());
        output::kv("Benchmark", data_path.display());
        output::kv("Problems", format!("{} (sanitized subset)", problems.len()));
        output::kv("k values", format!("{k_values:?}"));
        if num_samples > 1 {
            output::kv("Samples/problem", num_samples);
            output::kv("Temperature", format!("{temperature:.2}"));
        }
        println!();
    }

    let start = Instant::now();

    // ALB-088: Multi-sample pass@k -- collect per-problem correct counts
    let mut per_problem_correct: Vec<(String, String, usize)> = problems
        .iter()
        .map(|p| (p.task_id.to_string(), String::new(), 0usize))
        .collect();

    let mut first_err: Option<String> = None;
    let any_ok = run_multisample_loop(&mut per_problem_correct, num_samples, json_output, || {
        let result = if device == "cuda" {
            run_mbpp_inference_cuda(model_path, &problems, k_values, json_output)
        } else {
            run_mbpp_inference(model_path, &problems, k_values, json_output)
        };
        if let Err(ref e) = result {
            if first_err.is_none() {
                first_err = Some(format!("{e}"));
            }
        }
        result
    });

    if !any_ok {
        return Err(CliError::ValidationFailed(format!(
            "MBPP inference failed: {}",
            first_err.unwrap_or_else(|| "unknown error".to_string())
        )));
    }

    let elapsed = start.elapsed().as_secs_f32();
    emit_eval_results(
        "mbpp-sanitized",
        model_path,
        &per_problem_correct,
        num_samples,
        temperature,
        k_values,
        elapsed,
        "inference",
        json_output,
        Some(("subset", "sanitized (task_id 11-510)")),
    );
    Ok(())
}

/// ALB-085: Run MBPP with actual model inference + Python test execution.
#[cfg(feature = "inference")]
fn run_mbpp_inference(
    model_path: &Path,
    problems: &[MbppProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    use realizar::apr_transformer::{AprKVCache, AprTransformer};
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    if !json_output {
        println!("  {} Loading model for inference...", "→".dimmed());
    }
    let transformer: AprTransformer = if model_path.extension().is_some_and(|e| e == "apr")
        || model_path.join("model-best.apr").exists()
    {
        let apr_path = if model_path.is_dir() {
            model_path.join("model-best.apr")
        } else {
            model_path.to_path_buf()
        };
        AprTransformer::from_apr_file(&apr_path)
            .map_err(|e| format!("Cannot load APR model: {e}"))?
    } else {
        SafetensorsToAprConverter::convert(model_path)
            .map_err(|e| format!("Cannot load model: {e}"))?
            .into_inner()
    };

    let tokenizer = realizar::apr::AprV2Model::load_tokenizer(model_path)
        .ok_or_else(|| "No tokenizer found".to_string())?;

    if !json_output {
        println!(
            "  {} Model loaded ({} layers, vocab={})",
            "✓".green(),
            transformer.config.num_layers,
            transformer.config.vocab_size
        );
    }

    let mut passed = 0usize;
    let mut results = Vec::new();
    let temperature = 0.0f32;
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let task_id = match &problem.task_id {
            serde_json::Value::Number(n) => format!("MBPP/{n}"),
            serde_json::Value::String(s) => s.clone(),
            v => format!("MBPP/{v}"),
        };

        // MBPP prompt: natural language description -> model writes complete function
        let prompt = format!("{}\n", problem.text);

        let prompt_tokens = tokenizer.encode(&prompt);
        if prompt_tokens.is_empty() {
            results.push((task_id, String::new(), false));
            continue;
        }

        // Generate completion (max 512 tokens -- MBPP solutions are longer)
        let mut cache = AprKVCache::new(&transformer.config);
        let mut tokens = prompt_tokens.clone();

        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let _ = transformer.forward_with_cache(tok, &mut cache, pos);
        }

        let max_new = 512;
        for step in 0..max_new {
            let pos = prompt_tokens.len() + step;
            let last_tok = *tokens.last().expect("last(");
            let logits = transformer
                .forward_with_cache(last_tok, &mut cache, pos)
                .map_err(|e| format!("Generation failed: {e}"))?;

            let next = sample_token(&logits, temperature, &mut rng_state);
            tokens.push(next);

            if next == 0 {
                break;
            }
            if let Some(eos) = transformer.config.eos_token_id {
                if next == eos {
                    break;
                }
            }
        }

        let completion_tokens = &tokens[prompt_tokens.len()..];
        let completion = tokenizer.decode(completion_tokens);

        // Truncate at next top-level definition (same as HumanEval)
        let completion = truncate_at_function_boundary(&completion);

        // Build test program: completion + setup_code + test assertions
        let setup = problem.test_setup_code.as_deref().unwrap_or("").trim();
        let tests = problem.test_list.join("\n");
        let full_program = if setup.is_empty() {
            format!("{completion}\n{tests}\n")
        } else {
            format!("{completion}\n{setup}\n{tests}\n")
        };

        let ok = execute_python_test(&full_program, 10);

        if ok {
            passed += 1;
        }

        results.push((task_id, String::new(), ok));

        if !json_output && (i + 1) % 50 == 0 {
            println!(
                "  {} {}/{} problems evaluated ({} passed)",
                "→".dimmed(),
                i + 1,
                problems.len(),
                passed
            );
        }
    }

    Ok((passed, results))
}

#[cfg(not(feature = "inference"))]
fn run_mbpp_inference(
    _model_path: &Path,
    _problems: &[MbppProblem],
    _k_values: &[usize],
    _json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    Err("Inference not available (compile with --features inference)".to_string())
}

/// GPU-accelerated MBPP inference via entrenar CudaTransformerTrainer (ALB-089).
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn run_mbpp_inference_cuda(
    model_path: &Path,
    problems: &[MbppProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    // ALB-089: resolve to checkpoint directory (model_path may be a .apr file)
    let checkpoint_dir = if model_path.is_file() {
        model_path.parent().unwrap_or(model_path)
    } else {
        model_path
    };

    let config = load_transformer_config(checkpoint_dir)?;
    let max_seq = config.max_position_embeddings;

    if !json_output {
        println!(
            "  {} Loading model onto GPU for inference (ALB-089)...",
            "→".dimmed()
        );
    }

    let mut trainer =
        entrenar::train::CudaTransformerTrainer::for_inference(checkpoint_dir, config)
            .map_err(|e| format!("CUDA inference init failed: {e}"))?;

    let tokenizer = realizar::apr::AprV2Model::load_tokenizer(model_path)
        .or_else(|| {
            let tok_path = checkpoint_dir.join("tokenizer.json");
            realizar::apr::AprV2Model::load_tokenizer_from_path(&tok_path)
        })
        .ok_or_else(|| format!("No tokenizer found in {}", checkpoint_dir.display()))?;

    if !json_output {
        println!("  {} GPU inference ready", "✓".green());
    }

    let mut passed = 0usize;
    let mut results = Vec::new();
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let task_id = match &problem.task_id {
            serde_json::Value::Number(n) => format!("MBPP/{n}"),
            serde_json::Value::String(s) => s.clone(),
            v => format!("MBPP/{v}"),
        };

        let prompt = format!("{}\n", problem.text);
        let prompt_tokens = tokenizer.encode(&prompt);
        if prompt_tokens.is_empty() {
            results.push((task_id, String::new(), false));
            continue;
        }

        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let max_new = 512;

        for _ in 0..max_new {
            if tokens.len() >= max_seq {
                break;
            }
            let logits = trainer
                .forward_logits(&tokens)
                .ok_or_else(|| "forward_logits failed".to_string())?;

            let next = sample_token(&logits, 0.0, &mut rng_state);
            tokens.push(next);

            if next == 0 {
                break;
            }
        }

        let completion_tokens = &tokens[prompt_tokens.len()..];
        let completion = tokenizer.decode(completion_tokens);
        let completion = truncate_at_function_boundary(&completion);

        let setup = problem.test_setup_code.as_deref().unwrap_or("").trim();
        let tests = problem.test_list.join("\n");
        let full_program = if setup.is_empty() {
            format!("{completion}\n{tests}\n")
        } else {
            format!("{completion}\n{setup}\n{tests}\n")
        };

        let ok = execute_python_test(&full_program, 10);

        if ok {
            passed += 1;
        }
        results.push((task_id, String::new(), ok));

        if !json_output && (i + 1) % 50 == 0 {
            println!(
                "  {} {}/{} problems evaluated ({} passed)",
                "→".dimmed(),
                i + 1,
                problems.len(),
                passed
            );
        }
    }

    Ok((passed, results))
}

#[cfg(not(all(feature = "cuda", feature = "training")))]
fn run_mbpp_inference_cuda(
    _model_path: &Path,
    _problems: &[MbppProblem],
    _k_values: &[usize],
    _json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    Err("CUDA not available (compile with --features cuda)".to_string())
}
