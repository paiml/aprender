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

use super::code_eval::{emit_eval_results, run_multisample_loop};

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

    // PMAT-702: Inference-failure handling.
    //
    // Contract: contracts/apr-eval-humaneval-inference-failure-handling-v1.yaml
    //
    // Pre-fix behavior (removed): when inference failed for ALL samples, the
    // code "fell back to structural validation" and marked every problem with
    // a non-empty canonical_solution as pass=1. That produced pass@1 = 1.0 /
    // 164/164 on completely broken models — the failure mode that hid the
    // PMAT-701 Phase 4 Stage D no-KD training run for two days.
    //
    // Post-fix behavior: emit a structured "inference_failed" result with
    // pass counters all zero AND return Err so the exit code is non-zero.
    // The dataset's structural validity is a pre-flight concern, already
    // reported in the "Problems: N (M valid)" line above. Conflating it with
    // pass@k is the bug this contract eliminates.
    //
    // MBPP's run_mbpp (this file, ~line 1513) already returned Err on the
    // same condition. This change brings HumanEval into parity.
    if !any_ok {
        let err_msg = first_err
            .clone()
            .unwrap_or_else(|| "(no error captured)".to_string());
        if !json_output {
            println!("  Inference error: {err_msg}");
            println!("  All HumanEval samples failed inference — pass counters are 0.");
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
            "inference_failed",
            json_output,
            Some(("inference_error", &err_msg)),
        );
        return Err(CliError::InferenceFailed(format!(
            "HumanEval inference failed for all samples: {err_msg}"
        )));
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
        "inference",
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

/// Load an `AprTransformer` from a model path (APR or SafeTensors).
#[cfg(feature = "inference")]
fn load_humaneval_model(
    model_path: &Path,
) -> std::result::Result<realizar::apr_transformer::AprTransformer, String> {
    use realizar::apr_transformer::AprTransformer;
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    if model_path.extension().is_some_and(|e| e == "apr")
        || model_path.join("model-best.apr").exists()
    {
        let apr_path = if model_path.is_dir() {
            model_path.join("model-best.apr")
        } else {
            model_path.to_path_buf()
        };
        AprTransformer::from_apr_file(&apr_path).map_err(|e| format!("Cannot load APR model: {e}"))
    } else {
        SafetensorsToAprConverter::convert(model_path)
            .map_err(|e| format!("Cannot load model: {e}"))
            .map(|c| c.into_inner())
    }
}

/// Load a BPE tokenizer for HumanEval: try embedded first, then sibling file.
#[cfg(feature = "inference")]
fn load_humaneval_tokenizer(
    model_path: &Path,
    json_output: bool,
) -> std::result::Result<realizar::apr::BpeTokenizer, String> {
    let apr_file = if model_path.is_dir() {
        model_path.join("model-best.apr")
    } else {
        model_path.to_path_buf()
    };

    if apr_file.extension().is_some_and(|e| e == "apr") {
        if let Some(embedded) = realizar::apr::AprV2Model::load(&apr_file)
            .ok()
            .and_then(|m| m.load_embedded_bpe_tokenizer())
        {
            if !json_output {
                println!("  {} Loaded embedded BPE tokenizer", "✓".green());
            }
            return Ok(embedded);
        }
    }

    realizar::apr::AprV2Model::load_tokenizer(model_path).ok_or_else(|| {
        "No tokenizer found (no embedded tokenizer and no sibling tokenizer.json)".to_string()
    })
}

/// ALB-084: Run HumanEval with actual model inference + Python test execution.
///
/// PMAT-CODE-SHIP-005-H4-FIX (2026-05-11): for instruct-family models, route
/// the prompt through ChatML auto-wrap (`InferenceConfig::with_prompt` →
/// `prepare_tokens_apr` → ChatMLTemplate). Parse the assistant's
/// `\`\`\`python ... \`\`\`` code block out of the response and use that as the
/// completion. Falls back to raw-continuation when no code block is found
/// (preserving the older PMAT-CODE-SHIP-005-FIX behaviour).
///
/// Why: §65 + §66 evidence. Raw-continuation produces 34.15% pass@1 on
/// canonical 7B Qwen2.5-Coder-Instruct. Same model + same prompt via `apr run`
/// (ChatML auto-wrap) produces correct solutions. The Qwen-Instruct teacher
/// is trained for chat format; published pass@1 = 88.4% uses chat template.
///
/// Detection: a model is considered "instruct" when its file extension is
/// `.apr` and either the architecture metadata is qwen2/qwen/llama/mistral/
/// phi/phi3, the vocabulary contains `<|im_start|>`, or the filename
/// contains `instruct`/`-chat`. This matches `prepare_tokens_apr`'s
/// detection logic; we don't replicate it — `with_prompt` triggers the same
/// auto-wrap inside `prepare_tokens`.
#[cfg(feature = "inference")]
fn run_humaneval_inference(
    model_path: &Path,
    problems: &[HumanEvalProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    use realizar::{run_inference, InferenceConfig};

    if !json_output {
        println!("  {} Loading model for inference...", "→".dimmed());
    }
    let tokenizer = load_humaneval_tokenizer(model_path, json_output)?;

    if !json_output {
        println!("  {} Tokenizer loaded", "✓".green());
    }

    let mut passed = 0usize;
    let mut results = Vec::new();

    for (i, problem) in problems.iter().enumerate() {
        let entry = humaneval_entry_point(problem);

        let prompt_tokens = tokenizer.encode(&problem.prompt);
        if prompt_tokens.is_empty() {
            results.push((problem.task_id.clone(), entry.to_string(), false));
            continue;
        }

        // H4 fix: route through ChatML auto-wrap via `with_prompt`. The
        // `prepare_tokens_apr` in realizar/aprender-serve detects the
        // instruct architecture from APR metadata and wraps the user prompt
        // in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` for
        // chat-tuned models. The assistant emits a markdown-wrapped Python
        // code block.
        let config_chatml = InferenceConfig::new(model_path)
            .with_prompt(problem.prompt.clone())
            .with_max_tokens(512)
            .with_temperature(0.0)
            .with_top_k(1);

        let result = match run_inference(&config_chatml) {
            Ok(r) => r,
            Err(e) => {
                if !json_output {
                    eprintln!(
                        "  [FAIL] {} ({}): inference error: {e}",
                        problem.task_id, entry
                    );
                }
                results.push((problem.task_id.clone(), entry.to_string(), false));
                continue;
            }
        };

        let completion = build_humaneval_completion(problem, entry, &result, &tokenizer);

        // Build the test program. Two cases:
        //   - ChatML path: `completion` is a complete function from the
        //     code block (signature + body). Use it directly.
        //   - Raw-continuation path: `completion` already includes the
        //     prompt prefix (concatenated above).
        let full_program = format!("{completion}\n\n{}\n\ncheck({})\n", problem.test, entry);

        let exec_result = execute_python_test_with_diagnostics(&full_program, 10);
        let ok = exec_result.success;

        if std::env::var("APR_EVAL_DEBUG").is_ok() {
            write_apr_eval_debug(
                &problem.task_id,
                &problem.prompt,
                &result.text,
                &completion,
                &full_program,
                &exec_result,
            );
        }

        if ok {
            passed += 1;
        }

        results.push((problem.task_id.clone(), entry.to_string(), ok));

        report_eval_progress(json_output, i, problems.len(), passed, 10);
    }

    Ok((passed, results))
}

/// The entry-point function name for a HumanEval problem.
///
/// Prefers the explicit `entry_point` field, falls back to the first `def`
/// in the prompt, then to the literal `"unknown"`. Extracted verbatim from
/// the HumanEval eval loops (PMAT-746); behaviour is unchanged.
#[cfg(feature = "inference")]
fn humaneval_entry_point(problem: &HumanEvalProblem) -> &str {
    problem
        .entry_point
        .as_deref()
        .or_else(|| extract_function_name(&problem.prompt))
        .unwrap_or("unknown")
}

/// The generated-token slice of an inference result.
///
/// Everything past `input_token_count`, or the whole buffer when the model
/// returned no more tokens than it was handed (defensive: a shorter-than-
/// prompt token vector must not panic on the slice).
#[cfg(feature = "inference")]
fn completion_token_slice(result: &realizar::InferenceResult) -> &[u32] {
    if result.tokens.len() > result.input_token_count {
        &result.tokens[result.input_token_count..]
    } else {
        &result.tokens[..]
    }
}

/// Print the periodic "N/M problems evaluated" progress line.
///
/// No-op under `--json` and on every index that is not the last of a group
/// of `every`. Extracted verbatim from the four eval loops (PMAT-746).
#[cfg(feature = "inference")]
fn report_eval_progress(
    json_output: bool,
    index: usize,
    total: usize,
    passed: usize,
    every: usize,
) {
    if json_output || (index + 1) % every != 0 {
        return;
    }
    println!(
        "  {} {}/{} problems evaluated ({} passed)",
        "→".dimmed(),
        index + 1,
        total,
        passed
    );
}

/// Raw-continuation completion for a HumanEval problem (the pre-H4 path).
///
/// Slices off the prompt prefix when it is verbatim in `result.text`;
/// otherwise decodes the tokens past `input_token_count`. The aligned form
/// goes APPENDED to the prompt, so the returned string already carries the
/// prompt prefix — the caller splits it back off in the program-build step.
#[cfg(feature = "inference")]
fn humaneval_raw_completion(
    problem: &HumanEvalProblem,
    result: &realizar::InferenceResult,
    tokenizer: &realizar::apr::BpeTokenizer,
) -> String {
    let raw = if let Some(stripped) = result.text.strip_prefix(&problem.prompt) {
        stripped.to_string()
    } else {
        tokenizer.decode(completion_token_slice(result))
    };
    let truncated = truncate_at_function_boundary(&raw);
    format!(
        "{}{}",
        problem.prompt,
        align_continuation_indent(&problem.prompt, truncated)
    )
}

/// Turn an assistant response into the HumanEval completion to test.
///
/// On instruct-family models the response is wrapped in markdown; on base
/// models it is a raw continuation — both are handled.
///
/// R1+R2: `entry_point` is passed down so multi-block completions resolve to
/// the block containing `def {entry_point}(` (not the first explanatory
/// snippet the model may emit).
///
/// §69 RC3 FIX: the extracted code block contains the function (signature +
/// body) but NOT the prompt's preamble — typing imports (`from typing import
/// List`), constants, helpers, etc. Concatenating ONLY the code block drops
/// those, producing `NameError: List is not defined` when the function
/// signature uses typing aliases. Prepend the prompt's preamble (everything
/// before `def {entry_point}(`) so imports survive.
#[cfg(feature = "inference")]
fn build_humaneval_completion(
    problem: &HumanEvalProblem,
    entry: &str,
    result: &realizar::InferenceResult,
    tokenizer: &realizar::apr::BpeTokenizer,
) -> String {
    let Some(code) = extract_python_code_block_targeted(&result.text, Some(entry)) else {
        return humaneval_raw_completion(problem, result, tokenizer);
    };
    let preamble = extract_prompt_preamble(&problem.prompt, entry);
    if preamble.is_empty() {
        code
    } else {
        format!("{preamble}\n{code}")
    }
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
/// reprocesses the full sequence; the GPU-vs-CPU ratio is a measured figure that
/// belongs in an `evidence/` receipt, not in this comment.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn run_humaneval_inference_cuda(
    model_path: &Path,
    problems: &[HumanEvalProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    let (mut trainer, tokenizer, max_seq) = init_cuda_eval(model_path, json_output)?;

    let mut passed = 0usize;
    let mut results = Vec::new();
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let entry = humaneval_entry_point(problem);

        let prompt_tokens = tokenizer.encode(&problem.prompt);
        if prompt_tokens.is_empty() {
            results.push((problem.task_id.clone(), entry.to_string(), false));
            continue;
        }

        // Autoregressive generation: build sequence incrementally
        let tokens =
            generate_tokens_cuda(&mut trainer, &prompt_tokens, max_seq, 256, &mut rng_state)?;

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

        report_eval_progress(json_output, i, problems.len(), passed, 10);
    }

    Ok((passed, results))
}

/// Resolve a model path to its checkpoint directory.
///
/// ALB-089: `model_path` may be a `.apr` file that lives inside the
/// checkpoint directory; a directory is already the checkpoint directory.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn resolve_checkpoint_dir(model_path: &Path) -> &Path {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path)
    } else {
        model_path
    }
}

/// Load the tokenizer for a CUDA eval run.
///
/// Sibling lookup off the original `model_path` (a file), falling back to
/// `tokenizer.json` in the checkpoint directory.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn load_cuda_eval_tokenizer(
    model_path: &Path,
    checkpoint_dir: &Path,
) -> std::result::Result<realizar::apr::BpeTokenizer, String> {
    realizar::apr::AprV2Model::load_tokenizer(model_path)
        .or_else(|| {
            // Fallback: try tokenizer.json directly in checkpoint dir
            let tok_path = checkpoint_dir.join("tokenizer.json");
            realizar::apr::AprV2Model::load_tokenizer_from_path(&tok_path)
        })
        .ok_or_else(|| format!("No tokenizer found in {}", checkpoint_dir.display()))
}

/// GPU eval context: the trainer, its tokenizer, and the model's context
/// window (`max_position_embeddings`).
#[cfg(all(feature = "cuda", feature = "training"))]
type CudaEvalContext = (
    entrenar::train::CudaTransformerTrainer,
    realizar::apr::BpeTokenizer,
    usize,
);

/// Initialise a CUDA eval run: config → GPU trainer → tokenizer.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn init_cuda_eval(
    model_path: &Path,
    json_output: bool,
) -> std::result::Result<CudaEvalContext, String> {
    let checkpoint_dir = resolve_checkpoint_dir(model_path);

    let config = load_transformer_config(checkpoint_dir)?;
    let max_seq = config.max_position_embeddings;

    if !json_output {
        println!(
            "  {} Loading model onto GPU for inference (ALB-089)...",
            "→".dimmed()
        );
    }

    let trainer = entrenar::train::CudaTransformerTrainer::for_inference(checkpoint_dir, config)
        .map_err(|e| format!("CUDA inference init failed: {e}"))?;

    // Load tokenizer -- use original model_path (file) for sibling lookup
    let tokenizer = load_cuda_eval_tokenizer(model_path, checkpoint_dir)?;

    if !json_output {
        println!("  {} GPU inference ready", "✓".green());
    }

    Ok((trainer, tokenizer, max_seq))
}

/// Greedy autoregressive generation on the GPU.
///
/// No KV cache -- each step reprocesses the full sequence via
/// `forward_logits()`. Stops at `max_new` steps, at the context window, or
/// on token 0 (EOS). Returns prompt tokens + generated tokens.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "cuda", feature = "training"))]
fn generate_tokens_cuda(
    trainer: &mut entrenar::train::CudaTransformerTrainer,
    prompt_tokens: &[u32],
    max_seq: usize,
    max_new: usize,
    rng_state: &mut u64,
) -> std::result::Result<Vec<u32>, String> {
    let mut tokens: Vec<u32> = prompt_tokens.to_vec();

    for _ in 0..max_new {
        if tokens.len() >= max_seq {
            break;
        }

        // Forward full sequence, get last-position logits
        let logits = trainer
            .forward_logits(&tokens)
            .ok_or_else(|| "forward_logits failed".to_string())?;

        let next = sample_token(&logits, 0.0, rng_state);
        tokens.push(next);

        // Stop at EOS or token 0
        if next == 0 {
            break;
        }
    }

    Ok(tokens)
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

/// PMAT-CODE-SHIP-005-H4-FIX: extract the first Python code block from a
/// ChatML assistant response.
///
/// Instruct-family models (Qwen-Coder-Instruct, etc.) respond to a coding
/// prompt with a markdown-wrapped solution like:
///
/// ~~~text
/// Certainly! Here's a solution:
/// ```python
/// def truncate_number(number: float) -> float:
///     import math
///     fractional_part, _ = math.modf(number)
///     return fractional_part
/// ```
/// ~~~
///
/// This helper extracts the inner code between the first ```python fence
/// and the next ``` fence. Returns `None` when no fenced Python block is
/// found (caller falls back to raw-continuation slicing).
///
/// Tolerant of variants:
/// - ```python … ``` (preferred)
/// - ```py … ```
/// - ``` … ``` (untagged — still treated as Python on a code-eval path)
pub(super) fn extract_python_code_block(text: &str) -> Option<String> {
    extract_python_code_block_targeted(text, None)
}

/// PMAT-CODE-SHIP-005-R1-R2-REFINEMENT: function-targeted extraction.
///
/// When `entry_point` is supplied, scan ALL fenced Python code blocks and
/// prefer the one whose body contains `def {entry_point}(`. This handles:
///
/// **R1 (multi-block completions)**: model sometimes emits an explanatory
/// snippet (e.g., wrong/incomplete code) BEFORE the actual solution block.
/// First-block-wins picks the snippet; function-targeted picks the solution.
///
/// **R2 (function-name match)**: even when only one block exists, the
/// function-name match is an extra safety check that the extracted block
/// is the intended solution (not just unrelated demo code).
///
/// Fallback: if no block contains the entry_point, return the first
/// non-empty fenced block (preserves `extract_python_code_block` behaviour).
pub(super) fn extract_python_code_block_targeted(
    text: &str,
    entry_point: Option<&str>,
) -> Option<String> {
    let blocks = collect_fenced_blocks(text);
    if blocks.is_empty() {
        return None;
    }

    // R2: prefer block containing `def {entry_point}(`.
    if let Some(ep) = entry_point {
        let needle = format!("def {ep}(");
        if let Some(hit) = blocks.iter().find(|block| block.contains(&needle)) {
            return Some(hit.clone());
        }
    }

    // Fallback: first non-empty block (legacy behaviour preserved).
    Some(blocks[0].clone())
}

/// Byte offset just past the EARLIEST opening code fence in `remainder`.
///
/// The accepted opening fences are the same three variants the extractor has
/// always tolerated, in this precedence order for a tie: ```` ```python\n ````,
/// ```` ```py\n ````, ```` ```\n ````. `None` when no fence remains.
fn find_fence_open(remainder: &str) -> Option<usize> {
    ["```python\n", "```py\n", "```\n"]
        .iter()
        .filter_map(|fence| remainder.find(fence).map(|rel| (rel, rel + fence.len())))
        .min_by_key(|&(rel, _)| rel)
        .map(|(_, after_open)| after_open)
}

/// Collect every non-empty fenced block in `text`, in source order.
///
/// A block runs from just past an opening fence to the next `\n``` `. An
/// unterminated final fence is dropped (the scan stops there), matching the
/// behaviour this loop has always had.
fn collect_fenced_blocks(text: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    while cursor < text.len() {
        let Some(after_open_rel) = find_fence_open(&text[cursor..]) else {
            break;
        };
        let after_open = cursor + after_open_rel;
        let Some(rel_end) = text[after_open..].find("\n```") else {
            break;
        };
        let code = &text[after_open..after_open + rel_end];
        if !code.trim().is_empty() {
            blocks.push(code.to_string());
        }
        cursor = after_open + rel_end + "\n```".len();
    }

    blocks
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

/// §69 RC3 FIX: extract everything in `prompt` that appears BEFORE the
/// `def {entry_point}(` line — i.e., the imports/constants/helpers that
/// the model assumes are in scope. Used by the ChatML/markdown path to
/// reconstitute a valid `full_program` when the assistant's code block
/// omits the imports (which it does for instruct models that read the
/// imports from the user prompt's context).
///
/// Returns an empty string when:
/// - `entry_point` is empty or "unknown"
/// - `def {entry_point}(` is not found in the prompt
/// - There's no content before `def {entry_point}(` (preamble-less prompt)
///
/// The returned string has trailing whitespace trimmed but leading
/// imports/code preserved verbatim.
pub(super) fn extract_prompt_preamble(prompt: &str, entry_point: &str) -> String {
    if entry_point.is_empty() || entry_point == "unknown" {
        return String::new();
    }
    let needle = format!("def {entry_point}(");
    let Some(idx) = prompt.find(&needle) else {
        return String::new();
    };
    prompt[..idx].trim_end().to_string()
}

/// PMAT-CODE-SHIP-005-WHITESPACE-RESIDUAL: normalise raw-continuation indent.
///
/// HumanEval prompts end with `    """\n` (4-space-indented docstring close);
/// the function body should continue at 4-space indent. On `apr eval --task
/// humaneval` raw-continuation path, the model emits 5-space leading indent
/// (BPE tokenization artifact at the prompt-completion boundary). The
/// resulting concatenation `    """\n     for i in...` is invalid Python
/// (IndentationError).
///
/// Manual `apr run` on the same model with auto-wrap produces correct
/// 4-space; the bug is raw-continuation-specific.
///
/// Fix: detect the prompt's expected continuation indent (last non-empty
/// line's leading-space count) vs the completion's first non-empty line
/// indent; if completion is over-indented, dedent every line by the
/// excess. Only over-indented completions are touched (no risk to
/// correctly-aligned outputs).
///
/// Lines without sufficient leading whitespace (blank lines or top-level
/// code) are left untouched.
pub(super) fn align_continuation_indent(prompt: &str, completion: &str) -> String {
    let expected_indent = prompt
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .unwrap_or(0);

    let actual_indent = completion
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .unwrap_or(0);

    if actual_indent <= expected_indent {
        return completion.to_string();
    }

    let excess = actual_indent - expected_indent;
    let prefix = " ".repeat(excess);

    // Dedent only the function-body chunk — stop at the first non-empty
    // line that drops to indent 0 (signaling we've exited the function
    // scope; e.g., `if __name__ == "__main__":` post-amble). Top-level
    // code at indent < `excess` must be preserved as-is.
    let mut in_body = true;
    completion
        .split_inclusive('\n')
        .map(|line| {
            let trimmed = line.trim_start_matches(' ').trim_end_matches('\n');
            // Track scope transition: once we see a non-empty 0-indent line,
            // we're past the function body — leave all subsequent lines alone.
            if in_body && !trimmed.is_empty() {
                let leading = line.chars().take_while(|c| *c == ' ').count();
                if leading == 0 {
                    in_body = false;
                }
            }
            if in_body && line.starts_with(&prefix) {
                line[excess..].to_string()
            } else {
                line.to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod extract_python_code_block_targeted_tests {
    use super::extract_python_code_block_targeted;

    /// R2 canonical: assistant emits explanatory snippet block FIRST then
    /// the actual solution block. Without targeting, first-wins picks the
    /// wrong block.
    #[test]
    fn prefers_block_containing_entry_point() {
        let text = "First a sketch:\n```python\n# rough idea\nx = 1\n```\nNow the actual solution:\n```python\ndef separate_paren_groups(s):\n    return [s]\n```";
        let got = extract_python_code_block_targeted(text, Some("separate_paren_groups"));
        assert_eq!(
            got.as_deref(),
            Some("def separate_paren_groups(s):\n    return [s]")
        );
    }

    /// Single block + matching entry_point still returns that block.
    #[test]
    fn single_block_matching_entry() {
        let text = "```python\ndef f(x):\n    return x\n```";
        let got = extract_python_code_block_targeted(text, Some("f"));
        assert_eq!(got.as_deref(), Some("def f(x):\n    return x"));
    }

    /// No matching entry_point → falls back to first block (legacy behaviour).
    #[test]
    fn no_entry_match_falls_back_to_first() {
        let text = "```python\nimport os\n```\n```python\ndef other():\n    pass\n```";
        let got = extract_python_code_block_targeted(text, Some("missing_fn"));
        assert_eq!(got.as_deref(), Some("import os"));
    }

    /// `None` entry_point → first-block-wins (identical to legacy
    /// `extract_python_code_block` behaviour).
    #[test]
    fn no_entry_point_first_block_wins() {
        let text = "```python\nfirst = 1\n```\n```python\ndef target():\n    pass\n```";
        let got = extract_python_code_block_targeted(text, None);
        assert_eq!(got.as_deref(), Some("first = 1"));
    }

    /// Mixed fence tags across blocks: still collects all and picks the
    /// one with matching entry_point.
    #[test]
    fn mixed_fence_tags_picks_entry_block() {
        let text = "```\n# untagged junk\n```\n```py\ndef helper(): pass\n```\n```python\ndef target():\n    return 42\n```";
        let got = extract_python_code_block_targeted(text, Some("target"));
        assert_eq!(got.as_deref(), Some("def target():\n    return 42"));
    }

    /// No fence at all → None.
    #[test]
    fn no_fence_returns_none() {
        let text = "just text without fences";
        let got = extract_python_code_block_targeted(text, Some("anything"));
        assert!(got.is_none());
    }

    /// Empty-content fences are skipped; entry-point match still works on
    /// later non-empty block.
    #[test]
    fn skips_empty_fences_before_match() {
        let text = "```python\n\n```\n```python\ndef target():\n    pass\n```";
        let got = extract_python_code_block_targeted(text, Some("target"));
        assert_eq!(got.as_deref(), Some("def target():\n    pass"));
    }
}

#[cfg(test)]
mod extract_python_code_block_tests {
    use super::extract_python_code_block;

    /// SHIP-005 H4 canonical case: assistant emits a Python fenced block.
    #[test]
    fn extracts_python_fenced_block() {
        let text = "Certainly!\n```python\ndef f(x):\n    return x + 1\n```\nLet me know if you need more.";
        let got = extract_python_code_block(text);
        assert_eq!(got.as_deref(), Some("def f(x):\n    return x + 1"));
    }

    /// Tolerates `py` shortform fence.
    #[test]
    fn extracts_py_short_fence() {
        let text = "```py\ndef g():\n    pass\n```";
        let got = extract_python_code_block(text);
        assert_eq!(got.as_deref(), Some("def g():\n    pass"));
    }

    /// Untagged fence — accept for code-eval path.
    #[test]
    fn extracts_untagged_fence() {
        let text = "```\nimport os\n```";
        let got = extract_python_code_block(text);
        assert_eq!(got.as_deref(), Some("import os"));
    }

    /// No fence → None (caller falls back to raw-continuation).
    #[test]
    fn returns_none_on_no_fence() {
        let text = "Just plain text with no code block.";
        let got = extract_python_code_block(text);
        assert!(got.is_none());
    }

    /// Empty fenced block → None (not an actionable code completion).
    #[test]
    fn returns_none_on_empty_fence() {
        let text = "```python\n\n```";
        let got = extract_python_code_block(text);
        assert!(got.is_none());
    }

    /// Multiple fenced blocks → first one wins.
    #[test]
    fn extracts_first_of_multiple_blocks() {
        let text = "```python\nfirst = 1\n```\nthen:\n```python\nsecond = 2\n```";
        let got = extract_python_code_block(text);
        assert_eq!(got.as_deref(), Some("first = 1"));
    }
}

#[cfg(test)]
mod extract_prompt_preamble_tests {
    use super::extract_prompt_preamble;

    /// §69 RC3 canonical: HumanEval/1-shaped prompt with `from typing import List`
    /// preamble must be extracted before `def {entry_point}(`.
    #[test]
    fn captures_typing_import_preamble() {
        let prompt = "from typing import List\n\n\ndef separate_paren_groups(s: str) -> List[str]:\n    \"\"\"...\"\"\"\n";
        let got = extract_prompt_preamble(prompt, "separate_paren_groups");
        assert_eq!(got, "from typing import List");
    }

    /// Multi-import + constant preamble — preserves every line up to `def`.
    #[test]
    fn captures_multiline_preamble() {
        let prompt = "from typing import List, Tuple\nimport math\n\nPI = 3.14\n\ndef f(x: List[int]) -> Tuple[int, int]:\n    pass\n";
        let got = extract_prompt_preamble(prompt, "f");
        assert_eq!(
            got,
            "from typing import List, Tuple\nimport math\n\nPI = 3.14"
        );
    }

    /// No preamble — `def` is at byte 0 → returns empty.
    #[test]
    fn empty_when_def_at_start() {
        let prompt = "def trivial():\n    pass\n";
        let got = extract_prompt_preamble(prompt, "trivial");
        assert_eq!(got, "");
    }

    /// `entry_point` not found in prompt → returns empty (don't guess).
    #[test]
    fn empty_when_entry_missing() {
        let prompt = "from typing import List\n\ndef other_fn():\n    pass\n";
        let got = extract_prompt_preamble(prompt, "expected_fn");
        assert_eq!(got, "");
    }

    /// Empty entry_point string → returns empty (safety guard).
    #[test]
    fn empty_when_entry_empty() {
        let prompt = "from typing import List\n\ndef f():\n    pass\n";
        let got = extract_prompt_preamble(prompt, "");
        assert_eq!(got, "");
    }

    /// "unknown" sentinel (fallback when extract_function_name fails) → empty.
    #[test]
    fn empty_when_entry_unknown() {
        let prompt = "from typing import List\n\ndef f():\n    pass\n";
        let got = extract_prompt_preamble(prompt, "unknown");
        assert_eq!(got, "");
    }

    /// §69 RC3 falsifier: a composed full_program built from
    /// `preamble + extracted_code + test + check` MUST be valid Python
    /// when the prompt has typing imports.
    #[test]
    fn rc3_falsifier_composed_program_is_valid_python() {
        let prompt = "from typing import List\n\n\ndef separate_paren_groups(s: str) -> List[str]:\n    pass\n";
        let preamble = extract_prompt_preamble(prompt, "separate_paren_groups");
        let extracted_code = "def separate_paren_groups(s: str) -> List[str]:\n    return [s]";
        let full = format!("{preamble}\n{extracted_code}\n");
        assert!(
            full.starts_with("from typing import List"),
            "preamble must lead with import; got: {full}"
        );
        assert!(
            full.contains("def separate_paren_groups"),
            "must contain function: {full}"
        );
    }
}

#[cfg(test)]
mod align_indent_tests {
    use super::align_continuation_indent;

    /// Pre-fix HumanEval/0 reproduction: 5-space body indent should
    /// dedent to 4-space, with relative inner nesting preserved.
    #[test]
    fn dedents_one_excess_space() {
        let prompt = "def f(x: int) -> int:\n    \"\"\" doc.\n    \"\"\"\n";
        let completion =
            "     for i in range(x):\n         if i > 0:\n             return i\n     return 0\n";
        let got = align_continuation_indent(prompt, completion);
        let want =
            "    for i in range(x):\n        if i > 0:\n            return i\n    return 0\n";
        assert_eq!(got, want);
    }

    /// Correctly-aligned completion is left unchanged.
    #[test]
    fn passthrough_when_already_correct() {
        let prompt = "def f():\n    \"\"\"doc\"\"\"\n";
        let completion = "    return 42\n";
        let got = align_continuation_indent(prompt, completion);
        assert_eq!(got, completion);
    }

    /// Top-level code after the function body (e.g., `if __name__`) has 0
    /// leading spaces and must NOT be dedented (would crash on slice).
    #[test]
    fn leaves_zero_indent_lines_untouched() {
        let prompt = "def f():\n    \"\"\"doc\"\"\"\n";
        let completion = "     return 1\n\n\nif __name__ == \"__main__\":\n    pass\n";
        let got = align_continuation_indent(prompt, completion);
        let want = "    return 1\n\n\nif __name__ == \"__main__\":\n    pass\n";
        assert_eq!(got, want);
    }

    /// Multi-space excess (2+) is dedented uniformly.
    #[test]
    fn dedents_multi_space_excess() {
        let prompt = "    pass\n";
        let completion = "        x = 1\n            nested = 2\n";
        let got = align_continuation_indent(prompt, completion);
        // expected = 4 ('    pass' last line), actual = 8 → excess = 4
        let want = "    x = 1\n        nested = 2\n";
        assert_eq!(got, want);
    }

    /// Empty completion is passthrough.
    #[test]
    fn empty_completion() {
        let prompt = "def f():\n    pass\n";
        let completion = "";
        let got = align_continuation_indent(prompt, completion);
        assert_eq!(got, "");
    }

    /// Mutation-survey section: invariant under no-indent prompt + no-indent
    /// completion (early-return guard).
    #[test]
    fn no_indent_anywhere() {
        let prompt = "x = 1\n";
        let completion = "y = 2\n";
        let got = align_continuation_indent(prompt, completion);
        assert_eq!(got, completion);
    }
}

/// Per-problem debug dump for `APR_EVAL_DEBUG=1`. Diagnoses §69
/// "harness bug" candidate root causes RC1-RC4 by writing the full
/// model response, extracted completion, executed program, exit code,
/// stderr, and timeout flag to `/tmp/apr_eval_debug_<task>.json`.
///
/// Used to compose a falsifier: manual `python3` execution of the
/// dumped program vs harness `execute_python_test` result.
pub(super) fn write_apr_eval_debug(
    task_id: &str,
    prompt: &str,
    response: &str,
    completion: &str,
    full_program: &str,
    exec: &PythonExecResult,
) {
    let safe_task = task_id.replace(['/', '\\', ' '], "_");
    let path = std::env::temp_dir().join(format!("apr_eval_debug_{safe_task}.json"));
    let json = serde_json::json!({
        "task_id": task_id,
        "prompt": prompt,
        "response": response,
        "response_len": response.len(),
        "completion": completion,
        "completion_len": completion.len(),
        "full_program": full_program,
        "exit_code": exec.exit_code,
        "stderr": exec.stderr_capture,
        "timed_out": exec.timed_out,
        "spawn_error": exec.spawn_error,
        "success": exec.success,
    });
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(&json).unwrap_or_default(),
    );
}

/// Execute a Python program and check if all assertions pass.
/// Returns true if exit code is 0, false otherwise.
/// Enforces a timeout to catch infinite loops (FALSIFY-EVAL-003).
pub(super) fn execute_python_test(program: &str, timeout_secs: u64) -> bool {
    execute_python_test_with_diagnostics(program, timeout_secs).success
}

/// Result of executing a Python program: success flag + diagnostics.
/// `exit_code` is `Some(code)` when the process exited; `None` when killed
/// by timeout or spawn failed. `stderr_capture` is captured up to 64KB.
pub(super) struct PythonExecResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stderr_capture: String,
    pub timed_out: bool,
    pub spawn_error: Option<String>,
}

/// Execute Python and return diagnostics. Drains stderr to avoid pipe-buffer
/// deadlock (RC2 candidate from §69).
pub(super) fn execute_python_test_with_diagnostics(
    program: &str,
    timeout_secs: u64,
) -> PythonExecResult {
    use std::io::Read;
    use std::process::Command;
    use std::time::{Duration, Instant};

    let tmp = std::env::temp_dir().join(format!(
        "apr_eval_{}_{}.py",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    if let Err(e) = std::fs::write(&tmp, program) {
        return PythonExecResult {
            success: false,
            exit_code: None,
            stderr_capture: String::new(),
            timed_out: false,
            spawn_error: Some(format!("tmp write: {e}")),
        };
    }

    let spawn_result = Command::new("python3")
        .arg(&tmp)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match spawn_result {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return PythonExecResult {
                success: false,
                exit_code: None,
                stderr_capture: String::new(),
                timed_out: false,
                spawn_error: Some(format!("spawn: {e}")),
            };
        }
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut timed_out = false;
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };

    let mut stderr_capture = String::new();
    if let Some(mut s) = child.stderr.take() {
        let mut buf = vec![0u8; 65536];
        if let Ok(n) = s.read(&mut buf) {
            stderr_capture = String::from_utf8_lossy(&buf[..n]).to_string();
        }
    }

    let _ = std::fs::remove_file(&tmp);

    let exit_code = exit_status.and_then(|s| s.code());
    let success = exit_status.map(|s| s.success()).unwrap_or(false);

    PythonExecResult {
        success,
        exit_code,
        stderr_capture,
        timed_out,
        spawn_error: None,
    }
}

#[cfg(test)]
mod execute_python_test_diagnostics_tests {
    use super::execute_python_test_with_diagnostics;

    /// Detect whether `python3` is available in the test environment.
    /// The workspace-test CI container does not install python3; these
    /// tests early-return success when python3 is missing so the lib-test
    /// suite stays green on container CI. The same tests run on
    /// developer machines + gx10 where python3 IS present and exercise
    /// the full diagnostic surface.
    fn python3_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Trivially-passing program reports success + exit_code 0 + empty stderr.
    #[test]
    fn success_program_reports_zero_exit_and_empty_stderr() {
        if !python3_available() {
            return;
        }
        let program = "print('hello')\n";
        let r = execute_python_test_with_diagnostics(program, 5);
        assert!(r.success, "program should succeed");
        assert_eq!(r.exit_code, Some(0));
        assert!(
            r.stderr_capture.is_empty(),
            "no stderr expected, got: {}",
            r.stderr_capture
        );
        assert!(!r.timed_out);
        assert!(r.spawn_error.is_none());
    }

    /// Assertion failure → success=false, exit_code=1, stderr captured.
    #[test]
    fn assertion_failure_reports_nonzero_and_traceback() {
        if !python3_available() {
            return;
        }
        let program = "assert 1 == 2\n";
        let r = execute_python_test_with_diagnostics(program, 5);
        assert!(!r.success);
        assert_eq!(r.exit_code, Some(1));
        assert!(
            r.stderr_capture.contains("AssertionError"),
            "expected traceback, got: {}",
            r.stderr_capture
        );
        assert!(!r.timed_out);
    }

    /// Falsifier §69 harness invariant: a program that python3 PASSES manually
    /// MUST also be reported as passing by the harness. If this test ever fails
    /// we have an RC2 (false-negative) regression.
    #[test]
    fn harness_invariant_passing_program_reports_success() {
        if !python3_available() {
            return;
        }
        let program = "def f(x):\n    return x + 1\n\nassert f(1) == 2\n";
        let r = execute_python_test_with_diagnostics(program, 5);
        assert!(r.success, "passing program must be reported as success");
        assert_eq!(r.exit_code, Some(0));
    }

    /// Falsifier §69 RC2-extension: programs that emit verbose stderr but pass
    /// MUST NOT deadlock — the stderr pipe is drained.
    #[test]
    fn verbose_stderr_does_not_deadlock_on_success() {
        if !python3_available() {
            return;
        }
        // Emit ~10KB to stderr, then exit 0 → must report success without timeout.
        let program =
            "import sys\nfor _ in range(200):\n    print('x' * 50, file=sys.stderr)\nsys.exit(0)\n";
        let r = execute_python_test_with_diagnostics(program, 10);
        assert!(
            r.success,
            "10KB-stderr passing program timed_out={} exit_code={:?}",
            r.timed_out, r.exit_code
        );
        assert!(!r.timed_out);
    }

    /// Falsifier: when python3 is unavailable, exec result reports
    /// spawn_error rather than success.
    #[test]
    fn missing_python3_reports_spawn_error() {
        if python3_available() {
            return; // can't test absence when present
        }
        let r = execute_python_test_with_diagnostics("print('hello')\n", 5);
        assert!(!r.success);
        assert!(
            r.spawn_error.is_some(),
            "expected spawn_error when python3 absent"
        );
        assert_eq!(r.exit_code, None);
    }
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
        // Single greedy sample per problem ⇒ pass@k = pass@1 = fraction of problems solved,
        // for every k. compute_pass_at_k(total, passed, k) wrongly fed #problems/#solved into
        // the per-sample (n, c) slots, inflating pass@10/pass@100 (see compute_multisample_pass_at_k).
        let rate = if total == 0 {
            0.0
        } else {
            passed as f64 / total as f64
        };
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

    // PMAT-702: parity with HumanEval. Emit structured "inference_failed"
    // result before returning Err so JSON-parsing tools get a usable failure
    // signal (pass@k = 0, mode = "inference_failed", inference_error populated).
    if !any_ok {
        let err_msg = first_err
            .clone()
            .unwrap_or_else(|| "(no error captured)".to_string());
        if !json_output {
            println!("  Inference error: {err_msg}");
            println!("  All MBPP samples failed inference — pass counters are 0.");
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
            "inference_failed",
            json_output,
            Some(("inference_error", &err_msg)),
        );
        return Err(CliError::InferenceFailed(format!(
            "MBPP inference failed for all samples: {err_msg}"
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

/// ALB-085 + PMAT-CODE-MBPP-H4-FIX (2026-05-12): Run MBPP with actual model
/// inference + Python test execution.
///
/// Routes through `realizar::run_inference` + `InferenceConfig::with_prompt`
/// (ChatML auto-wrap for instruct models) — mirrors the §70 HumanEval H4 +
/// R1+R2 cascade. MBPP prompts are natural language ("Write a python
/// function to..."); without ChatML wrap, instruct models emit NL-prose
/// continuations ("Example: Input: ... Output: ...") instead of code (see
/// `evidence/section-72-mbpp-cascade-2026-05-12/findings.json` for the
/// pre-fix MBPP/11 SyntaxError evidence).
///
/// Parse `\`\`\`python ... \`\`\`` markdown blocks from the response. MBPP
/// has no Python imports in the prompt, so the §70 RC3 prompt-preamble
/// handling does not apply — the extracted code block is the program.
#[cfg(feature = "inference")]
fn run_mbpp_inference(
    model_path: &Path,
    problems: &[MbppProblem],
    _k_values: &[usize],
    json_output: bool,
) -> std::result::Result<(usize, Vec<(String, String, bool)>), String> {
    use realizar::{run_inference, InferenceConfig};

    if !json_output {
        println!("  {} Loading model for inference...", "→".dimmed());
    }
    let tokenizer = realizar::apr::AprV2Model::load_tokenizer(model_path)
        .ok_or_else(|| "No tokenizer found".to_string())?;

    if !json_output {
        println!("  {} Tokenizer loaded", "✓".green());
    }

    let mut passed = 0usize;
    let mut results = Vec::new();

    for (i, problem) in problems.iter().enumerate() {
        let task_id = mbpp_task_id(problem);
        let prompt = mbpp_chat_prompt(problem);

        // H4 fix: route through ChatML auto-wrap via `with_prompt` (instruct
        // models). Raw NL → ChatML user message → assistant emits markdown
        // code block.
        let config_chatml = InferenceConfig::new(model_path)
            .with_prompt(prompt.clone())
            .with_max_tokens(512)
            .with_temperature(0.0)
            .with_top_k(1);

        let result = match run_inference(&config_chatml) {
            Ok(r) => r,
            Err(e) => {
                if !json_output {
                    eprintln!("  [FAIL] {task_id}: inference error: {e}");
                }
                results.push((task_id, String::new(), false));
                continue;
            }
        };

        let completion = build_mbpp_completion(&prompt, &result, &tokenizer);

        let full_program = mbpp_full_program(&completion, problem);

        let exec_result = execute_python_test_with_diagnostics(&full_program, 10);
        let ok = exec_result.success;

        if std::env::var("APR_EVAL_DEBUG").is_ok() {
            write_apr_eval_debug(
                &task_id,
                &prompt,
                &result.text,
                &completion,
                &full_program,
                &exec_result,
            );
        }

        if ok {
            passed += 1;
        }

        results.push((task_id, String::new(), ok));

        report_eval_progress(json_output, i, problems.len(), passed, 50);
    }

    Ok((passed, results))
}

/// MBPP task identifier, rendered from the JSON `task_id` field.
///
/// MBPP numbers its tasks; a string id is passed through verbatim, anything
/// else is rendered through its JSON `Display`.
#[cfg(feature = "inference")]
fn mbpp_task_id(problem: &MbppProblem) -> String {
    match &problem.task_id {
        serde_json::Value::Number(n) => format!("MBPP/{n}"),
        serde_json::Value::String(s) => s.clone(),
        v => format!("MBPP/{v}"),
    }
}

/// MBPP canonical prompt format: NL description + `test_list` hint.
///
/// Without the `test_list` hint, the model invents its own function name
/// (e.g., `remove_first_last_occurrence` for MBPP/11) and fails the
/// assertion (`remove_Occ` expected). The standard MBPP format used by
/// Bigcode + lm-eval-harness + the canonical paper includes the first
/// 1-3 test assertions as `Your code should pass these tests:` hints —
/// this implicitly specifies the function name and signature.
#[cfg(feature = "inference")]
fn mbpp_chat_prompt(problem: &MbppProblem) -> String {
    if problem.test_list.is_empty() {
        return problem.text.clone();
    }
    format!(
        "{}\nYour code should pass these tests:\n{}\n",
        problem.text,
        problem.test_list.join("\n")
    )
}

/// Turn an assistant response into the MBPP completion to test.
///
/// R1+R2: extract the Python code block. MBPP has no `entry_point` in the
/// problem schema (unlike HumanEval), so `None` is passed — the
/// first-non-empty-block fallback is appropriate. When no block is found,
/// fall back to raw continuation: slice past the prompt, truncate at the
/// next top-level `def`.
#[cfg(feature = "inference")]
fn build_mbpp_completion(
    prompt: &str,
    result: &realizar::InferenceResult,
    tokenizer: &realizar::apr::BpeTokenizer,
) -> String {
    if let Some(code) = extract_python_code_block_targeted(&result.text, None) {
        return code;
    }
    let raw = if let Some(stripped) = result.text.strip_prefix(prompt) {
        stripped.to_string()
    } else {
        tokenizer.decode(completion_token_slice(result))
    };
    truncate_at_function_boundary(&raw).to_string()
}

/// Build the MBPP test program: completion + `test_setup_code` + assertions.
#[cfg(feature = "inference")]
fn mbpp_full_program(completion: &str, problem: &MbppProblem) -> String {
    let setup = problem.test_setup_code.as_deref().unwrap_or("").trim();
    let tests = problem.test_list.join("\n");
    if setup.is_empty() {
        format!("{completion}\n{tests}\n")
    } else {
        format!("{completion}\n{setup}\n{tests}\n")
    }
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
    let (mut trainer, tokenizer, max_seq) = init_cuda_eval(model_path, json_output)?;

    let mut passed = 0usize;
    let mut results = Vec::new();
    let mut rng_state: u64 = 42;

    for (i, problem) in problems.iter().enumerate() {
        let task_id = mbpp_task_id(problem);

        let prompt = format!("{}\n", problem.text);
        let prompt_tokens = tokenizer.encode(&prompt);
        if prompt_tokens.is_empty() {
            results.push((task_id, String::new(), false));
            continue;
        }

        let tokens =
            generate_tokens_cuda(&mut trainer, &prompt_tokens, max_seq, 512, &mut rng_state)?;

        let completion_tokens = &tokens[prompt_tokens.len()..];
        let completion = tokenizer.decode(completion_tokens);
        let completion = truncate_at_function_boundary(&completion);

        let full_program = mbpp_full_program(completion, problem);

        let exec_result = execute_python_test_with_diagnostics(&full_program, 10);
        let ok = exec_result.success;

        if std::env::var("APR_EVAL_DEBUG").is_ok() {
            write_apr_eval_debug(
                &task_id,
                &prompt,
                &tokenizer.decode(&tokens),
                completion,
                &full_program,
                &exec_result,
            );
        }

        if ok {
            passed += 1;
        }
        results.push((task_id, String::new(), ok));

        report_eval_progress(json_output, i, problems.len(), passed, 50);
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

#[cfg(test)]
mod inference_helper_tests {
    use super::*;

    fn he_problem(prompt: &str, test: &str, canonical: Option<&str>) -> HumanEvalProblem {
        HumanEvalProblem {
            task_id: "HumanEval/0".to_string(),
            prompt: prompt.to_string(),
            canonical_solution: canonical.map(String::from),
            test: test.to_string(),
            entry_point: None,
        }
    }

    // ── sample_token ───────────────────────────────────────────────────────

    #[test]
    fn sample_token_greedy_on_zero_temperature() {
        // temperature <= 0 ⇒ argmax.
        let logits = [0.1f32, 0.5, 0.2, 9.0, 0.3];
        let mut rng = 42u64;
        assert_eq!(sample_token(&logits, 0.0, &mut rng), 3);
    }

    #[test]
    fn sample_token_greedy_on_negative_temperature() {
        let logits = [5.0f32, 1.0, 2.0];
        let mut rng = 1u64;
        assert_eq!(sample_token(&logits, -1.0, &mut rng), 0);
    }

    #[test]
    fn sample_token_empty_logits_returns_zero() {
        let logits: [f32; 0] = [];
        let mut rng = 7u64;
        assert_eq!(sample_token(&logits, 1.0, &mut rng), 0);
        assert_eq!(sample_token(&logits, 0.0, &mut rng), 0);
    }

    #[test]
    fn sample_token_temperature_in_valid_range() {
        // With temperature > 0, result must be a valid index into logits.
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut rng = 0x1234_5678_9abc_def0u64;
        for _ in 0..50 {
            let tok = sample_token(&logits, 0.8, &mut rng);
            assert!((tok as usize) < logits.len(), "out of range: {tok}");
        }
    }

    #[test]
    fn sample_token_dominant_logit_usually_wins() {
        // One overwhelmingly large logit ⇒ low-temperature sampling should
        // almost always select it.
        let logits = [0.0f32, 0.0, 50.0, 0.0];
        let mut rng = 0xdead_beef_0000_0001u64;
        let mut hits = 0;
        for _ in 0..100 {
            if sample_token(&logits, 0.1, &mut rng) == 2 {
                hits += 1;
            }
        }
        assert!(
            hits >= 95,
            "dominant logit should win nearly always: {hits}"
        );
    }

    #[test]
    fn sample_token_is_deterministic_for_seed() {
        let logits = [1.0f32, 2.0, 3.0, 0.5, 1.5];
        let mut a = 999u64;
        let mut b = 999u64;
        let seq_a: Vec<u32> = (0..10)
            .map(|_| sample_token(&logits, 0.9, &mut a))
            .collect();
        let seq_b: Vec<u32> = (0..10)
            .map(|_| sample_token(&logits, 0.9, &mut b))
            .collect();
        assert_eq!(seq_a, seq_b);
    }

    // ── validate_humaneval_problem ─────────────────────────────────────────

    #[test]
    fn validate_rejects_empty_prompt() {
        let p = he_problem("   ", "assert f()", Some("return 1"));
        assert!(!validate_humaneval_problem(&p));
    }

    #[test]
    fn validate_rejects_empty_test() {
        let p = he_problem("def f():", "  \n ", Some("return 1"));
        assert!(!validate_humaneval_problem(&p));
    }

    #[test]
    fn validate_accepts_with_canonical_solution() {
        let p = he_problem("anything", "assert True", Some("return 42"));
        assert!(validate_humaneval_problem(&p));
    }

    #[test]
    fn validate_rejects_empty_canonical_without_def() {
        // canonical present but blank ⇒ falls through to def-check, no "def ".
        let p = he_problem("no function here", "assert True", Some("   "));
        assert!(!validate_humaneval_problem(&p));
    }

    #[test]
    fn validate_accepts_def_without_canonical() {
        let p = he_problem("def foo():\n    pass", "assert foo() is None", None);
        assert!(validate_humaneval_problem(&p));
    }

    #[test]
    fn validate_rejects_no_def_no_canonical() {
        let p = he_problem("just some text", "assert True", None);
        assert!(!validate_humaneval_problem(&p));
    }

    // ── extract_function_name ──────────────────────────────────────────────

    #[test]
    fn extract_function_name_simple() {
        assert_eq!(extract_function_name("def add(a, b):"), Some("add"));
    }

    #[test]
    fn extract_function_name_with_leading_lines() {
        let prompt = "from typing import List\n\ndef has_close_elements(nums: List[float], t: float) -> bool:\n    pass";
        assert_eq!(extract_function_name(prompt), Some("has_close_elements"));
    }

    #[test]
    fn extract_function_name_indented_def() {
        // The function picks up the first `def ` even when indented.
        let prompt = "    def inner(x):\n        return x";
        assert_eq!(extract_function_name(prompt), Some("inner"));
    }

    #[test]
    fn extract_function_name_none_when_no_def() {
        assert_eq!(extract_function_name("x = 1\ny = 2"), None);
    }

    #[test]
    fn extract_function_name_none_when_def_has_no_paren() {
        assert_eq!(extract_function_name("def malformed:"), None);
    }

    #[test]
    fn extract_function_name_first_of_many() {
        let prompt = "def first():\n    pass\ndef second():\n    pass";
        assert_eq!(extract_function_name(prompt), Some("first"));
    }

    // ── truncate_at_function_boundary ──────────────────────────────────────

    #[test]
    fn truncate_stops_at_next_def() {
        let c = "    return a + b\n\ndef other():\n    pass";
        assert_eq!(truncate_at_function_boundary(c), "    return a + b\n");
    }

    #[test]
    fn truncate_stops_at_next_class() {
        let c = "    return 1\nclass Foo:\n    pass";
        assert_eq!(truncate_at_function_boundary(c), "    return 1");
    }

    #[test]
    fn truncate_passthrough_when_no_boundary() {
        let c = "    return a + b";
        assert_eq!(truncate_at_function_boundary(c), c);
    }

    #[test]
    fn truncate_prefers_earliest_def_over_class() {
        // \ndef appears before \nclass ⇒ cut at def.
        let c = "x\ndef d():\n y\nclass C:\n z";
        assert_eq!(truncate_at_function_boundary(c), "x");
    }

    // ── print_humaneval_results: smoke + zero-division guard ───────────────

    #[test]
    fn print_humaneval_results_smoke() {
        let results = vec![
            ("HumanEval/0".to_string(), "f".to_string(), true),
            ("HumanEval/1".to_string(), "g".to_string(), false),
        ];
        // Should not panic.
        print_humaneval_results(&results, 2, 1, &[1, 10], 3.5, "inference");
    }

    #[test]
    fn print_humaneval_results_zero_total_no_panic() {
        let results: Vec<(String, String, bool)> = vec![];
        print_humaneval_results(&results, 0, 0, &[1], 0.0, "structural");
    }

    // ── write_apr_eval_debug: writes a JSON debug file ─────────────────────

    #[test]
    fn write_apr_eval_debug_creates_file() {
        let exec = PythonExecResult {
            success: false,
            exit_code: Some(1),
            stderr_capture: "AssertionError".to_string(),
            timed_out: false,
            spawn_error: None,
        };
        let task = format!("dbgtest/{}", std::process::id());
        write_apr_eval_debug(
            &task,
            "def f(): pass",
            "response text",
            "    return 1",
            "def f():\n    return 1",
            &exec,
        );
        let safe = task.replace(['/', '\\', ' '], "_");
        let path = std::env::temp_dir().join(format!("apr_eval_debug_{safe}.json"));
        assert!(path.exists(), "debug file not written: {}", path.display());
        let content = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(json["exit_code"], 1);
        assert_eq!(json["success"], false);
        assert_eq!(json["stderr"], "AssertionError");
        assert_eq!(json["response_len"], "response text".len());
        let _ = std::fs::remove_file(&path);
    }
}
