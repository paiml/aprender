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

        // Try to extract a Python code block from the assistant response.
        // On instruct-family models the response is wrapped in markdown;
        // on base models the response is raw continuation — both are handled.
        //
        // R1+R2: pass `entry_point` so multi-block completions resolve to
        // the block containing `def {entry_point}(` (not the first
        // explanatory snippet the model may emit).
        let completion =
            if let Some(code) = extract_python_code_block_targeted(&result.text, Some(entry)) {
                // ChatML/markdown path: assistant emitted `\`\`\`python\n…\n\`\`\``.
                //
                // §69 RC3 FIX: the extracted code block contains the function
                // (signature + body) but NOT the prompt's preamble — typing
                // imports (`from typing import List`), constants, helpers, etc.
                // Concatenating ONLY the code block drops those, producing
                // `NameError: List is not defined` when the function signature
                // uses typing aliases. Prepend the prompt's preamble (everything
                // before `def {entry_point}(`) so imports survive.
                let preamble = extract_prompt_preamble(&problem.prompt, entry);
                if preamble.is_empty() {
                    code
                } else {
                    format!("{preamble}\n{code}")
                }
            } else {
                // Raw-continuation fallback (pre-H4 path). Slice off the prompt
                // prefix when it's verbatim in result.text; otherwise decode
                // tokens past `input_token_count`. Apply dedent residual fix.
                let raw = if let Some(stripped) = result.text.strip_prefix(&problem.prompt) {
                    stripped.to_string()
                } else {
                    let completion_tokens = if result.tokens.len() > result.input_token_count {
                        &result.tokens[result.input_token_count..]
                    } else {
                        &result.tokens[..]
                    };
                    tokenizer.decode(completion_tokens)
                };
                let truncated = truncate_at_function_boundary(&raw);
                // The aligned form goes APPENDED to the prompt; encode that as
                // the full continuation. We then split the prompt back off in
                // the program-build step below.
                format!(
                    "{}{}",
                    problem.prompt,
                    align_continuation_indent(&problem.prompt, truncated)
                )
            };

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

/// PMAT-CODE-SHIP-005-H4-FIX: extract the first Python code block from a
/// ChatML assistant response.
///
/// Instruct-family models (Qwen-Coder-Instruct, etc.) respond to a coding
/// prompt with a markdown-wrapped solution like:
///
/// ```text
/// Certainly! Here's a solution:
/// ```python
/// def truncate_number(number: float) -> float:
///     import math
///     fractional_part, _ = math.modf(number)
///     return fractional_part
/// ```
/// ```
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
    // Collect all fenced blocks (any of the accepted opening fences).
    let mut blocks: Vec<String> = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let remainder = &text[cursor..];
        // Find the next opening fence (any variant); pick the earliest match.
        let mut best: Option<(usize, usize)> = None;
        for fence in ["```python\n", "```py\n", "```\n"] {
            if let Some(rel) = remainder.find(fence) {
                let after_open = rel + fence.len();
                match best {
                    None => best = Some((rel, after_open)),
                    Some((br, _)) if rel < br => best = Some((rel, after_open)),
                    _ => {}
                }
            }
        }
        let (_start_rel, after_open_rel) = match best {
            Some(p) => p,
            None => break,
        };
        let after_open = cursor + after_open_rel;
        if let Some(rel_end) = text[after_open..].find("\n```") {
            let code = &text[after_open..after_open + rel_end];
            if !code.trim().is_empty() {
                blocks.push(code.to_string());
            }
            cursor = after_open + rel_end + "\n```".len();
        } else {
            break;
        }
    }

    if blocks.is_empty() {
        return None;
    }

    // R2: prefer block containing `def {entry_point}(`.
    if let Some(ep) = entry_point {
        let needle = format!("def {ep}(");
        for block in &blocks {
            if block.contains(&needle) {
                return Some(block.clone());
            }
        }
    }

    // Fallback: first non-empty block (legacy behaviour preserved).
    Some(blocks[0].clone())
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
        let task_id = match &problem.task_id {
            serde_json::Value::Number(n) => format!("MBPP/{n}"),
            serde_json::Value::String(s) => s.clone(),
            v => format!("MBPP/{v}"),
        };

        // MBPP canonical prompt format: NL description + test_list hint.
        //
        // Without the test_list hint, the model invents its own function name
        // (e.g., `remove_first_last_occurrence` for MBPP/11) and fails the
        // assertion (`remove_Occ` expected). The standard MBPP format used by
        // Bigcode + lm-eval-harness + the canonical paper includes the first
        // 1-3 test assertions as `Your code should pass these tests:` hints —
        // this implicitly specifies the function name and signature.
        let test_hints = if problem.test_list.is_empty() {
            String::new()
        } else {
            format!(
                "\nYour code should pass these tests:\n{}\n",
                problem.test_list.join("\n")
            )
        };
        let prompt = format!("{}{}", problem.text, test_hints);

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

        // R1+R2: extract Python code block. MBPP has no entry_point in the
        // problem schema (unlike HumanEval), so we pass None — the
        // first-non-empty-block fallback is appropriate.
        let completion_owned =
            if let Some(code) = extract_python_code_block_targeted(&result.text, None) {
                // ChatML/markdown path: assistant emitted `\`\`\`python\n…\n\`\`\``.
                code
            } else {
                // Raw-continuation fallback (no code block found). Slice past the
                // prompt; truncate at next top-level def.
                let raw = if let Some(stripped) = result.text.strip_prefix(&prompt) {
                    stripped.to_string()
                } else {
                    let completion_tokens = if result.tokens.len() > result.input_token_count {
                        &result.tokens[result.input_token_count..]
                    } else {
                        &result.tokens[..]
                    };
                    tokenizer.decode(completion_tokens)
                };
                truncate_at_function_boundary(&raw).to_string()
            };
        let completion: &str = &completion_owned;

        // Build test program: completion + setup_code + test assertions
        let setup = problem.test_setup_code.as_deref().unwrap_or("").trim();
        let tests = problem.test_list.join("\n");
        let full_program = if setup.is_empty() {
            format!("{completion}\n{tests}\n")
        } else {
            format!("{completion}\n{setup}\n{tests}\n")
        };

        let exec_result = execute_python_test_with_diagnostics(&full_program, 10);
        let ok = exec_result.success;

        if std::env::var("APR_EVAL_DEBUG").is_ok() {
            write_apr_eval_debug(
                &task_id,
                &prompt,
                &result.text,
                completion,
                &full_program,
                &exec_result,
            );
        }

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
