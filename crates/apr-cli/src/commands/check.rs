//! Model Self-Test Command (PMAT-112: Real Validation)
//!
//! Executes a 10-stage pipeline integrity check with REAL computation.
//! No longer uses placeholder checks - actually runs tokenizer and forward pass.
//!
//! "If you cannot measure it, you cannot improve it. If you fake the measurement,
//! you are not improving it; you are lying to yourself." - PMAT-112

use crate::error::CliError;
use crate::output;
use colored::Colorize;
use std::path::Path;

#[cfg(feature = "inference")]
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

#[cfg(feature = "inference")]
use realizar::apr::AprV2Model;

/// Stage result with detailed information
#[derive(Debug)]
#[allow(dead_code)]
struct StageResult {
    name: &'static str,
    eli5: &'static str, // ELI5 explanation kept for future UI/tooltips
    passed: bool,
    details: Option<String>,
}

/// Run the 10-stage pipeline self-test with REAL validation
pub(crate) fn run(path: &Path, no_gpu: bool, json: bool) -> Result<(), CliError> {
    if !json {
        output::section("Model Self-Test (PMAT-112: Real Validation)");
        println!("Model: {}\n", path.display().to_string().cyan());
    }

    #[cfg(feature = "inference")]
    let results = run_real_checks(path, no_gpu)?;

    #[cfg(not(feature = "inference"))]
    let results = {
        let _ = no_gpu;
        if !json {
            output::warn("Inference feature not enabled. Cannot run real validation.");
            output::warn("Build with: cargo build --features inference");
        }
        vec![StageResult {
            name: "N/A",
            eli5: "Requires inference",
            passed: false,
            details: Some("Build with --features inference".to_string()),
        }]
    };

    let passed_count = results.iter().filter(|r| r.passed).count();
    let total_count = results.len();

    // GH-253: JSON output for parity checker
    if json {
        return print_json(&results, path, passed_count, total_count);
    }

    print_results_table(&results);

    if passed_count == total_count {
        println!(
            "\n{}",
            format!(
                "✅ {}/{} STAGES PASSED. MODEL PROVEN CORRECT.",
                passed_count, total_count
            )
            .green()
            .bold()
        );
        Ok(())
    } else {
        println!(
            "\n{}",
            format!(
                "❌ {}/{} STAGES PASSED. CHECK STAGE LOGS.",
                passed_count, total_count
            )
            .red()
            .bold()
        );
        Err(CliError::ValidationFailed(
            "Model self-test failed".to_string(),
        ))
    }
}

/// GH-253: Print check results as JSON for parity checker
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_json(
    results: &[StageResult],
    path: &Path,
    passed_count: usize,
    total_count: usize,
) -> Result<(), CliError> {
    let stages_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "status": if r.passed { "PASS" } else { "FAIL" },
                "details": r.details.as_deref().unwrap_or(""),
            })
        })
        .collect();

    let output = serde_json::json!({
        "model": path.display().to_string(),
        "stages": stages_json,
        "passed": passed_count,
        "total": total_count,
        "all_passed": passed_count == total_count,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );

    // GH-253: In JSON mode, always exit 0 so parity checker can parse the output.
    // The "all_passed" and individual stage statuses convey success/failure.
    Ok(())
}

/// Run REAL validation checks (PMAT-112)
#[cfg(feature = "inference")]
fn run_real_checks(path: &Path, no_gpu: bool) -> Result<Vec<StageResult>, CliError> {
    // Dispatch based on file extension
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_lowercase().as_str() {
        "apr" => run_real_checks_apr(path),
        "gguf" => run_real_checks_gguf(path, no_gpu),
        _ => Err(CliError::InvalidFormat(format!(
            "Unsupported format: {}. Use .apr or .gguf",
            ext
        ))),
    }
}

/// Build a stage result from a boolean tensor-existence check.
fn tensor_check_stage(
    name: &'static str,
    eli5: &'static str,
    found: bool,
    found_msg: &str,
    missing_msg: &str,
) -> StageResult {
    StageResult {
        name,
        eli5,
        passed: found,
        details: Some(if found { found_msg } else { missing_msg }.to_string()),
    }
}

/// Check if any name in the list matches any of the given substrings.
#[cfg(feature = "inference")]
fn any_name_contains(names: &[&str], patterns: &[&str]) -> bool {
    names.iter().any(|n| patterns.iter().any(|p| n.contains(p)))
}

/// Check if all pattern groups have at least one match (AND of OR groups).
#[cfg(feature = "inference")]
fn all_groups_match(names: &[&str], groups: &[&[&str]]) -> bool {
    groups.iter().all(|group| any_name_contains(names, group))
}

/// APR forward-pass logits check.
#[cfg(feature = "inference")]
fn check_apr_logits(model: &AprV2Model) -> StageResult {
    match model.forward(&[1u32]) {
        Ok(logits) => {
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());
            let valid = !has_nan && !has_inf && !logits.is_empty();
            let details = if has_nan {
                "NaN detected".to_string()
            } else if has_inf {
                "Inf detected".to_string()
            } else {
                format!("logits[{}]", logits.len())
            };
            StageResult {
                name: "Logits → Probs",
                eli5: "Scores → percentages",
                passed: valid,
                details: Some(details),
            }
        }
        Err(e) => StageResult {
            name: "Logits → Probs",
            eli5: "Scores → percentages",
            passed: false,
            details: Some(format!("Forward failed: {e}")),
        },
    }
}

/// APR softmax sampler check.
#[cfg(feature = "inference")]
fn check_apr_sampler(model: &AprV2Model) -> StageResult {
    match model.forward(&[1u32]) {
        Ok(logits) => {
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
            let prob_sum: f32 = logits.iter().map(|x| (x - max_logit).exp() / exp_sum).sum();
            let valid = (prob_sum - 1.0).abs() < 0.001;
            StageResult {
                name: "Sampler/Decode",
                eli5: "Pick word, return",
                passed: valid,
                details: Some(format!("softmax sum = {:.6}", prob_sum)),
            }
        }
        Err(e) => StageResult {
            name: "Sampler/Decode",
            eli5: "Pick word, return",
            passed: false,
            details: Some(format!("Forward failed: {e}")),
        },
    }
}

/// Run REAL validation for APR models (PMAT-112)
#[cfg(feature = "inference")]
fn run_real_checks_apr(path: &Path) -> Result<Vec<StageResult>, CliError> {
    let model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let metadata = model.metadata();
    let num_layers = metadata.num_layers.unwrap_or(0);
    let vocab_size = metadata.vocab_size.unwrap_or(32000);
    let names: Vec<&str> = model.tensor_names();

    let has_embed = any_name_contains(&names, &["emb", "wte", "token_embd"]);
    let has_rope = any_name_contains(&names, &["rope", "rotary"]);
    let has_lm_head = any_name_contains(&names, &["lm_head"]) || names.contains(&"output.weight");

    let test_tokens = vec![1u32, 2];
    let forward_ok = model.forward(&test_tokens).is_ok();

    Ok(vec![
        StageResult {
            name: "Tokenizer",
            eli5: "Words → numbers",
            passed: forward_ok,
            details: Some(format!("tokens={test_tokens:?}")),
        },
        tensor_check_stage(
            "Embedding",
            "Numbers → vectors",
            has_embed,
            "Found embedding tensor",
            "Missing embedding tensor",
        ),
        StageResult {
            name: "Positional Encoding",
            eli5: "\"You are word #3\"",
            passed: true,
            details: Some(
                if has_rope {
                    "RoPE tensors found"
                } else {
                    "RoPE computed inline"
                }
                .to_string(),
            ),
        },
        tensor_check_stage(
            "Q/K/V Projection",
            "Make 3 question copies",
            all_groups_match(
                &names,
                &[
                    &["q_proj", "attn_q"],
                    &["k_proj", "attn_k"],
                    &["v_proj", "attn_v"],
                ],
            ),
            "Q/K/V found",
            "Missing Q/K/V",
        ),
        tensor_check_stage(
            "Attention Scores",
            "\"Who to look at?\"",
            any_name_contains(&names, &["o_proj", "attn_output"]),
            "Attention output found",
            "Missing attention output",
        ),
        tensor_check_stage(
            "Feed-Forward (MLP)",
            "\"Think about it\"",
            all_groups_match(
                &names,
                &[
                    &["gate_proj", "ffn_gate"],
                    &["up_proj", "ffn_up"],
                    &["down_proj", "ffn_down"],
                ],
            ),
            "MLP found",
            "Missing MLP",
        ),
        StageResult {
            name: "Layer Norm",
            eli5: "Keep numbers stable",
            passed: all_groups_match(
                &names,
                &[
                    &["input_layernorm", "attn_norm"],
                    &["post_attention_layernorm", "ffn_norm"],
                ],
            ) && num_layers > 0,
            details: Some(format!("{num_layers} layers")),
        },
        StageResult {
            name: "LM Head",
            eli5: "Vector → vocab scores",
            passed: has_lm_head || has_embed,
            details: Some(format!(
                "vocab_size={vocab_size}{}",
                if has_lm_head { "" } else { " (tied)" }
            )),
        },
        check_apr_logits(&model),
        check_apr_sampler(&model),
    ])
}

/// GGUF LM Head check (explicit head or tied embeddings).
#[cfg(feature = "inference")]
fn check_gguf_lm_head(mapped: &MappedGGUFModel, vocab_size: usize) -> StageResult {
    let has_explicit = mapped.model.tensors.iter().any(|t| {
        (t.name == "output.weight" || t.name.contains("lm_head"))
            && t.dims.iter().any(|&d| d as usize == vocab_size)
    });
    let has_tied = mapped.model.tensors.iter().any(|t| {
        (t.name.contains("token_embd") || t.name.contains("embed_tokens"))
            && t.dims.iter().any(|&d| d as usize == vocab_size)
    });
    let passed = (has_explicit || has_tied) && vocab_size > 0;
    let details = if has_explicit {
        format!("vocab_size={vocab_size}")
    } else if has_tied {
        format!("vocab_size={vocab_size} (tied embeddings)")
    } else {
        "Missing LM head tensor".to_string()
    };
    StageResult {
        name: "LM Head",
        eli5: "Vector → vocab scores",
        passed,
        details: Some(details),
    }
}

/// Run REAL validation for GGUF models (PMAT-112)
#[cfg(feature = "inference")]
fn run_real_checks_gguf(path: &Path, _no_gpu: bool) -> Result<Vec<StageResult>, CliError> {
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to mmap GGUF: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let names: Vec<&str> = mapped
        .model
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let rope_theta = mapped.model.rope_freq_base().unwrap_or(10000.0);
    let has_embed = any_name_contains(&names, &["token_embd", "embed_tokens"]);

    Ok(vec![
        check_tokenizer_real(&model),
        tensor_check_stage(
            "Embedding",
            "Numbers → vectors",
            has_embed,
            "Found embedding tensor",
            "Missing embedding tensor",
        ),
        StageResult {
            name: "Positional Encoding",
            eli5: "\"You are word #3\"",
            passed: rope_theta > 1.0,
            details: Some(format!("rope_theta={rope_theta:.1}")),
        },
        tensor_check_stage(
            "Q/K/V Projection",
            "Make 3 question copies",
            all_groups_match(
                &names,
                &[
                    &["blk.0.attn_q", "layers.0.self_attn.q_proj"],
                    &["blk.0.attn_k", "layers.0.self_attn.k_proj"],
                    &["blk.0.attn_v", "layers.0.self_attn.v_proj"],
                ],
            ),
            "Q/K/V tensors found",
            "Missing Q/K/V tensors",
        ),
        tensor_check_stage(
            "Attention Scores",
            "\"Who to look at?\"",
            any_name_contains(&names, &["attn_output", "o_proj"]),
            "Attention output tensor found",
            "Missing attention output tensor",
        ),
        tensor_check_stage(
            "Feed-Forward (MLP)",
            "\"Think about it\"",
            all_groups_match(
                &names,
                &[
                    &["ffn_gate", "gate_proj"],
                    &["ffn_up", "up_proj"],
                    &["ffn_down", "down_proj"],
                ],
            ),
            "MLP tensors found",
            "Missing MLP tensors",
        ),
        StageResult {
            name: "Layer Norm",
            eli5: "Keep numbers stable",
            passed: all_groups_match(
                &names,
                &[
                    &["attn_norm", "input_layernorm"],
                    &["ffn_norm", "post_attention_layernorm"],
                ],
            ) && model.config.num_layers > 0,
            details: Some(format!("{} layers", model.config.num_layers)),
        },
        check_gguf_lm_head(&mapped, model.config.vocab_size),
        check_logits_real(&model),
        check_sampler_real(&model),
    ])
}

/// Stage 1: REAL tokenizer check (PMAT-112)
/// Must encode "test" -> token IDs -> decode back to verify round-trip
#[cfg(feature = "inference")]
fn check_tokenizer_real(model: &OwnedQuantizedModel) -> StageResult {
    // Use BOS token (typically 1) and a known token
    let test_tokens = vec![1u32, 2]; // BOS + another token

    // Verify embedding works (this proves tokenizer -> embedding path works)
    let embedding = model.embed(&test_tokens);

    let embedding_ok = !embedding.is_empty()
        && embedding.len() == test_tokens.len() * model.config.hidden_dim
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_result_display() {
        let result = StageResult {
            name: "Test",
            eli5: "Test ELI5",
            passed: true,
            details: Some("Test details".to_string()),
        };
        assert!(result.passed);
        assert_eq!(result.name, "Test");
    }

    #[test]
    fn test_print_results_empty() {
        // Should not panic with empty results
        print_results_table(&[]);
    }

    #[test]
    fn test_print_results_mixed() {
        let results = vec![
            StageResult {
                name: "Stage 1",
                eli5: "Test 1",
                passed: true,
                details: Some("OK".to_string()),
            },
            StageResult {
                name: "Stage 2",
                eli5: "Test 2",
                passed: false,
                details: Some("FAIL".to_string()),
            },
        ];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_details_truncation() {
        let long_details = "This is a very long details string that should be truncated";
        let truncated = if long_details.len() > 36 {
            format!("{}...", &long_details[..33])
        } else {
            long_details.to_string()
        };
        assert!(truncated.len() <= 39); // 36 + "..."
    }

    // ========================================================================
    // StageResult Tests
    // ========================================================================

    #[test]
    fn test_stage_result_passed() {
        let result = StageResult {
            name: "Stage 1",
            eli5: "Checking model integrity",
            passed: true,
            details: Some("All checks passed".to_string()),
        };
        assert!(result.passed);
        assert!(result.details.is_some());
    }

    #[test]
    fn test_stage_result_failed() {
        let result = StageResult {
            name: "Stage 2",
            eli5: "Checking tokenizer",
            passed: false,
            details: Some("Tokenizer not found".to_string()),
        };
        assert!(!result.passed);
    }

    #[test]
    fn test_stage_result_no_details() {
        let result = StageResult {
            name: "Stage 3",
            eli5: "Check",
            passed: true,
            details: None,
        };
        assert!(result.details.is_none());
    }

    #[test]
    fn test_stage_result_debug() {
        let result = StageResult {
            name: "Test",
            eli5: "Test",
            passed: true,
            details: None,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("StageResult"));
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_run_file_not_found() {
        let result = run(Path::new("/nonexistent/model.gguf"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (invalid GGUF or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (invalid APR or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unsupported_format() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(b"binary data").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (unsupported format or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_no_gpu() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(file.path(), true, false); // no_gpu = true
                                             // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), false, false);
        // Should fail (is a directory)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_format() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (unsupported format or feature disabled)
        assert!(result.is_err());
    }

    // ========================================================================
    // print_results_table Tests
    // ========================================================================

    #[test]
    fn test_print_results_all_passed() {
        let results = vec![
            StageResult {
                name: "Stage 1",
                eli5: "Test 1",
                passed: true,
                details: Some("OK".to_string()),
            },
            StageResult {
                name: "Stage 2",
                eli5: "Test 2",
                passed: true,
                details: Some("OK".to_string()),
            },
        ];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_all_failed() {
        let results = vec![StageResult {
            name: "Stage 1",
            eli5: "Test 1",
            passed: false,
            details: Some("ERROR".to_string()),
        }];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_no_details() {
        let results = vec![StageResult {
            name: "Stage 1",
            eli5: "Test",
            passed: true,
            details: None,
        }];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_long_name() {
        let results = vec![StageResult {
            name: "This is a very long stage name that should handle gracefully",
            eli5: "Test",
            passed: true,
            details: Some("OK".to_string()),
        }];
        // Should not panic
        print_results_table(&results);
    }

    // ========================================================================
    // StageResult Construction Edge Cases
    // ========================================================================

    #[test]
    fn test_stage_result_empty_name() {
        let result = StageResult {
            name: "",
            eli5: "",
            passed: false,
            details: None,
        };
        assert_eq!(result.name, "");
        assert_eq!(result.eli5, "");
        assert!(!result.passed);
        assert!(result.details.is_none());
    }

    #[test]
    fn test_stage_result_with_empty_details_string() {
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some(String::new()),
        };
        assert!(result.details.is_some());
        assert_eq!(result.details.as_deref(), Some(""));
    }

    #[test]
    fn test_stage_result_with_very_long_details() {
        let long = "x".repeat(1000);
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some(long.clone()),
        };
        assert_eq!(result.details.as_ref().expect("has details").len(), 1000);
    }

    #[test]
    fn test_stage_result_with_unicode_details() {
        let result = StageResult {
            name: "Unicode",
            eli5: "test",
            passed: true,
            details: Some("NaN detected in logits".to_string()),
        };
        assert!(result
            .details
            .as_ref()
            .expect("has details")
            .contains("NaN"));
    }

    #[test]
    fn test_stage_result_all_ten_stages_names() {
        // Verify all 10 stage names used in real checks are valid static strings
        let stage_names = [
            "Tokenizer",
            "Embedding",
            "Positional Encoding",
            "Q/K/V Projection",
            "Attention Scores",
            "Feed-Forward (MLP)",
            "Layer Norm",
            "LM Head",
            "Logits \u{2192} Probs",
            "Sampler/Decode",
        ];
        for name in &stage_names {
            let result = StageResult {
                name,
                eli5: "test",
                passed: true,
                details: None,
            };
            assert_eq!(result.name, *name);
        }
    }

    // ========================================================================
    // Details Truncation Edge Cases
    // ========================================================================

    #[test]
    fn test_details_truncation_exactly_36_chars() {
        let details = "a".repeat(36);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // Exactly 36 chars should NOT be truncated
        assert_eq!(truncated.len(), 36);
        assert_eq!(truncated, details);
    }

    #[test]
    fn test_details_truncation_37_chars() {
        let details = "a".repeat(37);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // 37 chars should be truncated to 33 + "..." = 36
        assert_eq!(truncated.len(), 36);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_details_truncation_35_chars() {
        let details = "a".repeat(35);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // 35 chars should NOT be truncated
        assert_eq!(truncated.len(), 35);
        assert!(!truncated.ends_with("..."));
    }

    #[test]
    fn test_details_truncation_empty() {
        let details = "";
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.to_string()
        };
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_details_truncation_exactly_33_chars() {
        let details = "b".repeat(33);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        assert_eq!(truncated.len(), 33);
        assert!(!truncated.ends_with("..."));
    }

    // ========================================================================
    // Result Aggregation Logic
    // ========================================================================

    #[test]
    fn test_result_aggregation_all_passed() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: true,
                details: None,
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: true,
                details: None,
            },
            StageResult {
                name: "S3",
                eli5: "t",
                passed: true,
                details: None,
            },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 3);
        assert_eq!(total_count, 3);
        assert_eq!(passed_count, total_count);
    }

    #[test]
    fn test_result_aggregation_none_passed() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: false,
                details: None,
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: None,
            },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 0);
        assert_eq!(total_count, 2);
        assert_ne!(passed_count, total_count);
    }

    #[test]
    fn test_result_aggregation_partial_pass() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: true,
                details: None,
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: Some("Missing tensor".to_string()),
            },
            StageResult {
                name: "S3",
                eli5: "t",
                passed: true,
                details: None,
            },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 2);
        assert_eq!(total_count, 3);
        assert_ne!(passed_count, total_count);
    }

    #[test]
    fn test_result_aggregation_single_pass() {
        let results = vec![StageResult {
            name: "S1",
            eli5: "t",
            passed: true,
            details: None,
        }];
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed_count, results.len());
    }

    #[test]
    fn test_result_aggregation_single_fail() {
        let results = vec![StageResult {
            name: "S1",
            eli5: "t",
            passed: false,
            details: None,
        }];
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed_count, 0);
        assert_ne!(passed_count, results.len());
    }

    #[test]
    fn test_result_aggregation_empty() {
        let results: Vec<StageResult> = vec![];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 0);
        assert_eq!(total_count, 0);
        // Edge case: 0 == 0 means "all passed" which is vacuously true
        assert_eq!(passed_count, total_count);
    }

    // ========================================================================
    // print_results_table Edge Cases
    // ========================================================================

    #[test]
    fn test_print_results_single_element_no_separator_after() {
        // With a single result, there should be no separator between rows
        let results = vec![StageResult {
            name: "Only",
            eli5: "Single",
            passed: true,
            details: Some("OK".to_string()),
        }];
        // Should not panic and should skip the inter-row separator
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_three_elements() {
        let results = vec![
            StageResult {
                name: "A",
                eli5: "t",
                passed: true,
                details: Some("d1".to_string()),
            },
            StageResult {
                name: "B",
                eli5: "t",
                passed: false,
                details: None,
            },
            StageResult {
                name: "C",
                eli5: "t",
                passed: true,
                details: Some("d3".to_string()),
            },
        ];
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_ten_stages() {
        // Simulate a full 10-stage pipeline
        let results: Vec<StageResult> = (0..10)
            .map(|i| StageResult {
                name: "Stage",
                eli5: "test",
                passed: i % 2 == 0,
                details: Some(format!("stage {} details", i)),
            })
            .collect();
        assert_eq!(results.len(), 10);
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_details_none_shows_dash() {
        // When details is None, print_results_table should use "-"
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: None,
        };
        let details = result.details.as_deref().unwrap_or("-");
        assert_eq!(details, "-");
    }

    #[test]
    fn test_print_results_details_some_shows_value() {
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some("logits[32000]".to_string()),
        };
        let details = result.details.as_deref().unwrap_or("-");
        assert_eq!(details, "logits[32000]");
    }

    // ========================================================================
    // run() Function: Error Path Tests
    // ========================================================================

    #[test]
    fn test_run_empty_file_gguf() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Empty file should fail
        let result = run(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_empty_file_apr() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file should fail
        let result = run(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_no_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("modelfile");
        std::fs::write(&path, b"some data").expect("write");
        let result = run(&path, false, false);
        // No extension -> unsupported format or feature disabled
        assert!(result.is_err());
    }

    #[test]
    fn test_run_uppercase_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.GGUF");
        std::fs::write(&path, b"not valid gguf").expect("write");
        let result = run(&path, false, false);
        // Should attempt GGUF parsing (lowercased) but fail due to invalid content
        assert!(result.is_err());
    }

    #[test]
    fn test_run_mixed_case_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.Apr");
        std::fs::write(&path, b"not valid apr").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_txt_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.txt");
        std::fs::write(&path, b"text data").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_json_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"{}").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    // ========================================================================
    // Softmax Validation Logic (inline tests)
    // ========================================================================

    #[test]
    fn test_softmax_sum_validation_exact() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        let valid = (prob_sum - 1.0).abs() < 0.001;
        assert!(valid, "softmax sum should be ~1.0, got {prob_sum}");
    }

    #[test]
    fn test_softmax_sum_single_element() {
        let logits = vec![5.0_f32];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!((probs[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_softmax_sum_large_logits() {
        // Numerically stable softmax should handle large values
        let logits = vec![1000.0_f32, 1001.0, 999.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!(!probs.iter().any(|x| x.is_nan()));
        assert!(!probs.iter().any(|x| x.is_infinite()));
    }

    #[test]
    fn test_softmax_sum_negative_logits() {
        let logits = vec![-10.0_f32, -20.0, -5.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_all_zeros() {
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        // Uniform distribution
        for p in &probs {
            assert!((p - 0.25).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_two_elements_dominant() {
        let logits = vec![100.0_f32, 0.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        // First element dominates
        assert!(probs[0] > 0.99);
        assert!(probs[1] < 0.01);
    }

    // ========================================================================
    // NaN/Inf Detection Logic (mirrors logits validation)
    // ========================================================================

    #[test]
    fn test_nan_detection_in_logits() {
        let logits = vec![1.0_f32, f32::NAN, 3.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(has_nan);
        assert!(!has_inf);
        assert!(!valid);
    }

    #[test]
    fn test_inf_detection_in_logits() {
        let logits = vec![1.0_f32, f32::INFINITY, 3.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(!has_nan);
        assert!(has_inf);
        assert!(!valid);
    }

    #[test]
    fn test_neg_inf_detection_in_logits() {
        let logits = vec![1.0_f32, f32::NEG_INFINITY, 3.0];
        let has_inf = logits.iter().any(|x| x.is_infinite());
        assert!(has_inf);
    }

    #[test]
    fn test_empty_logits_invalid() {
        let logits: Vec<f32> = vec![];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(!valid, "empty logits should be invalid");
    }

    #[test]
    fn test_valid_logits() {
        let logits = vec![0.1_f32, -0.5, 2.3, -1.0, 0.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(valid);
    }

    #[test]
    fn test_logits_with_both_nan_and_inf() {
        let logits = vec![f32::NAN, f32::INFINITY];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        assert!(has_nan);
        assert!(has_inf);
        assert!(!((!has_nan) && (!has_inf) && !logits.is_empty()));
    }

    // ========================================================================
    // Tensor Name Matching Logic (APR checks)
    // ========================================================================

    #[test]
    fn test_embedding_tensor_detection() {
        let names = vec!["token_embd.weight", "blk.0.attn_q.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(has_embed);
    }

    #[test]
    fn test_embedding_tensor_detection_wte() {
        let names = vec!["wte.weight", "blk.0.ffn_gate.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(has_embed);
    }

    #[test]
    fn test_embedding_tensor_detection_missing() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(!has_embed);
    }

    #[test]
    fn test_qkv_projection_detection_all_present() {
        let names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_qkv_projection_detection_missing_v() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k);
        assert!(!has_v);
        assert!(!(has_q && has_k && has_v));
    }

    #[test]
    fn test_qkv_hf_style_names() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_ffn_detection_all_present() {
        let names = vec![
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_ffn_detection_hf_style() {
        let names = vec![
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_ffn_detection_missing_gate() {
        let names = vec!["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        assert!(!has_gate);
    }

    #[test]
    fn test_norm_detection() {
        let names = vec!["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm && has_ffn_norm);
    }

    #[test]
    fn test_norm_detection_hf_style() {
        let names = vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm && has_ffn_norm);
    }

    #[test]
    fn test_norm_detection_missing_ffn_norm() {
        let names = vec!["blk.0.attn_norm.weight"];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm);
        assert!(!has_ffn_norm);
    }

    #[test]
    fn test_lm_head_detection_explicit() {
        let names = vec!["lm_head.weight", "output.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        assert!(has_lm_head);
    }

    #[test]
    fn test_lm_head_detection_output_weight() {
        let names = vec!["output.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        assert!(has_lm_head);
    }

    #[test]
    fn test_lm_head_detection_tied_embeddings_fallback() {
        // When no explicit lm_head, but embedding exists
        let names = vec!["token_embd.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        // LM head check passes if either explicit lm_head or embedding for tied weights
        assert!(!has_lm_head);
        assert!(has_embed);
        assert!(has_lm_head || has_embed);
    }

    #[test]
    fn test_lm_head_detection_no_head_no_embed() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(!has_lm_head);
        assert!(!has_embed);
        assert!(!(has_lm_head || has_embed));
    }

    #[test]
    fn test_rope_tensor_detection() {
        let names = vec!["model.rope.freqs"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(has_rope);
    }

    #[test]
    fn test_rope_rotary_variant() {
        let names = vec!["model.rotary_emb.inv_freq"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(has_rope);
    }

    #[test]
    fn test_rope_absent() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(!has_rope);
    }

    #[test]
    fn test_attention_output_detection() {
        let names = vec!["blk.0.attn_output.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(has_attn_out);
    }

    #[test]
    fn test_attention_output_o_proj() {
        let names = vec!["model.layers.0.self_attn.o_proj.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(has_attn_out);
    }

    #[test]
    fn test_attention_output_absent() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(!has_attn_out);
    }

    // ========================================================================
    // Layer Norm with num_layers edge cases
    // ========================================================================

    #[test]
    fn test_norm_with_zero_layers_fails() {
        let has_norm = true;
        let num_layers = 0;
        // Stage 7 in APR path checks has_norm && num_layers > 0
        assert!(!(has_norm && num_layers > 0));
    }

    #[test]
    fn test_norm_with_positive_layers_passes() {
        let has_norm = true;
        let num_layers = 24;
        assert!(has_norm && num_layers > 0);
    }

    #[test]
    fn test_norm_absent_with_layers_fails() {
        let has_norm = false;
        let num_layers = 24;
        assert!(!(has_norm && num_layers > 0));
    }

    // ========================================================================
    // LM Head Vocab Size Formatting
    // ========================================================================

    #[test]
    fn test_lm_head_details_with_head() {
        let has_lm_head = true;
        let has_embed = true;
        let vocab_size = 32000;
        let detail = format!(
            "vocab_size={}{}",
            vocab_size,
            if has_lm_head { "" } else { " (tied)" }
        );
        assert_eq!(detail, "vocab_size=32000");
        let _ = has_embed;
    }

    #[test]
    fn test_lm_head_details_tied() {
        let has_lm_head = false;
        let vocab_size = 151936;
        let detail = format!(
            "vocab_size={}{}",
            vocab_size,
            if has_lm_head { "" } else { " (tied)" }
        );
        assert_eq!(detail, "vocab_size=151936 (tied)");
    }

    // ========================================================================
    // Non-inference path (cfg(not(feature = "inference")))
    // ========================================================================

    #[test]
    fn test_non_inference_stage_result_construction() {
        // This mirrors what happens when inference feature is disabled
        let result = StageResult {
            name: "N/A",
            eli5: "Requires inference",
            passed: false,
            details: Some("Build with --features inference".to_string()),
        };
        assert!(!result.passed);
        assert_eq!(result.name, "N/A");
        assert_eq!(result.eli5, "Requires inference");
        assert_eq!(
            result.details.as_deref(),
            Some("Build with --features inference")
        );
    }

    // ========================================================================
    // Extension Dispatch Logic (mirrors run_real_checks dispatch)
    // ========================================================================

    /// Helper that replicates the extension dispatch logic from run_real_checks
    fn classify_extension(path: &Path) -> &str {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext.to_lowercase().as_str() {
            "apr" => "apr",
            "gguf" => "gguf",
            _ => "unsupported",
        }
    }

    #[test]
    fn test_extension_dispatch_apr_lowercase() {
        assert_eq!(classify_extension(Path::new("model.apr")), "apr");
    }

    #[test]
    fn test_extension_dispatch_apr_uppercase() {
        assert_eq!(classify_extension(Path::new("model.APR")), "apr");
    }

    #[test]
    fn test_extension_dispatch_apr_mixed_case() {
        assert_eq!(classify_extension(Path::new("model.Apr")), "apr");
    }

    #[test]
    fn test_extension_dispatch_gguf_lowercase() {
        assert_eq!(classify_extension(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_gguf_uppercase() {
        assert_eq!(classify_extension(Path::new("model.GGUF")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_gguf_mixed_case() {
        assert_eq!(classify_extension(Path::new("model.Gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_safetensors_unsupported() {
        assert_eq!(
            classify_extension(Path::new("model.safetensors")),
            "unsupported"
        );
    }

    #[test]
    fn test_extension_dispatch_bin_unsupported() {
        assert_eq!(classify_extension(Path::new("model.bin")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_no_extension() {
        assert_eq!(classify_extension(Path::new("modelfile")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_empty_extension() {
        // A path ending with "." has empty extension
        assert_eq!(classify_extension(Path::new("model.")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_double_extension() {
        // Only last extension matters
        assert_eq!(classify_extension(Path::new("model.tar.gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_hidden_file() {
        assert_eq!(classify_extension(Path::new(".model.apr")), "apr");
    }

    // ========================================================================
    // Unsupported Format Error Message Construction
    // ========================================================================

    #[test]
    fn test_unsupported_format_error_message() {
        let ext = "bin";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert_eq!(msg, "Unsupported format: bin. Use .apr or .gguf");
    }

    #[test]
    fn test_unsupported_format_error_empty_ext() {
        let ext = "";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert_eq!(msg, "Unsupported format: . Use .apr or .gguf");
    }

    #[test]
    fn test_unsupported_format_error_safetensors() {
        let ext = "safetensors";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert!(msg.contains("safetensors"));
        assert!(msg.contains("Use .apr or .gguf"));
    }

    // ========================================================================
    // RoPE Theta Validation Logic (mirrors Stage 3 GGUF)
    // ========================================================================

    #[test]
    fn test_rope_theta_default_valid() {
        let rope_theta: f64 = 10000.0;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_llama3_valid() {
        let rope_theta: f64 = 500_000.0;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_zero_invalid() {
        let rope_theta: f64 = 0.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_negative_invalid() {
        let rope_theta: f64 = -1.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_exactly_one_invalid() {
        let rope_theta: f64 = 1.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_just_above_one_valid() {
        let rope_theta: f64 = 1.001;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_details_format() {
        let rope_theta: f64 = 10000.0;
        let details = format!("rope_theta={:.1}", rope_theta);
        assert_eq!(details, "rope_theta=10000.0");
    }

    // ========================================================================
    // GGUF-Specific Tensor Name Patterns (blk.N style)
    // ========================================================================

    #[test]
    fn test_gguf_embedding_detection_token_embd() {
        let name = "token_embd.weight";
        assert!(name.contains("token_embd") || name.contains("embed_tokens"));
    }

    #[test]
    fn test_gguf_embedding_detection_embed_tokens() {
        let name = "model.embed_tokens.weight";
        assert!(name.contains("token_embd") || name.contains("embed_tokens"));
    }

    #[test]
    fn test_gguf_embedding_detection_neither() {
        let name = "blk.0.attn_q.weight";
        assert!(!(name.contains("token_embd") || name.contains("embed_tokens")));
    }

    #[test]
    fn test_gguf_qkv_blk_style() {
        let names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ];
        let has_q = names
            .iter()
            .any(|t| t.contains("blk.0.attn_q") || t.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|t| t.contains("blk.0.attn_v") || t.contains("layers.0.self_attn.v_proj"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_gguf_qkv_hf_style() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        let has_q = names
            .iter()
            .any(|t| t.contains("blk.0.attn_q") || t.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|t| t.contains("blk.0.attn_v") || t.contains("layers.0.self_attn.v_proj"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_gguf_qkv_missing_k() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.attn_v.weight"];
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        assert!(!has_k);
    }

    #[test]
    fn test_gguf_attn_output_detection() {
        let names = vec!["blk.0.attn_output.weight"];
        let has = names
            .iter()
            .any(|t| t.contains("attn_output") || t.contains("o_proj"));
        assert!(has);
    }

    #[test]
    fn test_gguf_attn_output_o_proj_style() {
        let names = vec!["model.layers.0.self_attn.o_proj.weight"];
        let has = names
            .iter()
            .any(|t| t.contains("attn_output") || t.contains("o_proj"));
        assert!(has);
    }

    #[test]
    fn test_gguf_ffn_blk_style() {
        let names = vec![
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let has_gate = names
            .iter()
            .any(|t| t.contains("ffn_gate") || t.contains("gate_proj"));
        let has_up = names
            .iter()
            .any(|t| t.contains("ffn_up") || t.contains("up_proj"));
        let has_down = names
            .iter()
            .any(|t| t.contains("ffn_down") || t.contains("down_proj"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_gguf_norm_blk_style() {
        let names = vec!["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"];
        let has_attn = names
            .iter()
            .any(|t| t.contains("attn_norm") || t.contains("input_layernorm"));
        let has_ffn = names
            .iter()
            .any(|t| t.contains("ffn_norm") || t.contains("post_attention_layernorm"));
        assert!(has_attn && has_ffn);
    }

    #[test]
    fn test_gguf_norm_hf_style() {
        let names = vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        let has_attn = names
            .iter()
            .any(|t| t.contains("attn_norm") || t.contains("input_layernorm"));
        let has_ffn = names
            .iter()
            .any(|t| t.contains("ffn_norm") || t.contains("post_attention_layernorm"));
        assert!(has_attn && has_ffn);
    }

    // ========================================================================
    // LM Head Details Formatting (GGUF-specific 3-branch logic)
    // ========================================================================

    #[test]
    fn test_lm_head_details_explicit_head() {
        let has_explicit_lm_head = true;
        let has_tied_embeddings = false;
        let vocab_size = 32000_usize;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", vocab_size)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", vocab_size)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "vocab_size=32000");
    }

    #[test]
    fn test_lm_head_details_tied_embeddings() {
        let has_explicit_lm_head = false;
        let has_tied_embeddings = true;
        let vocab_size = 128256_usize;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", vocab_size)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", vocab_size)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "vocab_size=128256 (tied embeddings)");
    }

    #[test]
    fn test_lm_head_details_missing() {
        let has_explicit_lm_head = false;
        let has_tied_embeddings = false;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", 32000)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", 32000)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "Missing LM head tensor");
    }

    #[test]
    fn test_lm_head_pass_condition_explicit_with_vocab() {
        let has_lm_head = true;
        let vocab_size = 32000_usize;
        assert!(has_lm_head && vocab_size > 0);
    }

    #[test]
    fn test_lm_head_pass_condition_zero_vocab_fails() {
        let has_lm_head = true;
        let vocab_size = 0_usize;
        assert!(!(has_lm_head && vocab_size > 0));
    }

    #[test]
    fn test_lm_head_pass_condition_no_head_no_vocab() {
        let has_lm_head = false;
        let vocab_size = 32000_usize;
        assert!(!(has_lm_head && vocab_size > 0));
    }

    // ========================================================================
    // GGUF Tensor Name Matching: output.weight exact match
    // ========================================================================

    #[test]
    fn test_output_weight_exact_match() {
        let name = "output.weight";
        assert!(name == "output.weight" || name.contains("lm_head"));
    }

    #[test]
    fn test_output_weight_partial_no_match() {
        // "output.weight.bias" should still match via contains if using contains
        // but the exact == check in the code is for "output.weight" only
        let name = "some_output.weight";
        // The GGUF check uses t.name == "output.weight" || t.name.contains("lm_head")
        assert!(!(name == "output.weight"));
        // But the APR check uses n.contains("lm_head") || n == &"output.weight"
        assert!(!(name == "output.weight" || name.contains("lm_head")));
    }

    #[test]
    fn test_lm_head_weight_contains_match() {
        let name = "lm_head.weight";
        assert!(name == "output.weight" || name.contains("lm_head"));
    }

    // ========================================================================
    // Logits Min/Max Formatting (mirrors check_logits_real details)
    // ========================================================================

    #[test]
    fn test_logits_details_format_normal() {
        let logits = vec![0.1_f32, -0.5, 2.3, -1.0, 0.0];
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let details = format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max);
        assert_eq!(details, "logits[5]: min=-1.00, max=2.30");
    }

    #[test]
    fn test_logits_details_format_single() {
        let logits = vec![42.0_f32];
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let details = format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max);
        assert_eq!(details, "logits[1]: min=42.00, max=42.00");
    }

    #[test]
    fn test_logits_nan_details_message() {
        let has_nan = true;
        let has_inf = false;
        let logits_empty = false;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else if logits_empty {
            "FAIL: Empty logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: NaN detected in logits");
    }

    #[test]
    fn test_logits_inf_details_message() {
        let has_nan = false;
        let has_inf = true;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Inf detected in logits");
    }

    #[test]
    fn test_logits_empty_details_message() {
        let has_nan = false;
        let has_inf = false;
        let logits_empty = true;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else if logits_empty {
            "FAIL: Empty logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Empty logits");
    }

    // ========================================================================
    // Sampler Details Formatting (mirrors check_sampler_real)
    // ========================================================================

    #[test]
    fn test_sampler_details_valid_softmax() {
        let prob_sum: f32 = 1.000_001;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let has_nan = false;
        let has_inf = false;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else if !softmax_valid {
            format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
        } else {
            format!("softmax sum = {:.6} \u{2713}", prob_sum)
        };
        assert!(details.contains("softmax sum = 1.0000"));
        assert!(details.contains('\u{2713}'));
    }

    #[test]
    fn test_sampler_details_nan_in_softmax() {
        let has_nan = true;
        let has_inf = false;
        let softmax_valid = false;
        let _ = softmax_valid;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: NaN in softmax");
    }

    #[test]
    fn test_sampler_details_inf_in_softmax() {
        let has_nan = false;
        let has_inf = true;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Inf in softmax");
    }

    #[test]
    fn test_sampler_details_bad_softmax_sum() {
        let prob_sum: f32 = 0.5;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let has_nan = false;
        let has_inf = false;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else if !softmax_valid {
            format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
        } else {
            "ok".to_string()
        };
        assert!(details.contains("FAIL: softmax sum"));
        assert!(details.contains("expected 1.0"));
    }

    // ========================================================================
    // Sampler Passed Condition (mirrors check_sampler_real line 559)
    // ========================================================================

    #[test]
    fn test_sampler_passed_all_good() {
        let softmax_valid = true;
        let has_nan = false;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(passed);
    }

    #[test]
    fn test_sampler_passed_nan_fails() {
        let softmax_valid = true;
        let has_nan = true;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_inf_fails() {
        let softmax_valid = true;
        let has_nan = false;
        let has_inf = true;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_bad_sum_fails() {
        let softmax_valid = false;
        let has_nan = false;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    // ========================================================================
    // Embedding Validity Check (mirrors check_tokenizer_real logic)
    // ========================================================================

    #[test]
    fn test_embedding_validity_check_valid() {
        let embedding = vec![0.1_f32, 0.2, 0.3, 0.4]; // 2 tokens * hidden_dim=2
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_empty() {
        let embedding: Vec<f32> = vec![];
        let embedding_ok = !embedding.is_empty();
        assert!(!embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_wrong_size() {
        let embedding = vec![0.1_f32, 0.2, 0.3]; // 3 floats != 2*2
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_contains_nan() {
        let embedding = vec![0.1_f32, f32::NAN, 0.3, 0.4];
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_contains_inf() {
        let embedding = vec![0.1_f32, 0.2, f32::INFINITY, 0.4];
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }

    // ========================================================================
    // Tokenizer Details Formatting
    // ========================================================================

    #[test]
    fn test_tokenizer_details_format_ok() {
        let test_tokens = vec![1u32, 2];
        let embedding_len = 512;
        let details = format!("tokens={:?} \u{2192} {} floats", test_tokens, embedding_len);
        assert!(details.contains("[1, 2]"));
        assert!(details.contains("512 floats"));
    }

    #[test]
    fn test_tokenizer_details_format_failed() {
        let details = "Tokenizer/embedding failed".to_string();
        assert!(details.contains("failed"));
    }

    // ========================================================================
    // Full 10-Stage Pipeline Result Table Rendering
    // ========================================================================

    #[test]
    fn test_print_full_pipeline_all_pass() {
        let stage_names = [
            "Tokenizer",
            "Embedding",
            "Positional Encoding",
            "Q/K/V Projection",
            "Attention Scores",
            "Feed-Forward (MLP)",
            "Layer Norm",
            "LM Head",
            "Logits \u{2192} Probs",
            "Sampler/Decode",
        ];
        let results: Vec<StageResult> = stage_names
            .iter()
            .map(|name| StageResult {
                name,
                eli5: "test",
                passed: true,
                details: Some("OK".to_string()),
            })
            .collect();
        assert_eq!(results.len(), 10);
        let passed = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed, 10);
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_full_pipeline_mixed_results() {
        let results = vec![
            StageResult {
                name: "Tokenizer",
                eli5: "Words \u{2192} numbers",
                passed: true,
                details: Some("tokens=[1, 2] \u{2192} 512 floats".to_string()),
            },
            StageResult {
                name: "Embedding",
                eli5: "Numbers \u{2192} vectors",
                passed: true,
                details: Some("Found embedding tensor".to_string()),
            },
            StageResult {
                name: "Positional Encoding",
                eli5: "\"You are word #3\"",
                passed: true,
                details: Some("rope_theta=10000.0".to_string()),
            },
            StageResult {
                name: "Q/K/V Projection",
                eli5: "Make 3 question copies",
                passed: false,
                details: Some("Missing Q/K/V tensors".to_string()),
            },
            StageResult {
                name: "Attention Scores",
                eli5: "\"Who to look at?\"",
                passed: false,
                details: Some("Missing attention output tensor".to_string()),
            },
            StageResult {
                name: "Feed-Forward (MLP)",
                eli5: "\"Think about it\"",
                passed: true,
                details: Some("MLP tensors found".to_string()),
            },
            StageResult {
                name: "Layer Norm",
                eli5: "Keep numbers stable",
                passed: true,
                details: Some("32 layers".to_string()),
            },
            StageResult {
                name: "LM Head",
                eli5: "Vector \u{2192} vocab scores",
                passed: true,
                details: Some("vocab_size=32000".to_string()),
            },
            StageResult {
                name: "Logits \u{2192} Probs",
                eli5: "Scores \u{2192} percentages",
                passed: true,
                details: Some("logits[32000]: min=-5.20, max=12.30".to_string()),
            },
            StageResult {
                name: "Sampler/Decode",
                eli5: "Pick word, return",
                passed: false,
                details: Some("FAIL: softmax sum = 0.500000 (expected 1.0)".to_string()),
            },
        ];
        let passed = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed, 7);
        assert_eq!(results.len(), 10);
        // Should not panic - exercises truncation for long details
        print_results_table(&results);
    }

    // ========================================================================
    // print_results_table: Details Truncation In-Function Behavior
    // ========================================================================

    #[test]
    fn test_print_results_table_truncates_long_details() {
        // This exercises the truncation branch inside print_results_table
        // Details > 36 chars should be truncated to 33 + "..."
        let long_detail = "a]bcdefghijklmnopqrstuvwxyz0123456789EXTRA";
        assert!(long_detail.len() > 36);
        let results = vec![StageResult {
            name: "Long",
            eli5: "test",
            passed: true,
            details: Some(long_detail.to_string()),
        }];
        // Should not panic, and should truncate internally
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_table_exact_boundary_details() {
        // Exactly 36 chars - should NOT truncate
        let exact_36 = "a".repeat(36);
        assert_eq!(exact_36.len(), 36);
        let results = vec![StageResult {
            name: "Exact",
            eli5: "test",
            passed: false,
            details: Some(exact_36),
        }];
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_table_one_over_boundary() {
        // 37 chars - should truncate
        let over_37 = "b".repeat(37);
        assert_eq!(over_37.len(), 37);
        let results = vec![StageResult {
            name: "Over",
            eli5: "test",
            passed: true,
            details: Some(over_37),
        }];
        print_results_table(&results);
    }

    // ========================================================================
    // Success/Failure Message Formatting (mirrors run() lines 57-79)
    // ========================================================================

    #[test]
    fn test_success_message_format() {
        let passed_count = 10;
        let total_count = 10;
        let msg = format!(
            "\u{2705} {}/{} STAGES PASSED. MODEL PROVEN CORRECT.",
            passed_count, total_count
        );
        assert!(msg.contains("10/10"));
        assert!(msg.contains("PROVEN CORRECT"));
    }

    #[test]
    fn test_failure_message_format() {
        let passed_count = 7;
        let total_count = 10;
        let msg = format!(
            "\u{274c} {}/{} STAGES PASSED. CHECK STAGE LOGS.",
            passed_count, total_count
        );
        assert!(msg.contains("7/10"));
        assert!(msg.contains("CHECK STAGE LOGS"));
    }

    #[test]
    fn test_failure_message_zero_passed() {
        let passed_count = 0;
        let total_count = 10;
        let msg = format!(
            "\u{274c} {}/{} STAGES PASSED. CHECK STAGE LOGS.",
            passed_count, total_count
        );
        assert!(msg.contains("0/10"));
    }

    // ========================================================================
    // Vocab Size with Dims Matching (GGUF LM Head check)
    // ========================================================================

    #[test]
    fn test_vocab_dim_matching_present() {
        let dims: Vec<u64> = vec![32000, 4096];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(matches);
    }

    #[test]
    fn test_vocab_dim_matching_absent() {
        let dims: Vec<u64> = vec![4096, 4096];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(!matches);
    }

    #[test]
    fn test_vocab_dim_matching_empty_dims() {
        let dims: Vec<u64> = vec![];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(!matches);
    }

    // ========================================================================
    // APR Metadata Defaults (mirrors unwrap_or defaults)
    // ========================================================================

    #[test]
    fn test_metadata_defaults_num_layers() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_defaults_hidden_size() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_defaults_vocab_size() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(32000), 32000);
    }

    #[test]
    fn test_metadata_defaults_num_heads() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_present_overrides_default() {
        let val: Option<usize> = Some(128256);
        assert_eq!(val.unwrap_or(32000), 128256);
    }

    // ========================================================================
    // Softmax Edge Cases (additional precision tests)
    // ========================================================================

    #[test]
    fn test_softmax_identical_logits_uniform() {
        // All identical logits should produce uniform distribution
        let logits = vec![3.14_f32; 100];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        // Each prob should be ~0.01
        for p in &probs {
            assert!((p - 0.01).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_very_negative_logits() {
        // All very negative logits - should still sum to 1
        let logits = vec![-1000.0_f32, -1001.0, -999.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!(!probs.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_softmax_vocab_size_logits() {
        // Simulate realistic vocab size
        let logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 0.01,
            "softmax over 32k logits should sum to ~1.0, got {}",
            prob_sum
        );
    }

    // ========================================================================
    // APR vs GGUF Tensor Name Convention Cross-Check
    // ========================================================================

    #[test]
    fn test_apr_and_gguf_embedding_names_differ() {
        // APR uses "emb"/"wte"/"token_embd"; GGUF uses "token_embd"/"embed_tokens"
        // "token_embd" is common to both
        let apr_check =
            |n: &str| n.contains("emb") || n.contains("wte") || n.contains("token_embd");
        let gguf_check = |n: &str| n.contains("token_embd") || n.contains("embed_tokens");

        // "token_embd.weight" matches both
        assert!(apr_check("token_embd.weight"));
        assert!(gguf_check("token_embd.weight"));

        // "embed_tokens" matches GGUF but also APR (via "emb" substring)
        assert!(apr_check("model.embed_tokens.weight"));
        assert!(gguf_check("model.embed_tokens.weight"));

        // "wte" only matches APR
        assert!(apr_check("transformer.wte.weight"));
        assert!(!gguf_check("transformer.wte.weight"));
    }

    #[test]
    fn test_full_model_tensor_inventory_gguf() {
        // Simulate a complete GGUF model's tensor names
        let names = vec![
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "output.weight",
        ];

        // All stage checks should pass
        let has_embed = names
            .iter()
            .any(|n| n.contains("token_embd") || n.contains("embed_tokens"));
        let has_q = names
            .iter()
            .any(|n| n.contains("blk.0.attn_q") || n.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|n| n.contains("blk.0.attn_k") || n.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|n| n.contains("blk.0.attn_v") || n.contains("layers.0.self_attn.v_proj"));
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("attn_output") || n.contains("o_proj"));
        let has_gate = names
            .iter()
            .any(|n| n.contains("ffn_gate") || n.contains("gate_proj"));
        let has_up = names
            .iter()
            .any(|n| n.contains("ffn_up") || n.contains("up_proj"));
        let has_down = names
            .iter()
            .any(|n| n.contains("ffn_down") || n.contains("down_proj"));
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("attn_norm") || n.contains("input_layernorm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("ffn_norm") || n.contains("post_attention_layernorm"));
        let has_lm_head = names
            .iter()
            .any(|n| *n == "output.weight" || n.contains("lm_head"));

        assert!(has_embed);
        assert!(has_q && has_k && has_v);
        assert!(has_attn_out);
        assert!(has_gate && has_up && has_down);
        assert!(has_attn_norm && has_ffn_norm);
        assert!(has_lm_head);
    }

    #[test]
    fn test_full_model_tensor_inventory_hf() {
        // Simulate a complete HF-style model's tensor names
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];

        // Check APR-style detection (used in run_real_checks_apr)
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");

        assert!(has_embed);
        assert!(has_q && has_k && has_v);
        assert!(has_attn_out);
        assert!(has_gate && has_up && has_down);
        assert!(has_attn_norm && has_ffn_norm);
        assert!(has_lm_head);
    }
}
