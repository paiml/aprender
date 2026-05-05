//! `apr trace --save-tensor` end-to-end dispatch for APR models.
//!
//! Wires the `--save-tensor` clap surface (PR-A #1405) all the way to
//! [`AprTransformer::forward_traced_with_save_tensor`] (PR-C-real step 1
//! #1408 + step 2 #1414). Before this module, the CLI only printed a
//! stub message and never invoked the wrapper — so the existing
//! `Embedding` and `LmHead` capture surface was unreachable.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] —
//! `cli_signature` invariant: `apr trace --payload <model> --save-tensor
//! <STAGE>[,<STAGE>...] [--save-tensor-dir <DIR>] [--save-tensor-layers
//! <RANGE>]` MUST produce per-stage f32 LE files with APRT headers.
//!
//! ## Scope
//!
//! - Handles the `.apr` extension only. The `.gguf` and `.safetensors`
//!   trace paths still print the stub; SHIP-007 PR-E live diagnostics
//!   convert GGUF→APR at import boundary so the canonical 7B teacher
//!   bisection runs through this code path.
//! - Uses a fixed test prompt ("What is 2+2?") matching the existing
//!   `run_traced_inference_apr` pattern in `vector_stats.rs`. A future
//!   `--prompt` CLI flag is a follow-up; for the SHIP-007 bisection a
//!   stable prompt is what makes APR vs GGUF byte comparison meaningful.

#![cfg(feature = "inference")]

use std::path::{Path, PathBuf};

use crate::error::CliError;

/// Default output directory when the user does not pass `--save-tensor-dir`.
///
/// Mirrors the convention in the contract's `cli_signature` equation:
/// `<model-stem>-trace/` next to the input model file.
fn default_output_dir(path: &Path) -> PathBuf {
    let parent = match path.parent() {
        Some(p) if p.as_os_str().is_empty() => Path::new("."),
        Some(p) => p,
        None => Path::new("."),
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");
    parent.join(format!("{stem}-trace"))
}

/// Run `apr trace --payload <APR> --save-tensor <STAGES>` end-to-end.
///
/// Loads the APR model, tokenizes a fixed test prompt, runs
/// [`AprTransformer::forward_traced_with_save_tensor`] with a plan
/// derived from the CLI args, and prints a summary listing the files
/// that landed under `output_dir`.
///
/// # Errors
///
/// - [`CliError::ValidationFailed`] if the plan args are malformed
///   (unknown stage name, bad layer range).
/// - [`CliError::ModelLoadFailed`] if the APR file cannot be loaded.
/// - [`CliError::InferenceFailed`] if the embedded BPE tokenizer is
///   missing or the forward pass errors.
pub fn run_save_tensor_apr(
    path: &Path,
    stages: &str,
    dir: Option<&Path>,
    layers: &str,
) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::AprTransformer;
    use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;

    let output_dir = match dir {
        Some(p) => p.to_path_buf(),
        None => default_output_dir(path),
    };

    let plan = SaveTensorPlan::from_cli(stages, layers, output_dir.clone()).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr trace --save-tensor: bad plan args (stages={stages:?}, layers={layers:?}): {e:?}"
        ))
    })?;

    println!("{}", "=== apr trace --save-tensor (APR) ===".cyan().bold());
    println!("Model:        {}", path.display());
    println!("Stages:       {stages}");
    println!("Layers:       {layers}");
    println!("Output dir:   {}", output_dir.display());
    println!();

    // Load embedded tokenizer + encode a fixed test prompt.
    let model = AprV2Model::load(path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;
    let test_prompt = "What is 2+2?";
    let test_tokens: Vec<u32> = match model.load_embedded_bpe_tokenizer() {
        Some(tokenizer) => tokenizer.encode(test_prompt),
        None => {
            return Err(CliError::InferenceFailed(
                "FATAL: APR file has no embedded tokenizer. Cannot trace without proper \
                 tokenization. Re-import with: apr import <source>.gguf -o <output>.apr"
                    .to_string(),
            ));
        }
    };
    println!("Test prompt:  {test_prompt:?}");
    println!(
        "Token ids:    {test_tokens:?} ({} tokens)",
        test_tokens.len()
    );
    println!();

    // Load AprTransformer (the path that supports forward_traced).
    let transformer = AprTransformer::from_apr_file(path).map_err(|e| {
        CliError::InferenceFailed(format!("AprTransformer::from_apr_file failed: {e}"))
    })?;

    // SHIP-007 PR-C-real step 3 LIVE: forward_traced_with_save_tensor delegates
    // to forward_traced_with_plan which threads the SaveTensorPlan through every
    // capture point in a single forward pass — Embedding, AttnNorm, QkvMatmul,
    // QkvBias, Attention, AttnOut, PostAttnResidual, FfnNorm, FfnGate, FfnUp,
    // FfnSilu, FfnSwigl, FfnOut, PostFfnResidual per-layer + FinalNorm + LmHead
    // whole-model.
    let trace = transformer
        .forward_traced_with_save_tensor(&test_tokens, &plan)
        .map_err(|e| {
            CliError::InferenceFailed(format!("forward_traced_with_save_tensor failed: {e}"))
        })?;

    // Walk output_dir to print what landed.
    let mut written: Vec<PathBuf> = Vec::new();
    collect_bin_files(&output_dir, &mut written).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot enumerate {}: {e}", output_dir.display()))
    })?;
    written.sort();

    println!(
        "{} {} stage tensor file(s):",
        "Wrote".green().bold(),
        written.len()
    );
    for p in &written {
        let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({} bytes)", p.display(), bytes);
    }
    println!();
    println!(
        "Forward pass succeeded — {} layer activations, {} logits",
        trace.layer_activations.len(),
        trace.logits.len()
    );
    println!(
        "Stages captured in single forward pass via SHIP-007 PR-C-real step 3 \
         (Embedding/AttnNorm/QkvMatmul/QkvBias/Attention/AttnOut/PostAttnResidual/\
         FfnNorm/FfnGate/FfnUp/FfnSilu/FfnSwigl/FfnOut/PostFfnResidual per-layer \
         + FinalNorm/LmHead whole-model)."
    );

    Ok(())
}

/// Run `apr trace --save-tensor <STAGES> <GGUF>` end-to-end for a Qwen3-MoE
/// arch GGUF model.
///
/// M-MOE-SUB-2 step (a) CLI completion (companion-spec M77): wires the
/// `--save-tensor` / `--save-tensor-layers` / `--save-tensor-dir` clap
/// fields through to
/// [`OwnedQuantizedModel::forward_qwen3_moe_traced_with_plan`] (M74,
/// aprender PR #1516 squash `3138d134d`).
///
/// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.1.0 step (a). The
/// CPU-traced sibling captures `MoeRouter` + `MoeFfnOut` per-layer when
/// the plan selects them; this function exists so an operator can
/// invoke the wireup from the CLI without re-authoring the dispatch
/// every test.
///
/// # Errors
///
/// - [`CliError::ValidationFailed`] if the plan args are malformed.
/// - [`CliError::ModelLoadFailed`] if the GGUF file cannot be loaded
///   or the arch is not `qwen3_moe`.
/// - [`CliError::InferenceFailed`] if the forward pass errors.
pub fn run_save_tensor_gguf_moe(
    path: &Path,
    stages: &str,
    dir: Option<&Path>,
    layers: &str,
) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;

    let output_dir = match dir {
        Some(p) => p.to_path_buf(),
        None => default_output_dir(path),
    };

    let plan = SaveTensorPlan::from_cli(stages, layers, output_dir.clone()).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr trace --save-tensor: bad plan args (stages={stages:?}, layers={layers:?}): {e:?}"
        ))
    })?;

    println!(
        "{}",
        "=== apr trace --save-tensor (GGUF MoE) ===".cyan().bold()
    );
    println!("Model:        {}", path.display());
    println!("Stages:       {stages}");
    println!("Layers:       {layers}");
    println!("Output dir:   {}", output_dir.display());
    println!();

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to create quantized model: {e}")))?;

    let canonical_arch =
        realizar::tensor_names::normalize_architecture(&model.config().architecture);
    let raw_arch_lower = model.config().architecture.to_lowercase();
    let is_qwen3_moe = canonical_arch == "qwen3_moe"
        || raw_arch_lower == "qwen3moe"
        || raw_arch_lower == "qwen3_moe";
    if !is_qwen3_moe {
        return Err(CliError::ModelLoadFailed(format!(
            "apr trace --save-tensor for GGUF: only qwen3_moe arch supported here \
             (got {}); use the dense path or .apr conversion",
            model.config().architecture
        )));
    }

    let test_prompt = "What is 2+2?";
    let test_tokens = mapped
        .model
        .encode(test_prompt)
        .unwrap_or_else(|| vec![1u32]);
    println!("Test prompt:  {test_prompt:?}");
    println!(
        "Token ids:    {test_tokens:?} ({} tokens)",
        test_tokens.len()
    );
    println!();

    let num_experts = mapped.model.expert_count().ok_or_else(|| {
        CliError::ValidationFailed(
            "qwen3_moe trace: missing 'expert_count' in GGUF metadata".to_string(),
        )
    })?;
    let num_experts_per_tok = mapped.model.expert_used_count().ok_or_else(|| {
        CliError::ValidationFailed(
            "qwen3_moe trace: missing 'expert_used_count' in GGUF metadata".to_string(),
        )
    })?;
    let moe_intermediate = mapped.model.expert_feed_forward_length().ok_or_else(|| {
        CliError::ValidationFailed(
            "qwen3_moe trace: missing 'expert_feed_forward_length' in GGUF metadata".to_string(),
        )
    })?;

    let data = mapped.data();
    let num_layers = model.config().num_layers;
    let mut moe_layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        moe_layers.push(
            realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .map_err(|e| {
                    CliError::ModelLoadFailed(format!(
                        "qwen3_moe trace: load layer {layer_idx}: {e}"
                    ))
                })?,
        );
    }

    let trace = model
        .forward_qwen3_moe_traced_with_plan(
            &test_tokens,
            &moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
            Some(&plan),
        )
        .map_err(|e| {
            CliError::InferenceFailed(format!("forward_qwen3_moe_traced_with_plan failed: {e}"))
        })?;

    let mut written: Vec<PathBuf> = Vec::new();
    collect_bin_files(&output_dir, &mut written).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot enumerate {}: {e}", output_dir.display()))
    })?;
    written.sort();

    println!(
        "{} {} stage tensor file(s):",
        "Wrote".green().bold(),
        written.len()
    );
    for p in &written {
        let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({} bytes)", p.display(), bytes);
    }
    println!();
    println!(
        "Forward pass succeeded — {} layer activations, {} logits",
        trace.layer_activations.len(),
        trace.logits.len()
    );
    println!(
        "MoE stages captured per `trace-moe-gpu-sub-stages-v1.yaml` v1.1.0 step (a) \
         (MoeRouter / MoeFfnOut per-layer when plan selects them)."
    );

    Ok(())
}

/// Recursively collect `*.bin` files under `dir`.
fn collect_bin_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            collect_bin_files(&p, out)?;
        } else if p.extension().and_then(|e| e.to_str()) == Some("bin") {
            out.push(p);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_dir_uses_model_stem() {
        let p = default_output_dir(Path::new("/tmp/foo/bar.apr"));
        assert_eq!(p, PathBuf::from("/tmp/foo/bar-trace"));
    }

    #[test]
    fn default_output_dir_handles_bare_filename() {
        // Bare filename: parent is "" → falls back to ".".
        let p = default_output_dir(Path::new("model.apr"));
        assert_eq!(p, PathBuf::from("./model-trace"));
    }

    #[test]
    fn default_output_dir_handles_no_extension() {
        let p = default_output_dir(Path::new("/tmp/no_ext"));
        assert_eq!(p, PathBuf::from("/tmp/no_ext-trace"));
    }

    #[test]
    fn collect_bin_files_recurses_per_layer_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        let layer0 = tmp.path().join("layer-0");
        std::fs::create_dir_all(&layer0).unwrap();
        std::fs::write(layer0.join("embedding.bin"), b"x").unwrap();
        std::fs::write(tmp.path().join("lm_head.bin"), b"y").unwrap();
        std::fs::write(tmp.path().join("ignore.txt"), b"z").unwrap(); // non-.bin
        let mut found = Vec::new();
        collect_bin_files(tmp.path(), &mut found).unwrap();
        found.sort();
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|p| p.ends_with("layer-0/embedding.bin")));
        assert!(found.iter().any(|p| p.ends_with("lm_head.bin")));
    }

    #[test]
    fn collect_bin_files_missing_dir_is_ok() {
        let mut found = Vec::new();
        collect_bin_files(Path::new("/nonexistent/path"), &mut found).unwrap();
        assert!(found.is_empty());
    }
}
