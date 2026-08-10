use crate::error::Result;
use std::path::{Path, PathBuf};

/// Explain command - provides documentation about errors, tensors, kernels, and models.
///
/// The positional argument is auto-detected: if it looks like a file path
/// (exists on disk or has a model file extension), it's treated as `--file`.
/// Otherwise it's treated as an error code or family name.
#[allow(clippy::unnecessary_wraps, clippy::fn_params_excessive_bools)]
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(
    code_or_file: Option<String>,
    file: Option<PathBuf>,
    tensor: Option<&str>,
    kernel: bool,
    json: bool,
    verbose: bool,
    proof_status: bool,
) -> Result<()> {
    // Auto-detect: is the positional argument a file path or an error code?
    let (code, resolved_file) = match code_or_file {
        Some(arg) => {
            let path = PathBuf::from(&arg);
            if is_model_path(&arg, &path) {
                (None, Some(path))
            } else {
                (Some(arg), file)
            }
        }
        None => (None, file),
    };

    if kernel {
        return explain_kernel(
            code.as_deref(),
            resolved_file.as_deref(),
            json,
            verbose,
            proof_status,
        );
    }

    if let Some(c) = code {
        explain_error_code(&c, json);
    } else if let Some(t) = tensor {
        explain_tensor(t, resolved_file.as_deref(), json);
    } else if let Some(ref f) = resolved_file {
        explain_file(f, json)?;
    } else {
        eprintln!("Please provide an error code, model file path, or --tensor/--kernel");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  apr explain E001              # explain error code");
        eprintln!("  apr explain model.gguf        # explain model architecture");
        eprintln!("  apr explain --tensor q_proj    # explain tensor role");
        eprintln!("  apr explain --kernel qwen2     # explain kernel dispatch");
        eprintln!("  apr explain --kernel model.apr # explain kernel from model file");
        return Err(crate::error::CliError::ValidationFailed(
            "No argument provided — see usage above".into(),
        ));
    }
    Ok(())
}

/// Emit a kernel error message, respecting --json mode.
fn emit_kernel_error(json_mode: bool, msg: &str) {
    if json_mode {
        let err = serde_json::json!({ "error": msg });
        println!("{}", serde_json::to_string_pretty(&err).unwrap_or_default());
    } else {
        eprintln!("Error: {msg}");
    }
}

/// Kernel explainability: resolve family and display kernel dispatch pipeline.
fn explain_kernel(
    code_or_family: Option<&str>,
    file: Option<&Path>,
    json: bool,
    verbose: bool,
    proof_status: bool,
) -> Result<()> {
    use super::kernel_explain::*;

    // Resolution chain: file → code_or_family string → error
    let family =
        resolve_family_from_file(file, json).or_else(|| resolve_family_from_string(code_or_family));

    let Some(family) = family else {
        emit_unresolved_kernel_error(code_or_family, json);
        std::process::exit(1);
    };

    let config_mapping = resolve_config_mapping(file, code_or_family);

    if json {
        let output = build_json_output(&family, config_mapping, proof_status);
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else {
        print_human_output(&family, &config_mapping, verbose, proof_status);
    }

    Ok(())
}

/// Resolve kernel family from a file path (config.json or model file).
fn resolve_family_from_file(
    file: Option<&Path>,
    json: bool,
) -> Option<super::kernel_explain::FamilyInfo> {
    use super::kernel_explain::*;

    let path = file?;

    // Validate file exists and is readable
    if !path.exists() {
        eprintln!("Error: File not found: {}", path.display());
        std::process::exit(1);
    }
    if path.is_dir() {
        eprintln!(
            "Error: '{}' is a directory, not a file. Provide a config.json or model file.",
            path.display()
        );
        std::process::exit(1);
    }

    if path.extension().map_or(false, |e| e == "json") {
        resolve_family_from_json_file(path, json)
    } else {
        resolve_family_from_model_file(path, json)
    }
}

/// Resolve family from a direct config.json file, with detailed diagnostics on failure.
fn resolve_family_from_json_file(
    path: &Path,
    json: bool,
) -> Option<super::kernel_explain::FamilyInfo> {
    use super::kernel_explain::*;

    let result = resolve_from_config_json(path);
    if result.is_none() {
        diagnose_json_resolution_failure(path, json);
    }
    result
}

/// Diagnose why a config.json file failed to resolve a family, emit error, and exit.
fn diagnose_json_resolution_failure(path: &Path, json: bool) {
    use super::kernel_explain::extract_json_string;

    match std::fs::read_to_string(path) {
        Ok(content) => {
            let trimmed = content.trim();
            if trimmed.is_empty() {
                emit_kernel_error(json, &format!("'{}' is empty", path.display()));
            } else if trimmed.starts_with('[') {
                emit_kernel_error(
                    json,
                    &format!(
                    "'{}' is a JSON array, not a JSON object. config.json must be a JSON object.",
                    path.display()
                ),
                );
            } else if !content.contains('{') {
                emit_kernel_error(json, &format!("'{}' is not valid JSON", path.display()));
            } else {
                diagnose_missing_model_type(path, &content, json);
            }
            std::process::exit(1);
        }
        Err(e) => {
            emit_kernel_error(json, &format!("Could not read '{}': {e}", path.display()));
            std::process::exit(1);
        }
    }
}

/// Diagnose a config.json that parses as JSON but has no resolvable model_type.
fn diagnose_missing_model_type(path: &Path, content: &str, json: bool) {
    use super::kernel_explain::extract_json_string;

    let model_type = extract_json_string(content, "model_type");
    if let Some(mt) = model_type {
        emit_kernel_error(
            json,
            &format!(
                "Unknown model_type '{}' in '{}'. \
                 Run `apr explain --kernel` for supported families.",
                mt,
                path.display()
            ),
        );
    } else {
        let has_arch = content.contains("\"architectures\"");
        let msg = if has_arch {
            format!(
                "No \"model_type\" field in '{}'. \
                 Found \"architectures\" but could not resolve family from it.",
                path.display()
            )
        } else {
            format!(
                "No \"model_type\" field in '{}'. \
                 Ensure the file is a HuggingFace config.json.",
                path.display()
            )
        };
        emit_kernel_error(json, &msg);
    }
}

/// Resolve family from a model file by finding config.json in the same directory.
fn resolve_family_from_model_file(
    path: &Path,
    json: bool,
) -> Option<super::kernel_explain::FamilyInfo> {
    use super::kernel_explain::*;

    let real_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    // Companion lookup: find_sibling_file handles hash-prefixed / symlinked
    // model paths robustly (CB-510 / publish-safety) and only returns Some
    // when the file exists — subsuming the prior with_file_name + .exists().
    if let Some(config_path) = realizar::safetensors::find_sibling_file(&real_path, "config.json") {
        resolve_from_config_json(&config_path)
    } else {
        emit_kernel_error(
            json,
            &format!(
                "No config.json found alongside '{}'. \
                 Kernel analysis requires a HuggingFace config.json in the same directory.",
                path.display()
            ),
        );
        std::process::exit(1);
    }
}

/// Resolve kernel family from a string argument (family name, HF repo, or path).
fn resolve_family_from_string(
    code_or_family: Option<&str>,
) -> Option<super::kernel_explain::FamilyInfo> {
    use super::kernel_explain::*;

    code_or_family.and_then(|input| {
        let input = input.strip_prefix("hf://").unwrap_or(input).trim();
        // Try as config.json path
        let path = Path::new(input);
        if path.exists() && path.extension().map_or(false, |e| e == "json") {
            return resolve_from_config_json(path);
        }
        // Try as HF repo ID -> look up cached config.json
        if input.contains('/') {
            if input.contains("..") {
                return None; // Reject path traversal
            }
            let cache_base = dirs::home_dir()
                .unwrap_or_default()
                .join(".apr/cache/hf")
                .join(input)
                .join("config.json");
            if cache_base.exists() {
                return resolve_from_config_json(&cache_base);
            }
        }
        resolve_family(input)
    })
}

/// Emit error when kernel family could not be resolved from any source.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn emit_unresolved_kernel_error(code_or_family: Option<&str>, json: bool) {
    use super::kernel_explain::*;

    let raw_input = code_or_family
        .map(|s| s.strip_prefix("hf://").unwrap_or(s).trim())
        .unwrap_or("(none)");
    let input = if raw_input.len() > 80 {
        &raw_input[..raw_input.floor_char_boundary(80)]
    } else {
        raw_input
    };
    let suffix = if raw_input.len() > 80 { "..." } else { "" };

    if json {
        let err = serde_json::json!({
            "error": format!("Could not resolve kernel class for '{input}{suffix}'"),
            "available_families": load_families().iter().map(|f| &f.family).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&err).unwrap_or_default());
    } else {
        eprintln!("Error: Could not resolve kernel class for '{input}{suffix}'");
        eprintln!();
        eprintln!("Available families:");
        let families = load_families();
        for f in &families {
            eprintln!(
                "  {:<12} {} (Class {})",
                f.family,
                f.display_name,
                f.kernel_class.letter()
            );
        }
        let aliases = family_aliases();
        if !aliases.is_empty() {
            eprintln!();
            eprintln!("Also accepted (aliases):");
            let mut shown: Vec<String> = Vec::new();
            for (alias, _target) in aliases {
                if !shown.contains(&alias.to_string()) {
                    shown.push(alias.to_string());
                }
            }
            for chunk in shown.chunks(8) {
                eprintln!("  {}", chunk.join(", "));
            }
        }
    }
}

/// Extract config.json mapping from a file path or HF cache.
fn resolve_config_mapping(
    file: Option<&Path>,
    code_or_family: Option<&str>,
) -> std::collections::BTreeMap<String, super::kernel_explain::ConfigField> {
    use super::kernel_explain::*;

    file.map(|p| {
        let config_path = if p.extension().map_or(false, |e| e == "json") {
            p.to_path_buf()
        } else {
            // Companion lookup: prefer the robust sibling finder (handles
            // hash-prefixed / symlinked model paths); fall back to the literal
            // sibling dir so extract_config_mapping reports not-found as before
            // (parent().join avoids with_file_name per publish-safety).
            realizar::safetensors::find_sibling_file(p, "config.json").unwrap_or_else(|| {
                p.parent().map_or_else(
                    || std::path::PathBuf::from("config.json"),
                    |d| d.join("config.json"),
                )
            })
        };
        extract_config_mapping(&config_path)
    })
    .or_else(|| {
        code_or_family.and_then(|input| {
            let input = input.strip_prefix("hf://").unwrap_or(input);
            if input.contains('/') {
                let cache_path = dirs::home_dir()?
                    .join(".apr/cache/hf")
                    .join(input)
                    .join("config.json");
                if cache_path.exists() {
                    return Some(extract_config_mapping(&cache_path));
                }
            }
            let path = Path::new(input);
            if path.exists() && path.extension().map_or(false, |e| e == "json") {
                return Some(extract_config_mapping(path));
            }
            None
        })
    })
    .unwrap_or_default()
}

/// Check if an argument looks like a model file path (exists or has model extension)
fn is_model_path(arg: &str, path: &Path) -> bool {
    // Check if file exists on disk
    if path.exists() {
        return true;
    }
    // Check for known model file extensions
    let extensions = [".apr", ".gguf", ".safetensors", ".bin", ".pt", ".pth"];
    extensions.iter().any(|ext| arg.ends_with(ext))
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn explain_error_code(code: &str, json: bool) {
    // GH-510: Respect --json flag for error code explanations
    if json {
        let (title, description, troubleshooting) = match code {
            "E001" => (
                "Invalid Magic Bytes",
                "The file does not start with a recognized format header.",
                vec![
                    "Run `apr validate <file>` to check format",
                    "Verify file was not corrupted during download",
                ],
            ),
            "E002" => (
                "Corrupted Data",
                "The payload checksum does not match the header.",
                vec![
                    "Run `apr validate --checksum` to verify",
                    "Check source file integrity (MD5/SHA256)",
                ],
            ),
            _ => {
                let err = serde_json::json!({
                    "error": format!("Error code '{}' not recognized", code),
                    "available_codes": ["E001", "E002", "E003", "E004", "E005", "E006"],
                });
                println!("{}", serde_json::to_string_pretty(&err).unwrap_or_default());
                return;
            }
        };
        let output = serde_json::json!({
            "code": code,
            "title": title,
            "description": description,
            "troubleshooting": troubleshooting,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return;
    }

    println!("Explain error code: {code}");
    match code {
        "E001" => {
            println!("**E001: Invalid Magic Bytes**");
            println!("The file does not start with a recognized format header.");
            println!(
                "- **Expected**: GGUF (`GGUF`), SafeTensors (u64 LE + `{{\"`), APR (`APR\\0`)"
            );
            println!("- **Troubleshooting**:");
            println!("  1. Run `apr validate <file>` to check format.");
            println!("  2. Verify file was not corrupted during download.");
        }
        "E002" => {
            println!("**E002: Corrupted Data**");
            println!("The payload checksum does not match the header.");
            println!("- **Common Causes**: Interrupted download, bit rot, disk error.");
            println!("- **Troubleshooting**:");
            println!("  1. Run `apr validate --checksum` to verify.");
            println!("  2. Check source file integrity (MD5/SHA256).");
        }
        _ => {
            // PMAT-191: Structured error response (no "Unknown Error")
            println!("Error code '{code}' not recognized.");
            println!();
            println!("Available error codes:");
            println!("  E001  Invalid magic bytes (not an APR file)");
            println!("  E002  Corrupted data (checksum mismatch)");
            println!("  E003  Unsupported format version");
            println!("  E004  Missing required tensor");
            println!("  E005  Dimension mismatch");
            println!("  E006  Quantization error");
            println!();
            println!("Run `apr validate <file>` for detailed diagnostics.");
        }
    }
}

/// PMAT-266: Explain tensor — look up in actual model file via RosettaStone
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn explain_tensor(tensor_name: &str, file: Option<&Path>, json: bool) {
    // GH-510: Respect --json flag for tensor explanations
    if json {
        let role = TENSOR_ROLES
            .iter()
            .find(|(patterns, _)| patterns.iter().any(|p| tensor_name.contains(p)))
            .map(|(_, desc)| *desc);

        let mut output = serde_json::json!({
            "tensor": tensor_name,
            "role": role.unwrap_or("unknown"),
        });

        // If a file is provided, try to look up the actual tensor
        if let Some(path) = file {
            if path.exists() {
                let rosetta = aprender::format::rosetta::RosettaStone::new();
                if let Ok(report) = rosetta.inspect(path) {
                    let matching: Vec<_> = report
                        .tensors
                        .iter()
                        .filter(|t| t.name == tensor_name || t.name.contains(tensor_name))
                        .map(|t| {
                            serde_json::json!({
                                "name": t.name,
                                "shape": t.shape,
                                "dtype": format!("{:?}", t.dtype),
                            })
                        })
                        .collect();
                    if !matching.is_empty() {
                        output["matches"] = serde_json::json!(matching);
                    }
                    output["file"] = serde_json::json!(path.display().to_string());
                }
            }
        }

        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return;
    }

    println!("Explain tensor: {tensor_name}");

    // If a file is provided, try to look up the actual tensor
    if let Some(path) = file {
        if explain_tensor_from_file(tensor_name, path) {
            return;
        }
    }

    // Fallback: explain by naming convention
    explain_tensor_role(tensor_name);
}

/// Attempt to explain a tensor by inspecting a model file.
/// Returns `true` if the file was successfully inspected (tensor found or not),
/// `false` if the file could not be read.
fn explain_tensor_from_file(tensor_name: &str, path: &Path) -> bool {
    if !path.exists() {
        return false;
    }

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let Ok(report) = rosetta.inspect(path) else {
        return false;
    };

    // Find matching tensor (exact or fuzzy)
    let matching: Vec<_> = report
        .tensors
        .iter()
        .filter(|t| t.name == tensor_name || t.name.contains(tensor_name))
        .collect();

    if matching.is_empty() {
        println!("Tensor '{tensor_name}' not found in {}", path.display());
        print_tensor_suggestions(tensor_name, &report.tensors);
    } else {
        for t in &matching {
            println!("\n**{}**", t.name);
            println!("- **Shape**: {:?}", t.shape);
            println!("- **DType**: {:?}", t.dtype);
            explain_tensor_role(&t.name);
        }
    }

    true
}

/// Print similar tensor name suggestions when an exact/fuzzy match fails
fn print_tensor_suggestions(tensor_name: &str, tensors: &[aprender::format::rosetta::TensorInfo]) {
    let suggestions: Vec<_> = tensors
        .iter()
        .filter(|t| {
            let parts: Vec<&str> = tensor_name.split('.').collect();
            parts.iter().any(|p| t.name.contains(p))
        })
        .take(5)
        .collect();

    if !suggestions.is_empty() {
        println!("\nDid you mean:");
        for s in &suggestions {
            println!("  - {} ({:?}, {:?})", s.name, s.shape, s.dtype);
        }
    }
}

/// Tensor naming convention table: (pattern, role description)
const TENSOR_ROLES: &[(&[&str], &str)] = &[
    (
        &["embed", "token_embd"],
        "Token embedding — maps token IDs to dense vectors",
    ),
    (
        &["lm_head", "output.weight"],
        "Language model head — projects hidden states to vocabulary logits",
    ),
    (&["q_proj"], "Query projection in attention mechanism"),
    (&["k_proj"], "Key projection in attention mechanism"),
    (&["v_proj"], "Value projection in attention mechanism"),
    (
        &["o_proj", "out_proj"],
        "Output projection in attention mechanism",
    ),
    (
        &["gate_proj", "fc1"],
        "Feed-forward gate/first projection (SwiGLU or FFN)",
    ),
    (&["up_proj"], "Feed-forward up projection (SwiGLU)"),
    (&["down_proj", "fc2"], "Feed-forward down projection"),
    (
        &["layernorm", "input_layernorm"],
        "Layer normalization — stabilizes activations",
    ),
    (
        &["rms_norm", "post_attention_layernorm"],
        "RMS normalization — pre/post attention normalization",
    ),
    (&["conv1"], "First convolutional layer (feature extraction)"),
    (
        &["conv2"],
        "Second convolutional layer (stride-2 downsampling)",
    ),
    (
        &["positional", "pos_embed"],
        "Positional encoding — provides sequence position information",
    ),
    (
        &["encoder_attn", "cross_attn"],
        "Cross-attention — attends to encoder output from decoder",
    ),
    (
        &["self_attn"],
        "Self-attention — attends within the same sequence",
    ),
    // GH-635: GGUF tensor naming conventions (blk.N.attn_*, blk.N.ffn_*)
    (&["attn_q"], "Query projection (GGUF convention)"),
    (&["attn_k"], "Key projection (GGUF convention)"),
    (&["attn_v"], "Value projection (GGUF convention)"),
    (
        &["attn_output"],
        "Attention output projection (GGUF convention)",
    ),
    (&["ffn_up"], "Feed-forward up projection (GGUF convention)"),
    (
        &["ffn_down"],
        "Feed-forward down projection (GGUF convention)",
    ),
    (&["ffn_gate"], "Feed-forward gate projection (GGUF SwiGLU)"),
    (&["attn_norm"], "Attention normalization (GGUF convention)"),
    (
        &["ffn_norm"],
        "Feed-forward normalization (GGUF convention)",
    ),
    (
        &["output_norm"],
        "Final output normalization (GGUF convention)",
    ),
    (
        &["rope_freqs"],
        "Rotary positional encoding frequencies (RoPE)",
    ),
];

/// Explain a tensor's role based on naming conventions
fn explain_tensor_role(name: &str) {
    let role = TENSOR_ROLES
        .iter()
        .find(|(patterns, _)| patterns.iter().any(|p| name.contains(p)))
        .map(|(_, desc)| *desc);

    match role {
        Some(desc) => println!("- **Role**: {desc}"),
        None => println!("- **Role**: (unknown convention — use `apr tensors <file>` for details)"),
    }
}

/// Layer prefix patterns for counting transformer layers
const LAYER_PREFIXES: &[&str] = &[
    "model.layers.",
    "model.encoder.layers.",
    "model.decoder.layers.",
    "encoder.layers.",
    "decoder.layers.",
    "blk.",
];

/// Count transformer layers from tensor names
fn count_layers(tensor_names: &[String]) -> usize {
    tensor_names
        .iter()
        .filter_map(|n| {
            LAYER_PREFIXES
                .iter()
                .find_map(|prefix| n.strip_prefix(prefix))
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |n| n + 1)
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn explain_file(path: &Path, json: bool) -> Result<()> {
    if !path.exists() {
        print_not_found(path, json);
        return Err(crate::error::CliError::FileNotFound(path.to_path_buf()));
    }
    let report = inspect_or_report(path, json)?;
    let tensor_names: Vec<String> = report.tensors.iter().map(|t| t.name.clone()).collect();
    let (arch, examples) = detect_architecture(&tensor_names);
    let n_layers = count_layers(&tensor_names);
    if json {
        print_json_summary(path, &report, arch, examples, n_layers);
    } else {
        print_text_summary(path, &report, arch, examples, n_layers);
    }
    Ok(())
}

/// Report a missing model file.
///
/// This used to `println!` on STDOUT, where it was indistinguishable from a
/// successful explanation to anything reading the command's output — and the
/// caller then returned `Ok(())`, so the process exited 0 as well.
///
/// In text mode this now emits nothing: the `CliError::FileNotFound` the caller
/// returns already renders as `error: File not found: <path>` on stderr, and
/// printing it here too would duplicate it. In JSON mode the error object stays
/// on stdout, because a `--json` consumer parses stdout and a machine-readable
/// error object is a legitimate result there.
#[allow(clippy::disallowed_methods)]
fn print_not_found(path: &Path, json: bool) {
    if json {
        let err = serde_json::json!({ "error": format!("File not found: {}", path.display()) });
        println!("{}", serde_json::to_string_pretty(&err).unwrap_or_default());
    }
}

#[allow(clippy::disallowed_methods)]
fn inspect_or_report(
    path: &Path,
    json: bool,
) -> Result<aprender::format::rosetta::InspectionReport> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    match rosetta.inspect(path) {
        Ok(r) => Ok(r),
        Err(e) => {
            if json {
                let err = serde_json::json!({ "error": format!("Failed to inspect model: {e}") });
                println!("{}", serde_json::to_string_pretty(&err).unwrap_or_default());
            } else {
                // The message itself is carried by the returned CliError; only
                // the follow-up hint is additive.
                eprintln!(
                    "Run `apr validate {0}` for format diagnostics.",
                    path.display()
                );
            }
            Err(crate::error::CliError::ValidationFailed(format!(
                "failed to inspect {}: {e}",
                path.display()
            )))
        }
    }
}

fn detect_architecture(tensor_names: &[String]) -> (&'static str, &'static str) {
    let has_encoder = tensor_names
        .iter()
        .any(|n| n.starts_with("encoder") || n.starts_with("model.encoder"));
    let has_decoder = tensor_names
        .iter()
        .any(|n| n.starts_with("decoder") || n.starts_with("model.decoder"));
    let has_model_layers = tensor_names
        .iter()
        .any(|n| n.starts_with("model.layers.") || n.starts_with("blk."));
    let has_transformer_h = tensor_names.iter().any(|n| n.starts_with("transformer.h."));

    if has_encoder && has_decoder {
        ("Encoder-Decoder Transformer", "Whisper, T5, BART")
    } else if has_encoder {
        ("Encoder-Only Transformer", "BERT, RoBERTa")
    } else if has_decoder || has_model_layers {
        ("Decoder-Only Transformer", "LLaMA, Qwen2, GPT")
    } else if has_transformer_h {
        ("Decoder-Only Transformer", "GPT-2")
    } else {
        ("Unknown", "")
    }
}

#[allow(clippy::disallowed_methods)]
fn print_json_summary(
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
    arch: &str,
    examples: &str,
    n_layers: usize,
) {
    let mut output = serde_json::json!({
        "file": path.display().to_string(),
        "format": format!("{}", report.format),
        "tensor_count": report.tensors.len(),
        "architecture": arch,
    });
    if !examples.is_empty() {
        output["examples"] = serde_json::json!(examples);
    }
    if n_layers > 0 {
        output["layers"] = serde_json::json!(n_layers);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

fn print_text_summary(
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
    arch: &str,
    examples: &str,
    n_layers: usize,
) {
    println!("Explain model architecture: {}", path.display());
    println!("- **Format**: {}", report.format);
    println!("- **Tensors**: {}", report.tensors.len());
    println!("- **Architecture**: {arch}");
    if !examples.is_empty() {
        println!("- **Examples**: {examples}");
    }
    if n_layers > 0 {
        println!("- **Layers**: {n_layers}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: call run() with default flags (no kernel/json/verbose/proof)
    fn run_default(
        code_or_file: Option<String>,
        file: Option<PathBuf>,
        tensor: Option<&str>,
    ) -> Result<()> {
        run(code_or_file, file, tensor, false, false, false, false)
    }

    // ========================================================================
    // Error Code Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_known_error_code_e002() {
        let result = run_default(Some("E002".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_unknown_error_code() {
        let result = run_default(Some("E999".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_error_code_e001() {
        let result = run_default(Some("E001".to_string()), None, None);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Tensor Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_known_tensor() {
        let result = run_default(None, None, Some("encoder.conv1.weight"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_unknown_tensor() {
        let result = run_default(None, None, Some("unknown.tensor"));
        assert!(result.is_ok());
    }

    // ========================================================================
    // File/Model Explanation Tests
    // ========================================================================

    // These two used to assert `is_ok()` on paths that do not exist, which is
    // how the exit-0-on-missing-file defect stayed green: the tests asserted
    // the shape of the call, not the behaviour, and so encoded the bug.
    #[test]
    fn test_explain_file() {
        let result = run_default(None, Some(PathBuf::from("/path/to/model.apr")), None);
        assert!(
            result.is_err(),
            "a nonexistent .apr must be an error, not a silent success"
        );
    }

    #[test]
    fn test_explain_file_with_gguf_extension() {
        let result = run_default(None, Some(PathBuf::from("model.gguf")), None);
        assert!(
            result.is_err(),
            "a nonexistent .gguf must be an error, not a silent success"
        );
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_explain_no_arguments() {
        let result = run_default(None, None, None);
        assert!(result.is_err(), "explain with no args should return error");
    }

    #[test]
    fn test_explain_empty_code() {
        let result = run_default(Some(String::new()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_empty_tensor() {
        let result = run_default(None, None, Some(""));
        assert!(result.is_ok());
    }

    // ========================================================================
    // Kernel Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_kernel_qwen2() {
        // Should resolve and print kernel info (not fail)
        let result = run(
            Some("qwen2".to_string()),
            None,
            None,
            true,
            false,
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_kernel_json() {
        let result = run(
            Some("llama".to_string()),
            None,
            None,
            true,
            true,
            false,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_kernel_verbose_proof() {
        let result = run(
            Some("gpt2".to_string()),
            None,
            None,
            true,
            false,
            true,
            true,
        );
        assert!(result.is_ok());
    }

    // ========================================================================
    // Exit-code discipline (dogfood 0.63.0)
    // ========================================================================
    //
    // `apr explain /nonexistent/model.apr` printed "File not found: ..." on
    // STDOUT and exited 0. Every CI wrapper and shell script that branches on
    // `$?` read that as a successful explanation. `explain_file` returned `()`,
    // so there was no way for the failure to reach the process exit code.

    #[test]
    fn missing_model_file_is_an_error_not_a_success() {
        let missing = PathBuf::from("/nonexistent/dogfood-explain-missing.apr");
        assert!(!missing.exists(), "fixture path must not exist");

        let result = run_default(None, Some(missing.clone()), None);

        assert!(
            result.is_err(),
            "apr explain on a missing file must fail; returning Ok makes the \
             process exit 0 and a missing model reads as success"
        );
        assert!(
            matches!(result, Err(crate::error::CliError::FileNotFound(_))),
            "expected FileNotFound (exit code 3), got {result:?}"
        );
    }

    #[test]
    fn missing_model_file_is_an_error_in_json_mode_too() {
        let missing = PathBuf::from("/nonexistent/dogfood-explain-missing.apr");
        let result = run(None, Some(missing), None, false, true, false, false);
        assert!(
            result.is_err(),
            "--json must not turn a missing file into exit 0"
        );
    }

    #[test]
    fn unreadable_model_file_is_an_error_not_a_success() {
        // A file that exists but is not a model: inspect() fails, and that
        // used to be swallowed the same way.
        let dir = std::env::temp_dir().join("apr-dogfood-explain-bogus");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let bogus = dir.join("not-a-model.apr");
        std::fs::write(&bogus, b"this is not a model file").expect("write fixture");

        let result = run_default(None, Some(bogus.clone()), None);
        let _ = std::fs::remove_file(&bogus);

        assert!(
            result.is_err(),
            "apr explain on an unreadable model must fail, not exit 0"
        );
    }
}
