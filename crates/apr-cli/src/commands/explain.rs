use crate::error::Result;
use std::path::PathBuf;

/// Explain command - provides documentation about errors, tensors, and models.
///
/// Returns Result for consistency with other CLI commands and to enable
/// future error handling (e.g., file read failures, network errors).
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn run(
    code: Option<String>,
    file: Option<PathBuf>,
    tensor: Option<&str>,
) -> Result<()> {
    if let Some(c) = code {
        explain_error_code(&c);
    } else if let Some(t) = tensor {
        explain_tensor(t, file.as_ref());
    } else if let Some(f) = file {
        explain_file(&f);
    } else {
        println!("Please provide --code, --tensor, or --file");
    }
    Ok(())
}

fn explain_error_code(code: &str) {
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
fn explain_tensor(tensor_name: &str, file: Option<&PathBuf>) {
    println!("Explain tensor: {tensor_name}");

    // If a file is provided, look up the actual tensor
    if let Some(path) = file {
        if path.exists() {
            let rosetta = aprender::format::rosetta::RosettaStone::new();
            if let Ok(report) = rosetta.inspect(path) {
                // Find matching tensor (exact or fuzzy)
                let matching: Vec<_> = report
                    .tensors
                    .iter()
                    .filter(|t| t.name == tensor_name || t.name.contains(tensor_name))
                    .collect();

                if matching.is_empty() {
                    println!("Tensor '{tensor_name}' not found in {}", path.display());
                    // Suggest similar tensors
                    let suggestions: Vec<_> = report
                        .tensors
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
                } else {
                    for t in &matching {
                        println!("\n**{}**", t.name);
                        println!("- **Shape**: {:?}", t.shape);
                        println!("- **DType**: {:?}", t.dtype);
                        explain_tensor_role(&t.name);
                    }
                }
                return;
            }
        }
    }

    // Fallback: explain by naming convention
    explain_tensor_role(tensor_name);
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

fn explain_file(path: &PathBuf) {
    println!("Explain model architecture: {}", path.display());

    if !path.exists() {
        println!("File not found: {}", path.display());
        return;
    }

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = match rosetta.inspect(path) {
        Ok(r) => r,
        Err(e) => {
            println!("Failed to inspect model: {e}");
            println!(
                "Run `apr validate {0}` for format diagnostics.",
                path.display()
            );
            return;
        }
    };

    println!("- **Format**: {:?}", report.format);
    println!("- **Tensors**: {}", report.tensors.len());

    let tensor_names: Vec<String> = report.tensors.iter().map(|t| t.name.clone()).collect();
    let has_encoder = tensor_names
        .iter()
        .any(|n| n.starts_with("encoder") || n.starts_with("model.encoder"));
    let has_decoder = tensor_names
        .iter()
        .any(|n| n.starts_with("decoder") || n.starts_with("model.decoder"));
    let has_model_layers = tensor_names.iter().any(|n| n.starts_with("model.layers."));

    let (arch, examples) = if has_encoder && has_decoder {
        ("Encoder-Decoder Transformer", "Whisper, T5, BART")
    } else if has_encoder {
        ("Encoder-Only Transformer", "BERT, RoBERTa")
    } else if has_decoder || has_model_layers {
        ("Decoder-Only Transformer", "LLaMA, Qwen2, GPT")
    } else {
        ("Unknown", "")
    };

    println!("- **Architecture**: {arch}");
    if !examples.is_empty() {
        println!("- **Examples**: {examples}");
    }

    let n_layers = count_layers(&tensor_names);
    if n_layers > 0 {
        println!("- **Layers**: {n_layers}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Error Code Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_known_error_code_e002() {
        // E002 is explicitly handled
        let result = run(Some("E002".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_unknown_error_code() {
        // Unknown error codes should list available codes
        let result = run(Some("E999".to_string()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_error_code_e001() {
        let result = run(Some("E001".to_string()), None, None);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Tensor Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_known_tensor() {
        let result = run(None, None, Some("encoder.conv1.weight"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_unknown_tensor() {
        let result = run(None, None, Some("unknown.tensor"));
        assert!(result.is_ok());
    }

    // ========================================================================
    // File/Model Explanation Tests
    // ========================================================================

    #[test]
    fn test_explain_file() {
        let result = run(None, Some(PathBuf::from("/path/to/model.apr")), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_file_with_gguf_extension() {
        let result = run(None, Some(PathBuf::from("model.gguf")), None);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_explain_no_arguments() {
        // Should print help message
        let result = run(None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_empty_code() {
        let result = run(Some(String::new()), None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_empty_tensor() {
        let result = run(None, None, Some(""));
        assert!(result.is_ok());
    }
}
