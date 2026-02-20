
/// Infer architecture from user option or model config string.
fn infer_architecture(user_arch: &Architecture, config_arch: Option<&str>) -> Architecture {
    if *user_arch != Architecture::Auto {
        return user_arch.clone();
    }
    config_arch
        .and_then(Architecture::from_model_type)
        .unwrap_or(Architecture::Auto)
}

/// Emit warnings for unverified architectures; error in strict mode.
fn warn_unverified_architecture(arch: &Architecture, strict: bool) -> Result<()> {
    if arch.is_inference_verified() {
        return Ok(());
    }
    eprintln!(
        "[PMAT-224] WARNING: Architecture '{}' has not been verified for inference.",
        arch.display_name()
    );
    eprintln!(
        "[PMAT-224] Verified architectures: Qwen2, Qwen3, Qwen3.5, LLaMA, Phi. Use --strict to reject unverified."
    );
    if strict {
        return Err(AprenderError::FormatError {
            message: format!(
                "Architecture '{}' is not verified for inference (--strict mode). \
                 Remove --strict to import anyway, or specify --arch qwen2/llama.",
                arch.display_name()
            ),
        });
    }
    Ok(())
}

/// Validate F32 tensors against layout contract.
fn validate_contract_f32(
    layout: &crate::format::layout_contract::LayoutContract,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    vocab_size: usize,
    hidden_dim: usize,
    strict: bool,
) -> Result<()> {
    if vocab_size == 0 || hidden_dim == 0 {
        eprintln!(
            "[CONTRACT] WARNING: Cannot validate contract - missing vocab_size or hidden_dim"
        );
        return Ok(());
    }
    for (name, (_data, shape)) in tensors {
        if let Err(e) = layout.validate_apr_shape(name, shape, vocab_size, hidden_dim) {
            eprintln!(
                "[CONTRACT-VIOLATION] {}: {} (see contracts/tensor-layout-v1.yaml)",
                name, e
            );
            if strict {
                return Err(AprenderError::FormatError {
                    message: format!("Contract violation: {e}"),
                });
            }
        }
    }
    eprintln!(
        "[CONTRACT] Validated {} tensors against tensor-layout-v1.yaml (vocab={}, hidden={})",
        tensors.len(),
        vocab_size,
        hidden_dim
    );
    Ok(())
}

include!("source_load_result.rs");
include!("import_part_03.rs");
include!("safe_tensors_load_result.rs");
include!("tensor_accumulator.rs");
include!("import_part_06.rs");
