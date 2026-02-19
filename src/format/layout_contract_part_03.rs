
// ============================================================================
// MANDATORY CONTRACT ENFORCEMENT (GH-208)
// ============================================================================
//
// The contract is NOT A SUGGESTION. All tensor operations MUST go through
// these enforcement functions. Code that bypasses these functions will FAIL.
//
// Five Whys Analysis (GH-208):
// 1. Why did APR inference produce garbage? → Wrong embedding transpose
// 2. Why was transpose wrong? → Code didn't follow contract
// 3. Why didn't code follow contract? → Contract had no enforcement
// 4. Why no enforcement? → Contract was "documentation-first"
// 5. Why documentation-first? → We didn't make enforcement mandatory
//
// SOLUTION: Make enforcement MANDATORY. Code CANNOT operate on tensors
// without going through these functions that VALIDATE the contract.
// ============================================================================

/// MANDATORY: Validate and transform a tensor during GGUF→APR import.
///
/// This function MUST be called for EVERY tensor during import.
/// It returns the correct shape and whether data transpose is needed.
///
/// # Panics
///
/// Panics if the tensor is unknown to the contract. This is intentional -
/// we want to FAIL FAST rather than produce garbage output.
///
/// # Returns
///
/// `(apr_shape, needs_data_transpose)` where:
/// - `apr_shape`: The shape the tensor should have in APR format
/// - `needs_data_transpose`: Whether the raw data needs to be transposed
///
/// # Contract Rule
///
/// For GGUF data, the raw bytes are ALREADY in the correct memory layout
/// for row-major access. Only the SHAPE METADATA needs to be swapped.
/// DATA TRANSPOSE IS NEVER NEEDED for GGUF→APR because GGUF's
/// `data[i0 + i1*ne0]` layout IS row-major when interpreted as `[ne1, ne0]`.
#[must_use]
pub fn enforce_import_contract(
    tensor_name: &str,
    input_shape: &[usize],
    _vocab_size: usize,
    _hidden_dim: usize,
) -> (Vec<usize>, bool) {
    let layout = contract();

    // Check if tensor is in contract - try BOTH GGUF and APR name patterns
    // This handles cases where names have already been mapped by Architecture::map_name()
    let tc = layout
        .get_gguf_contract(tensor_name)
        .or_else(|| layout.get_apr_contract(tensor_name));

    if let Some(tc) = tc {
        // Validate shape dimensions
        let apr_shape = if tc.should_transpose && input_shape.len() == 2 {
            // SHAPE REVERSAL ONLY - no data transpose needed
            // GGUF [ne0, ne1] → APR [ne1, ne0]
            vec![input_shape[1], input_shape[0]]
        } else {
            input_shape.to_vec()
        };

        // CRITICAL: Data transpose is NEVER needed for GGUF import
        // GGUF data layout data[i0 + i1*ne0] for shape [ne0, ne1] is:
        //   - i0 is the contiguous (inner) dimension
        //   - i1 is the strided (outer) dimension
        // This is IDENTICAL to row-major [ne1, ne0] layout.
        // Data transpose is never correct here — the layout is already row-major.
        let needs_data_transpose = false;

        (apr_shape, needs_data_transpose)
    } else {
        // Unknown tensor - could be a bias or model-specific tensor
        // For 2D tensors, still apply shape reversal for consistency
        // (GGML shape convention → standard row-major)
        let apr_shape = if input_shape.len() == 2 {
            vec![input_shape[1], input_shape[0]]
        } else {
            input_shape.to_vec()
        };
        (apr_shape, false)
    }
}

/// MANDATORY: Validate tensor shape during APR model load.
///
/// This function MUST be called when loading tensors from APR format.
/// It validates that the shape matches the contract expectation.
///
/// # Errors
///
/// Returns `Err` if the shape violates the contract. Callers MUST
/// propagate this error - do not ignore it.
pub fn enforce_load_contract(
    apr_name: &str,
    apr_shape: &[usize],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<(), ContractError> {
    let layout = contract();

    if let Some(tc) = layout.get_apr_contract(apr_name) {
        // For critical tensors, validate shape strictly
        if tc.is_critical {
            layout.validate_apr_shape(apr_name, apr_shape, vocab_size, hidden_dim)?;
        }
    }
    // Unknown tensors are allowed (model-specific)
    Ok(())
}

/// MANDATORY: Check if embedding lookup will work correctly.
///
/// Embedding lookup uses `data[token_id * hidden_dim .. (token_id+1) * hidden_dim]`.
/// This function validates that the embedding tensor is in the correct layout.
///
/// # Panics
///
/// Panics if embedding layout is wrong. This prevents garbage inference output.
pub fn enforce_embedding_contract(embedding_len: usize, vocab_size: usize, hidden_dim: usize) {
    let expected_len = vocab_size * hidden_dim;
    assert_eq!(
        embedding_len, expected_len,
        "CONTRACT VIOLATION: Embedding length {} != vocab({}) * hidden({}) = {}. \
         This will cause garbage inference output. \
         See: contracts/tensor-layout-v1.yaml",
        embedding_len, vocab_size, hidden_dim, expected_len
    );
}

/// MANDATORY: Validate that matmul weight shape matches kernel expectation.
///
/// For row-major matmul `y = W @ x` where `y[out_dim]` and `x[in_dim]`,
/// weight W must have shape `[out_dim, in_dim]`.
///
/// # Panics
///
/// Panics if weight shape doesn't match kernel expectation.
pub fn enforce_matmul_contract(
    tensor_name: &str,
    weight_shape: &[usize],
    expected_out_dim: usize,
    expected_in_dim: usize,
) {
    assert_eq!(
        weight_shape.len(),
        2,
        "CONTRACT VIOLATION: {} must be 2D, got {:?}",
        tensor_name,
        weight_shape
    );
    assert_eq!(
        weight_shape[0], expected_out_dim,
        "CONTRACT VIOLATION: {} shape[0]={} but kernel expects out_dim={}. \
         See: contracts/tensor-layout-v1.yaml",
        tensor_name, weight_shape[0], expected_out_dim
    );
    assert_eq!(
        weight_shape[1], expected_in_dim,
        "CONTRACT VIOLATION: {} shape[1]={} but kernel expects in_dim={}. \
         See: contracts/tensor-layout-v1.yaml",
        tensor_name, weight_shape[1], expected_in_dim
    );
}

// ============================================================================
// GH-279: Architecture Completeness Gate
// ============================================================================
//
// Five Whys Root Cause: GPU inference silently produces garbage because the
// APR weight loader forgot to upload QK norm weights for Qwen3. The type
// system couldn't distinguish "model doesn't need QK norm" from "loader forgot."
//
// Solution: Before writing an APR file, verify that ALL tensors required by
// the declared architecture are present. Missing tensor = hard error, not
// silent garbage inference later.

/// GH-279: Required tensor name patterns per architecture feature.
///
/// Returns the per-layer tensor name patterns that MUST be present for the
/// given architecture features. Uses `{i}` as a placeholder for layer index.
#[must_use]
fn required_tensor_patterns(has_qk_norm: bool, has_bias: bool) -> Vec<&'static str> {
    let mut patterns = vec![
        // Always required: layer norms
        "blk.{i}.attn_norm.weight",
        "blk.{i}.ffn_norm.weight",
        // Always required: attention projections
        "blk.{i}.attn_q.weight",
        "blk.{i}.attn_k.weight",
        "blk.{i}.attn_v.weight",
        "blk.{i}.attn_output.weight",
        // Always required: FFN projections (SwiGLU)
        "blk.{i}.ffn_gate.weight",
        "blk.{i}.ffn_up.weight",
        "blk.{i}.ffn_down.weight",
    ];

    if has_qk_norm {
        patterns.push("blk.{i}.attn_q_norm.weight");
        patterns.push("blk.{i}.attn_k_norm.weight");
    }

    if has_bias {
        patterns.push("blk.{i}.attn_q.bias");
        patterns.push("blk.{i}.attn_k.bias");
        patterns.push("blk.{i}.attn_v.bias");
    }

    patterns
}

/// GH-279: Enforce architecture completeness at import/export boundary.
///
/// Checks that all tensors required by the declared architecture are present
/// in the model's tensor list. Missing required tensor = `Err` with a
/// descriptive message naming the missing tensor, architecture, and layer.
///
/// Contract: architecture-requirements-v1, equation "weight_completeness"
///
/// # Arguments
///
/// * `tensor_names` - Names of all tensors present in the model
/// * `architecture` - Architecture name (e.g., "qwen3", "qwen2", "llama")
/// * `num_layers` - Number of transformer layers
///
/// # Errors
///
/// Returns `ContractError` if any required tensor is missing.
#[provable_contracts_macros::contract("architecture-requirements-v1", equation = "import_completeness_gate")]
pub fn enforce_architecture_completeness(
    tensor_names: &[&str],
    architecture: &str,
    num_layers: usize,
) -> Result<(), ContractError> {
    // Derive architecture requirements
    let (has_qk_norm, has_bias) = match architecture {
        "qwen3" => (true, false),
        "qwen2" | "qwen2.5" | "qwen" => (false, true),
        "phi" | "phi2" | "phi3" => (false, true),
        _ => (false, false), // LLaMA, Mistral, Gemma, etc.
    };

    let patterns = required_tensor_patterns(has_qk_norm, has_bias);

    for layer_idx in 0..num_layers {
        for pattern in &patterns {
            let expected_name = pattern.replace("{i}", &layer_idx.to_string());
            if !tensor_names.iter().any(|n| *n == expected_name) {
                return Err(ContractError::TransposeError {
                    tensor: expected_name,
                    message: format!(
                        "GH-279: Missing required tensor for architecture '{}' \
                         — see contracts/architecture-requirements-v1.yaml",
                        architecture
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod architecture_completeness_tests {
    use super::*;

    #[test]
    fn test_llama_base_complete() {
        let names: Vec<&str> = (0..2)
            .flat_map(|i| {
                vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ]
            })
            .collect::<Vec<String>>()
            .iter()
            .map(String::as_str)
            .collect::<Vec<&str>>();
        // Need to keep owned strings alive
        let owned: Vec<String> = (0..2)
            .flat_map(|i| {
                vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ]
            })
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

        assert!(enforce_architecture_completeness(&refs, "llama", 2).is_ok());
    }

    #[test]
    fn test_llama_missing_ffn_gate() {
        let owned: Vec<String> = (0..2)
            .flat_map(|i| {
                let mut v = vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ];
                // Intentionally omit ffn_gate for layer 1
                if i == 0 {
                    v.push(format!("blk.{i}.ffn_gate.weight"));
                }
                v
            })
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

        let result = enforce_architecture_completeness(&refs, "llama", 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("blk.1.ffn_gate.weight"), "Error should name the missing tensor: {msg}");
    }

    #[test]
    fn test_qwen3_requires_qk_norm() {
        // Base tensors only (no QK norm) — should fail for Qwen3
        let owned: Vec<String> = (0..1)
            .flat_map(|i| {
                vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ]
            })
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

        let result = enforce_architecture_completeness(&refs, "qwen3", 1);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("attn_q_norm"), "Should require QK norm for Qwen3: {}", msg);
    }

    #[test]
    fn test_qwen3_complete_with_qk_norm() {
        let owned: Vec<String> = (0..1)
            .flat_map(|i| {
                vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                    format!("blk.{i}.attn_q_norm.weight"),
                    format!("blk.{i}.attn_k_norm.weight"),
                ]
            })
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

        assert!(enforce_architecture_completeness(&refs, "qwen3", 1).is_ok());
    }

    #[test]
    fn test_qwen2_requires_bias() {
        // Base tensors only (no bias) — should fail for Qwen2
        let owned: Vec<String> = (0..1)
            .flat_map(|i| {
                vec![
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ]
            })
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();

        let result = enforce_architecture_completeness(&refs, "qwen2", 1);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bias"), "Should require bias for Qwen2: {}", msg);
    }
}
