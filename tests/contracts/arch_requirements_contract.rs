// CONTRACT: architecture-requirements-v1.yaml
// HASH: sha256:gh279-arch-enforcement
//
// FALSIFY-ARCH-001: Every architecture has correct required roles
// FALSIFY-ARCH-002: Missing required weight produces error (not silent skip)
// FALSIFY-ARCH-003: Optional weights are allowed to be (0, 0)
// FALSIFY-ARCH-004: Import completeness gate catches missing tensors

use aprender::format::layout_contract::enforce_architecture_completeness;

// ============================================================================
// FALSIFY-ARCH-001: Required roles are correct per architecture
// ============================================================================

/// Generate a complete set of per-layer tensor names for the base architecture.
fn base_tensor_names(num_layers: usize) -> Vec<String> {
    (0..num_layers)
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
        .collect()
}

/// Add QK norm tensor names.
fn with_qk_norm(mut names: Vec<String>, num_layers: usize) -> Vec<String> {
    for i in 0..num_layers {
        names.push(format!("blk.{i}.attn_q_norm.weight"));
        names.push(format!("blk.{i}.attn_k_norm.weight"));
    }
    names
}

/// Add bias tensor names.
fn with_bias(mut names: Vec<String>, num_layers: usize) -> Vec<String> {
    for i in 0..num_layers {
        names.push(format!("blk.{i}.attn_q.bias"));
        names.push(format!("blk.{i}.attn_k.bias"));
        names.push(format!("blk.{i}.attn_v.bias"));
    }
    names
}

// ============================================================================
// FALSIFY-ARCH-001: Complete models pass for all architectures
// ============================================================================

#[test]
fn falsify_arch_001_llama_complete() {
    let names = base_tensor_names(4);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "llama", 4).is_ok(),
        "LLaMA with all base tensors should pass"
    );
}

#[test]
fn falsify_arch_001_mistral_complete() {
    let names = base_tensor_names(4);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "mistral", 4).is_ok(),
        "Mistral with all base tensors should pass"
    );
}

#[test]
fn falsify_arch_001_qwen3_complete() {
    let names = with_qk_norm(base_tensor_names(4), 4);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "qwen3", 4).is_ok(),
        "Qwen3 with base + QK norm tensors should pass"
    );
}

#[test]
fn falsify_arch_001_qwen2_complete() {
    let names = with_bias(base_tensor_names(4), 4);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "qwen2", 4).is_ok(),
        "Qwen2 with base + bias tensors should pass"
    );
}

// ============================================================================
// FALSIFY-ARCH-002: Missing required weight produces error
// ============================================================================

#[test]
fn falsify_arch_002_qwen3_missing_q_norm() {
    // Qwen3 WITHOUT QK norm → must fail
    let names = base_tensor_names(2);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let result = enforce_architecture_completeness(&refs, "qwen3", 2);
    assert!(result.is_err(), "Qwen3 without QK norm must fail");

    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("attn_q_norm"),
        "Error must name the missing tensor: {msg}"
    );
}

#[test]
fn falsify_arch_002_qwen2_missing_bias() {
    // Qwen2 WITHOUT bias → must fail
    let names = base_tensor_names(2);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let result = enforce_architecture_completeness(&refs, "qwen2", 2);
    assert!(result.is_err(), "Qwen2 without bias must fail");

    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("bias"), "Error must mention bias: {msg}");
}

#[test]
fn falsify_arch_002_missing_attn_norm() {
    // Any arch with missing attn_norm → must fail
    let mut names = base_tensor_names(1);
    // Remove attn_norm for layer 0
    names.retain(|n| n != "blk.0.attn_norm.weight");
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let result = enforce_architecture_completeness(&refs, "llama", 1);
    assert!(result.is_err(), "Missing attn_norm must fail");
}

// ============================================================================
// FALSIFY-ARCH-003: Optional weights are allowed to be absent
// ============================================================================

#[test]
fn falsify_arch_003_llama_no_bias_ok() {
    // LLaMA does NOT require bias — base tensors alone should pass
    let names = base_tensor_names(2);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "llama", 2).is_ok(),
        "LLaMA should not require bias tensors"
    );
}

#[test]
fn falsify_arch_003_llama_no_qk_norm_ok() {
    // LLaMA does NOT require QK norm — base tensors alone should pass
    let names = base_tensor_names(2);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(
        enforce_architecture_completeness(&refs, "llama", 2).is_ok(),
        "LLaMA should not require QK norm tensors"
    );
}

// ============================================================================
// FALSIFY-ARCH-004: Missing tensor at specific layer is caught
// ============================================================================

#[test]
fn falsify_arch_004_missing_at_layer_3_not_0() {
    // 4 layers, but layer 3 is missing ffn_down
    let mut names = base_tensor_names(4);
    names.retain(|n| n != "blk.3.ffn_down.weight");
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();

    let result = enforce_architecture_completeness(&refs, "llama", 4);
    assert!(result.is_err(), "Missing tensor at layer 3 must be caught");

    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("blk.3.ffn_down.weight"),
        "Error must name the specific missing tensor: {msg}"
    );
}
