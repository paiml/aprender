
#[test]
fn test_owned_quantized_tensor_clone() {
    // Verify Clone implementation for OwnedQuantizedTensor
    let original = OwnedQuantizedTensor {
        data: vec![1, 2, 3, 4],
        in_dim: 2,
        out_dim: 2,
        qtype: GGUF_TYPE_Q8_0,
    };

    let cloned = original.clone();

    assert_eq!(cloned.data, original.data);
    assert_eq!(cloned.in_dim, original.in_dim);
    assert_eq!(cloned.out_dim, original.out_dim);
    assert_eq!(cloned.qtype, original.qtype);
}

#[test]
fn test_owned_qkv_weights_clone() {
    // Verify Clone implementation for OwnedQKVWeights
    let tensor = OwnedQuantizedTensor {
        data: vec![1, 2, 3],
        in_dim: 1,
        out_dim: 3,
        qtype: GGUF_TYPE_Q4_K,
    };

    let original = OwnedQKVWeights::Fused(tensor);
    let cloned = original.clone();

    assert_eq!(cloned.out_dim(), original.out_dim());
}

#[test]
fn test_owned_quantized_layer_clone() {
    // Verify Clone implementation for OwnedQuantizedLayer
    let original = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0, 2.0],
        attn_norm_bias: Some(vec![0.1, 0.2]),
        qkv_weight: OwnedQKVWeights::Fused(OwnedQuantizedTensor {
            data: vec![1, 2, 3],
            in_dim: 1,
            out_dim: 3,
            qtype: GGUF_TYPE_Q4_K,
        }),
        qkv_bias: None,
        attn_output_weight: OwnedQuantizedTensor {
            data: vec![4, 5],
            in_dim: 1,
            out_dim: 2,
            qtype: GGUF_TYPE_Q4_K,
        },
        attn_output_bias: None,
        ffn_up_weight: OwnedQuantizedTensor {
            data: vec![6, 7],
            in_dim: 1,
            out_dim: 2,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_up_bias: None,
        ffn_down_weight: OwnedQuantizedTensor {
            data: vec![8, 9],
            in_dim: 2,
            out_dim: 1,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        post_attn_norm_weight: None,
        post_ffw_norm_weight: None,
    };

    let cloned = original.clone();

    assert_eq!(cloned.attn_norm_weight, original.attn_norm_weight);
    assert_eq!(cloned.attn_norm_bias, original.attn_norm_bias);
    assert_eq!(cloned.qkv_weight.out_dim(), original.qkv_weight.out_dim());
}

// ── #2535: MoE placeholder dense-FFN must not read as a truncated file ──────
//
// `QuantizedTransformer::from_gguf_for_moe` fills the dense FFN slots of a
// Mixture-of-Experts model with PLACEHOLDER refs (offset=0, byte_size=0,
// num_elements=0) — a MoE model has no dense FFN at all; its weights live in
// per-expert `ffn_{up,gate,down}_exps` routed by `ffn_gate_inp`. Its doc
// comment states the contract: consumers MUST check `moe_layers[i].is_some()`
// before touching a dense FFN tensor.
//
// `validate_quantized_tensors` was such a consumer and did not check, so it
// saw `data.is_empty() && dims > 0`, concluded "truncated", and refused to load
// Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf — a file proven COMPLETE (direct
// GGUF parse: last tensor ends at 18,556,689,568 == file size; `apr validate`
// reported "VALID: 579 tensors checked, 0 contract violations").

/// The fix keys off `normalize_architecture(...) == "qwen3_moe"`. GGUF files in
/// the wild carry the arch tag as `qwen3moe` (no underscore) — that is what
/// Qwen3-Coder-30B actually reports. If this mapping ever regresses, the MoE
/// branch silently stops being taken and the false "truncated/corrupt model"
/// rejection returns, so pin it directly rather than leaving it implicit.
#[test]
fn moe_arch_tag_normalizes_to_the_form_the_validator_keys_on() {
    use crate::tensor_names::normalize_architecture;
    assert_eq!(
        normalize_architecture("qwen3moe"),
        "qwen3_moe",
        "#2535: the on-disk GGUF arch tag must normalize to the value \
         validate_quantized_tensors compares against, or the dense-FFN skip \
         is dead code and complete MoE files are rejected as corrupt"
    );
    // Already-canonical input must be stable (idempotent), otherwise a
    // double-normalization anywhere would fall out of the MoE branch.
    assert_eq!(normalize_architecture("qwen3_moe"), "qwen3_moe");
}

/// Non-vacuity companion. Without this, the test above would still pass if
/// `normalize_architecture` returned "qwen3_moe" for EVERYTHING, which would
/// make every dense model skip its FFN truncation checks and silently disarm
/// PMAT-750's fail-closed guarantee.
#[test]
fn a_dense_arch_does_not_normalize_into_the_moe_branch() {
    use crate::tensor_names::normalize_architecture;
    for dense in ["llama", "qwen2", "phi2"] {
        assert_ne!(
            normalize_architecture(dense),
            "qwen3_moe",
            "#2535: dense arch '{dense}' must NOT take the MoE branch — it would \
             skip real ffn_up/ffn_down truncation checks"
        );
    }
}
