//! PMAT-785: centralized GPU-resident quant-eligibility gate tests.
//!
//! These tests lock the single source of truth used by BOTH the primary
//! `apr run`/`apr serve` path (`infer::is_legacy_gguf_quant` /
//! `model_has_legacy_quant`) AND the construction-time gate that protects every
//! serve `generate_gpu_resident` entry point
//! (`OwnedQuantizedModel::has_gpu_unsupported_quant`, called from
//! `OwnedQuantizedModelCuda::with_max_seq_len`).
//!
//! Invariant: a model carrying a quant type WITHOUT a verified GPU GEMV kernel
//! must be flagged so it routes to CPU (loud) or errors, never shipping silent
//! Q4_K-decode garbage on the GPU (PMAT-781/783 class).

use crate::gguf::gpu_unsupported_quant_qtype;
use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{ArchConstraints, GGUFConfig, OwnedQKVWeights};

fn test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

/// The whitelist predicate is the single source of truth. GPU-eligible set is
/// exactly {F32(0), Q4_0(2), Q4_1(3), Q5_0(6), Q8_0(8), Q4_K(12), Q5_K(13),
/// Q6_K(14)}; everything else gates to CPU. This is what the construction gate
/// and the primary-path gate both consume — they MUST agree.
#[test]
fn gpu_unsupported_quant_qtype_whitelist_is_exact() {
    // Supported → NOT gated.
    for &q in &[0u32, 2, 3, 6, 8, 12, 13, 14] {
        assert!(
            !gpu_unsupported_quant_qtype(q),
            "qtype {q} has a verified GPU kernel and must be GPU-eligible"
        );
    }
    // Unsupported → gated to CPU (would hit resolve_qtype's unwrap_or(Q4K)).
    for &q in &[
        1u32, /*F16*/
        7,    /*Q5_1*/
        9,    /*Q8_1*/
        10,   /*Q2_K*/
        11,   /*Q3_K*/
        15,   /*Q8_K*/
        30,   /*BF16*/
        100,  /*IQ*/
    ] {
        assert!(
            gpu_unsupported_quant_qtype(q),
            "qtype {q} has no verified GPU kernel and MUST force CPU"
        );
    }
}

/// A model whose every projection tensor is Q4_K must NOT be flagged: supported
/// quants stay GPU-eligible (no regression).
#[test]
fn supported_q4k_model_is_gpu_eligible() {
    let model = create_test_model_with_config(&test_config());
    assert!(
        !model.has_gpu_unsupported_quant(),
        "all-Q4K model must remain GPU-eligible (no regression)"
    );
}

/// An unsupported quant hidden in the lm_head must flag the whole model.
#[test]
fn unsupported_quant_in_lm_head_forces_cpu() {
    let mut model = create_test_model_with_config(&test_config());
    model.lm_head_weight.qtype = 7; // Q5_1 — no GPU kernel
    assert!(
        model.has_gpu_unsupported_quant(),
        "Q5_1 in lm_head must force CPU"
    );
}

/// An unsupported quant hidden ONLY in the fused QKV tensor must flag the model
/// (the pre-PMAT-783 gate omitted QKV; the centralized method must cover it).
#[test]
fn unsupported_quant_in_qkv_forces_cpu() {
    let mut model = create_test_model_with_config(&test_config());
    if let OwnedQKVWeights::Fused(t) = &mut model.layers[0].qkv_weight {
        t.qtype = 10; // Q2_K — no GPU kernel
    }
    assert!(
        model.has_gpu_unsupported_quant(),
        "Q2_K hidden in QKV must force CPU"
    );
}

/// An unsupported quant hidden ONLY in the FFN gate must flag the model.
#[test]
fn unsupported_quant_in_ffn_gate_forces_cpu() {
    let mut model = create_test_model_with_config(&test_config());
    if let Some(g) = model.layers[0].ffn_gate_weight.as_mut() {
        g.qtype = 11; // Q3_K — no GPU kernel
        assert!(
            model.has_gpu_unsupported_quant(),
            "Q3_K hidden in the FFN gate must force CPU"
        );
    }
}

/// An unsupported quant in attn_output / ffn_up / ffn_down must each flag the
/// model — every tensor the GPU-resident forward pass touches is covered.
#[test]
fn unsupported_quant_in_each_projection_forces_cpu() {
    let cfg = test_config();

    let mut m_out = create_test_model_with_config(&cfg);
    m_out.layers[0].attn_output_weight.qtype = 9; // Q8_1
    assert!(
        m_out.has_gpu_unsupported_quant(),
        "Q8_1 in attn_output must force CPU"
    );

    let mut m_up = create_test_model_with_config(&cfg);
    m_up.layers[0].ffn_up_weight.qtype = 15; // Q8_K
    assert!(
        m_up.has_gpu_unsupported_quant(),
        "Q8_K in ffn_up must force CPU"
    );

    let mut m_down = create_test_model_with_config(&cfg);
    m_down.layers[0].ffn_down_weight.qtype = 1; // F16
    assert!(
        m_down.has_gpu_unsupported_quant(),
        "F16 in ffn_down must force CPU"
    );
}
