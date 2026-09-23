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

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    gpu_qtype_excluded_on, gpu_unsupported_quant_qtype_on, GPU_QTYPES_UNLOADABLE_AT_CC,
    GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR,
};
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
/// exactly {F32(0), F16(1), Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7), Q8_0(8),
/// Q2_K(10), Q4_K(12), Q5_K(13), Q6_K(14), IQ2_XXS(16), IQ3_XXS(18), IQ4_NL(20), IQ3_S(21), IQ2_S(22),
/// IQ4_XS(23), BF16(30)}; everything
/// else gates to CPU. This is what the construction gate and the primary-path
/// gate both consume — they MUST agree.
#[test]
fn gpu_unsupported_quant_qtype_whitelist_is_exact() {
    // Supported → NOT gated.
    // #3477: F16(1) and IQ4_XS(23) MOVED to the supported set once their GEMV
    // kernels existed AND were measured against the CPU decoder on real model
    // bytes — 217/217 tensors exact at `88d25d265` for F16, 10/10 exact at
    // `b782b4257` for IQ4_XS, each with a planted-fault control going RED on a
    // tensor of its own type. Not moved on the kernel's existence alone.
    // #3869/#3884/#3885: IQ4_NL(20), IQ3_S(21) and Q5_1(7) joined the supported
    // set on the same terms — a GEMV kernel measured against the CPU decoder on
    // device, each with planted faults proven RED first. Q5_1's were the 5th bit
    // dropped and the affine min dropped; IQ3_S's were the scale nibble, the 9th
    // grid bit and the sign bits. Never moved on a kernel's existence alone.
    // #3950: IQ2_XXS(16) joined on the same terms — 95/95 real tensors of
    // Qwen3.5-0.8B-UD-IQ2_XXS over all 6 shapes, |err|/sum|w||x| <= 2.7e-7,
    // four planted faults RED first.
    // #3908: BF16(30) joined on the same terms and is the only one measured
    // BIT-EXACT (0 ULP, 64 rows, RTX 4090 sm_89) rather than within a tolerance,
    // because bf16 decoding rounds nothing. Faults: shift 8 not 16, byte-swapped
    // halfword, and row stride k not k*2 (the LAYOUT-001 fault) - all RED first.
    // #3953/#3963: IQ2_S(22) and IQ3_XXS(18) joined in one combined admission,
    // each on every real tensor and shape of Qwen3.5-0.8B-UD-IQ2_XXS with its CPU
    // decoder proven bit-exact against gguf-py first, and planted faults RED.
    // #3960: Q2_K(10) joined in the same combined admission (every real value
    // bitwise vs gguf-py; A/B 4.3e-9; four faults RED on real bytes).
    for &q in &[
        0u32, 1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 21, 22, 23, 30,
    ] {
        assert!(
            !gpu_unsupported_quant_qtype_on(q, None),
            "qtype {q} has a verified GPU kernel and must be GPU-eligible"
        );
    }
    // Unsupported → gated to CPU (would hit resolve_qtype's unwrap_or(Q4K)).
    for &q in &[
        9u32, /*Q8_1*/
        11,   /*Q3_K*/
        15,   /*Q8_K*/
        17,   /*IQ2_XS*/
        100,  /*IQ*/
    ] {
        assert!(
            gpu_unsupported_quant_qtype_on(q, None),
            "qtype {q} has no verified GPU kernel and MUST force CPU"
        );
    }
}

/// #3968/#4096: IQ3_S(21) and IQ2_S(22) are whitelisted in general but refused on a
/// device of compute-capability major >= 12, where their kernels do not load. Below the
/// threshold, and with no device known, the device-independent list applies unchanged.
#[test]
fn iq3s_iq2s_are_refused_at_cc12_and_admitted_below() {
    assert_eq!(GPU_QTYPES_UNLOADABLE_AT_CC, [21, 22]);
    assert_eq!(GPU_QTYPE_EXCLUSION_MIN_CC_MAJOR, 12);
    for q in [21u32, 22] {
        assert!(
            !gpu_unsupported_quant_qtype_on(q, None),
            "ggml {q}: no device known -> the general list admits it"
        );
        assert!(
            !gpu_unsupported_quant_qtype_on(q, Some(8)),
            "ggml {q}: sm_89 loads it (lambda, 163/163)"
        );
        assert!(
            !gpu_unsupported_quant_qtype_on(q, Some(11)),
            "ggml {q}: below the threshold is unchanged"
        );
        assert!(
            gpu_unsupported_quant_qtype_on(q, Some(12)),
            "ggml {q}: sm_12x must refuse it (#4096)"
        );
        assert!(gpu_qtype_excluded_on(q, Some(12)));
    }
}

/// The exclusion is NARROW: at cc 12 every other whitelisted type is still admitted, and a
/// type outside the whitelist is still refused below the threshold.
#[test]
fn the_cc12_exclusion_touches_only_its_listed_types() {
    for q in [0u32, 1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 16, 18, 20, 23, 30] {
        assert!(
            !gpu_unsupported_quant_qtype_on(q, Some(12)),
            "ggml {q} must stay GPU-eligible on sm_12x"
        );
        assert!(!gpu_qtype_excluded_on(q, Some(12)));
    }
    for q in [9u32, 11, 15, 17] {
        assert!(
            gpu_unsupported_quant_qtype_on(q, Some(8)),
            "ggml {q} has no kernel on any device"
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
///
/// #3885: the example was Q5_1(7) until Q5_1 got a measured GEMV kernel, then
/// IQ3_XXS(18) until #3963 measured IQ3_XXS's. Re-aimed at IQ2_XS(17): a real
/// ggml type with no GPU kernel (no claim that it is present in the fleet). A
/// row is re-aimed, never deleted.
#[test]
fn unsupported_quant_in_lm_head_forces_cpu() {
    let mut model = create_test_model_with_config(&test_config());
    model.lm_head_weight.qtype = 17; // IQ2_XS — no GPU kernel
    assert!(
        model.has_gpu_unsupported_quant(),
        "IQ2_XS in lm_head must force CPU"
    );
}

/// An unsupported quant hidden ONLY in the fused QKV tensor must flag the model
/// (the pre-PMAT-783 gate omitted QKV; the centralized method must cover it).
#[test]
fn unsupported_quant_in_qkv_forces_cpu() {
    let mut model = create_test_model_with_config(&test_config());
    if let OwnedQKVWeights::Fused(t) = &mut model.layers[0].qkv_weight {
        t.qtype = 19; // IQ1_S — no GPU kernel (was Q2_K until #3960 measured it)
    }
    assert!(
        model.has_gpu_unsupported_quant(),
        "IQ1_S hidden in QKV must force CPU"
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
    // Was F16(1) until #3477 gave it a measured kernel, then BF16(30) until
    // #3908 did the same. Re-aimed at IQ1_M(29), which has no kernel - a row
    // is re-aimed, never deleted.
    m_down.layers[0].ffn_down_weight.qtype = 29; // IQ1_M
    assert!(
        m_down.has_gpu_unsupported_quant(),
        "IQ1_M in ffn_down must force CPU"
    );
}

/// #3685: `first_gpu_unsupported_quant` NAMES what the gate refuses, and a supported model
/// never produces a name, so `apr parity` never reports a refusal for a GPU-eligible quant.
/// `has_gpu_unsupported_quant` is defined through it; the two must agree on every model.
#[test]
fn first_gpu_unsupported_quant_names_the_type_and_stays_none_for_supported() {
    let supported = create_test_model_with_config(&test_config());
    assert_eq!(
        supported.first_gpu_unsupported_quant(),
        None,
        "all-Q4K: nothing to name"
    );

    let mut lm = create_test_model_with_config(&test_config());
    // Was Q5_1(7) until #3885, then IQ3_XXS(18) until #3963 gave each a kernel.
    lm.lm_head_weight.qtype = 17; // IQ2_XS — no GPU kernel
    assert_eq!(lm.first_gpu_unsupported_quant(), Some(17));

    let mut down = create_test_model_with_config(&test_config());
    // Was F16(1), the #3685 model, then BF16(30). Both now HAVE measured
    // kernels (#3477, #3908), so neither is an example of an unsupported quant;
    // IQ1_M(29) is, and the property under test - that the first unsupported
    // projection is named - is unchanged.
    down.layers[0].ffn_down_weight.qtype = 29; // IQ1_M
    assert_eq!(down.first_gpu_unsupported_quant(), Some(29));

    for m in [&supported, &lm, &down] {
        assert_eq!(
            m.has_gpu_unsupported_quant(),
            m.first_gpu_unsupported_quant().is_some()
        );
    }
}
