//! #3992: the dense CUDA constructors refuse a Mixture-of-Experts model by name.
//!
//! The refusal runs before `CudaExecutor::new`, so these tests need no GPU: the
//! point is that a qwen3moe model never reaches the dense forward, where it failed
//! with `ffn_gate_ptr is null (0)`, an error that names neither the model nor
//! the route.

use crate::api::test_helpers::create_test_quantized_model;
use crate::gguf::{ArchConstraints, GGUFConfig, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn model(architecture: &str) -> OwnedQuantizedModel {
    let config = GGUFConfig {
        architecture: architecture.to_string(),
        constraints: ArchConstraints::from_architecture(architecture),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    create_test_quantized_model(&config)
}

#[test]
fn the_dense_constructors_refuse_a_moe_model_by_name() {
    // `qwen35moe` is the real Qwen3.5-35B-A3B string; the table maps it to
    // is_moe=false, so this is the case a flag-only check would let through.
    for arch in ["qwen3moe", "qwen3_moe", "qwen35moe"] {
        let m = model(arch);
        let err = OwnedQuantizedModelCuda::with_max_seq_len(m, 0, 64)
            .err()
            .unwrap_or_else(|| panic!("{arch}: a MoE model was accepted by the dense constructor"));
        let msg = err.to_string();
        assert!(msg.contains("Mixture-of-Experts"), "{arch}: {msg}");
        assert!(
            msg.contains(arch),
            "{arch}: the refusal must name the model: {msg}"
        );
        assert!(
            msg.contains("apr run"),
            "{arch}: the refusal must name the route: {msg}"
        );
        // Recoverable, like every other CudaInitError: the model comes back.
        assert_eq!(err.into_model().config().architecture, arch);

        let msg = OwnedQuantizedModelCuda::new(model(arch), 0)
            .err()
            .expect("`new` delegates to the same gate")
            .to_string();
        assert!(msg.contains("Mixture-of-Experts"), "{arch} via new: {msg}");
    }
}

/// Positive control: a dense model is not refused by THIS gate. It may still fail
/// later (no GPU in this test), but never with the MoE refusal.
#[test]
fn a_dense_model_is_not_refused_as_moe() {
    let m = model("llama");
    assert!(!m.config().constraints.is_moe);
    if let Err(e) = OwnedQuantizedModelCuda::with_max_seq_len(m, 0, 64) {
        assert!(
            !e.to_string().contains("Mixture-of-Experts"),
            "a dense model was refused as MoE: {e}"
        );
    }
}

/// The refusal must precede `CudaExecutor::new` and every weight upload, or a
/// refused 18 GB MoE file would first be uploaded to the card (outside any lock).
/// The no-GPU test above shows it precedes `CudaExecutor::new` at runtime; this
/// pins the order in the source, so a reordering fails without a GPU either.
#[test]
fn the_moe_refusal_runs_before_cuda_init_and_upload() {
    let src = include_str!("mod.rs");
    let body = src
        .split("pub fn with_max_seq_len(")
        .nth(1)
        .and_then(|t| t.split("\n    }\n").next())
        .expect("with_max_seq_len body");
    let check = body
        .find("Self::check_not_moe(model)")
        .expect("with_max_seq_len calls check_not_moe");
    let build = body
        .find("Self::build(")
        .expect("with_max_seq_len delegates to build");
    assert!(check < build, "check_not_moe must run before build: {body}");
    assert!(
        !body.contains("CudaExecutor::new"),
        "no CUDA init before the check: {body}"
    );
    // `build` is where CUDA init and the upload (`preload_and_verify`) happen.
    let build_body = src.split("fn build(").nth(1).expect("build body");
    assert!(build_body.contains("CudaExecutor::new("));
    // `new` goes through the gate too.
    let new_body = src
        .split("pub fn new(model: OwnedQuantizedModel")
        .nth(1)
        .expect("new");
    let new_body = new_body.split("\n    }\n").next().expect("new body");
    assert!(
        new_body.contains("Self::with_max_seq_len("),
        "`new` must delegate to the gate: {new_body}"
    );
}
