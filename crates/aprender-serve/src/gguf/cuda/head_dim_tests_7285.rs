//! #7285: the CUDA executor must be sized with the model's real per-head width.
//!
//! `OwnedQuantizedModelCuda` sized the GPU KV cache, the workspace (`q_dim = heads * head_dim`)
//! and the per-head QK-norm with `hidden_dim / num_heads`. For Qwen3-0.6B that is 1024 / 16 = 64,
//! but the file says `attention.key_length = 128` (q_dim 2048, kv_dim 1024). The GPU therefore ran
//! with half-width heads and the load-time parity gate (cosine 0.453) rightly refused it, so
//! `apr` served the model on the CPU at about 7 tok/s. `GGUFConfig::head_dim()` is the one place
//! that honours the explicit value.
//!
//! These tests need no GPU: the first pins the arithmetic of the real model, the second is a source
//! guard that fails while any executor-shape site still divides by hand.

use crate::gguf::{ArchConstraints, GGUFConfig};

fn qwen3_0_6b() -> GGUFConfig {
    GGUFConfig {
        architecture: "qwen3".to_string(),
        constraints: ArchConstraints::from_architecture("qwen3"),
        hidden_dim: 1024,
        intermediate_dim: 3072,
        num_layers: 28,
        num_heads: 16,
        num_kv_heads: 8,
        vocab_size: 151_936,
        context_length: 40_960,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        rope_type: 2,
        explicit_head_dim: Some(128),
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: Some(151_645),
    }
}

#[test]
fn qwen3_0_6b_head_dim_is_the_file_value_not_hidden_over_heads() {
    let c = qwen3_0_6b();
    assert_eq!(
        c.hidden_dim / c.num_heads,
        64,
        "the derivation the GPU path used"
    );
    assert_eq!(c.head_dim(), 128, "attention.key_length");
    assert_eq!(c.num_heads * c.head_dim(), 2048, "q_dim");
    assert_eq!(c.num_kv_heads * c.head_dim(), 1024, "kv_dim");
}

/// Code lines (comments dropped, whitespace removed) that derive a head width by hand.
fn hand_derived(src: &str) -> Vec<String> {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .map(|l| l.split_whitespace().collect::<String>())
        .filter(|l| l.contains("hidden_dim/") && l.contains("num_heads"))
        .collect()
}

#[test]
fn no_cuda_executor_shape_site_divides_hidden_dim_by_num_heads() {
    let sites = [
        ("mod.rs", include_str!("mod.rs")),
        ("mod_parity_gate.rs", include_str!("mod_parity_gate.rs")),
        (
            "generate_batched_streaming.rs",
            include_str!("generate_batched_streaming.rs"),
        ),
    ];
    let mut bad = Vec::new();
    for (name, src) in sites {
        for line in hand_derived(src) {
            bad.push(format!("{name}: {line}"));
        }
    }
    assert!(
        bad.is_empty(),
        "use GGUFConfig::head_dim() (#7285), not hidden_dim / num_heads:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_guard_sees_a_hand_derivation() {
    // Mutant check: the scanner must flag the shape it exists to catch.
    let bad = "let head_dim = model.config.hidden_dim / model.config.num_heads;\n";
    assert_eq!(hand_derived(bad).len(), 1);
    assert!(hand_derived("// hidden_dim / num_heads\nlet h = c.head_dim();\n").is_empty());
}
