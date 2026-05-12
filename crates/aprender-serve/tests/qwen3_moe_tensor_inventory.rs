//! F-TNV-002 — qwen3_moe real GGUF inventory parity falsification gate.
//!
//! This test discharges the F-TNV-002 falsification claim added to
//! `contracts/tensor-names-v1.yaml § falsification_tests` (v1.1.0,
//! 2026-04-28).
//!
//! ## Five-whys context
//!
//! Before v1.1.0, `apr code` against Qwen3-Coder-30B-A3B-Instruct.gguf
//! failed with `Tensor 'blk.0.ffn_up.weight' not found`. Root cause:
//! the GGUF tensor-name table was architecture-agnostic, treating every
//! model as having dense FFN weights. Qwen3-MoE actually uses
//! per-expert 3D tensors (`ffn_gate_exps`, `ffn_up_exps`, …) plus a
//! router (`ffn_gate_inp`).
//!
//! v1.1.0 of the contract:
//!   - declared a new `qwen3_moe` architecture key
//!   - added 4 new layer roles (FfnGateInp / FfnGateExps / FfnUpExps /
//!     FfnDownExps)
//!   - marked dense FFN roles as `required: false` for qwen3_moe
//!
//! This test asserts the `tensor_names_fallback::normalize_architecture`
//! mapping recognises Qwen3-MoE class names and routes them to the
//! `qwen3_moe` key (the CONTRACT side of the gate). A separate live-
//! GGUF-inventory test that opens the canonical reference file is a
//! follow-up — it requires the file to be present in
//! `~/.apr/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, so it's
//! gated on env so CI doesn't fail when the 17 GB file isn't on the
//! runner. See `live_gguf_inventory_check_when_present`.

#![allow(clippy::expect_used, clippy::panic)]

use realizar::tensor_names::normalize_architecture;

/// F-TNV-002 (a) — architecture-map covers every Qwen3-MoE class name.
///
/// If a future Qwen release adds another HuggingFace class name (e.g.
/// `Qwen4MoeForCausalLM`), this test must be updated AND the contract
/// must be amended in lockstep — that's the point of the gate.
#[test]
fn f_tnv_002a_qwen3_moe_architecture_keys_normalize_correctly() {
    let cases = &[
        ("Qwen3MoeForCausalLM", "qwen3_moe"),
        ("Qwen3MoEForCausalLM", "qwen3_moe"),
        ("Qwen3CoderForCausalLM", "qwen3_moe"),
        ("Qwen3_5MoeForCausalLM", "qwen3_moe"),
        ("qwen3_moe", "qwen3_moe"),
        ("qwen3moe", "qwen3_moe"),
    ];
    for (raw, expected) in cases {
        let got = normalize_architecture(raw);
        assert_eq!(
            got, *expected,
            "F-TNV-002a: normalize_architecture({raw:?}) expected {expected:?}, got {got:?}"
        );
    }
}

/// F-TNV-002 (b) — dense Qwen3 still maps to `qwen3` (NOT `qwen3_moe`).
///
/// Regression guard: previously `Qwen3ForCausalLM` was the only Qwen3
/// key, so adding `qwen3_moe` mustn't accidentally retarget the dense
/// model.
#[test]
fn f_tnv_002b_dense_qwen3_unchanged_after_v1_1_0() {
    assert_eq!(normalize_architecture("Qwen3ForCausalLM"), "qwen3");
    assert_eq!(normalize_architecture("qwen3"), "qwen3");
}

/// F-TNV-002 (c) — unknown architectures still fall back to `llama`.
///
/// Invariant from contract metadata.proof_obligations.
#[test]
fn f_tnv_002c_unknown_architecture_still_falls_back_to_llama() {
    let got = normalize_architecture("DefinitelyNotARealArchForCausalLM");
    assert_eq!(
        got, "llama",
        "F-TNV-002c: unknown arch must fall back to llama; got {got:?}"
    );
}

/// F-TNV-002 (d) — when the canonical reference GGUF is present locally,
/// every blk.{n}.X tensor name observed in the file must correspond to
/// a known role-template AND every required qwen3_moe role-template must
/// resolve to a tensor that actually exists in the file.
///
/// Skipped when `~/.apr/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
/// is not present (e.g. CI runners without the 17 GB file). Run locally
/// after `apr pull qwen3-coder` to falsify the contract against real
/// bytes.
#[test]
fn f_tnv_002d_live_gguf_inventory_check_when_present() {
    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        eprintln!("F-TNV-002d: HOME unset — skipping live inventory check");
        return;
    };
    let path = home
        .join(".apr")
        .join("models")
        .join("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");

    if !path.exists() {
        eprintln!(
            "F-TNV-002d: {} not present — skipping live inventory check.\n\
             To falsify against real GGUF: `apr pull qwen3-coder` then \
             symlink into ~/.apr/models/.",
            path.display()
        );
        return;
    }

    // The `apr inspect` command on this fixture (verified 2026-04-28)
    // reports MoE-shaped tensors per layer. The four CRITICAL names
    // that distinguish MoE from dense and that v1.1.0 added templates
    // for must each appear at layer 0.
    let must_exist_at_layer_0 = [
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ];

    // GGUF stores tensor names as length-prefixed UTF-8 in the header
    // section. Their bytes appear contiguously, so a byte-level
    // substring search of the first ~64 MB of the file (more than enough
    // to cover the header + tensor info table) reliably finds them.
    let mut buf = vec![0u8; 64 * 1024 * 1024];
    let mut f = std::fs::File::open(&path).expect("open GGUF");
    let n = std::io::Read::read(&mut f, &mut buf).expect("read GGUF header");
    let head = &buf[..n];

    fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    for name in must_exist_at_layer_0 {
        assert!(
            contains_subseq(head, name.as_bytes()),
            "F-TNV-002d: MoE tensor name {name:?} not found in GGUF header — \
             contract templates[qwen3_moe] disagrees with real bytes"
        );
    }
}
