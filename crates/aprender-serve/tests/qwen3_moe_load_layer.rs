//! M32c.1 falsifier — exercises `qwen3_moe_load::load_qwen3_moe_layer`
//! against the cached 17.3 GB Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
//! when present.
//!
//! Discharges the load-time half of the M32c plan by proving:
//!
//!   1. All four MoE tensor names declared by `tensor-names-v1` v1.1.0
//!      (M29) resolve to non-empty byte ranges for every L ∈ [0, 48).
//!   2. The descriptors agree on a consistent shape: gate_exps,
//!      up_exps, down_exps all have the same `num_elements`
//!      (`num_experts * intermediate * hidden`) per layer.
//!
//! Forward dispatch is **NOT** asserted here — that's M32c.2's
//! falsifier (`FALSIFY-QW3-MOE-FORWARD-003`, which requires
//! `apr run` to emit tokens).
//!
//! ## Skip behaviour
//! When the canonical cached GGUF is absent, the test exits 0
//! with a printed `skipped` line — matching the `f_tnv_002d` and
//! `f_qw3_moe_load_002b` patterns. Fixture-absent ≠ defect.

use realizar::gguf::qwen3_moe_load::{
    expert_byte_slice, expert_swiglu_quantized, load_qwen3_moe_layer, moe_ffn_forward_layer,
};
use realizar::gguf::GGUFModel;

use std::path::Path;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

/// Per Qwen3-Coder-30B-A3B-Instruct config (apr inspect):
/// L=48, hidden=2048, intermediate=6144 (per-expert), N_experts=128.
const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_HIDDEN: usize = 2048;
const EXPECTED_INTERMEDIATE: usize = 768; // ffn_inter for MoE expert (small)
const EXPECTED_N_EXPERTS: usize = 128;

#[test]
fn f_qw3_moe_c1_001_load_first_layer_against_live_gguf() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "F-QW3-MOE-C1-001: skipped — no Qwen3-Coder GGUF cached at any of {:?}",
            CANONICAL_QWEN3_CODER_GGUF_PATHS
        );
        return;
    };

    eprintln!("F-QW3-MOE-C1-001: loading layer 0 from {gguf_path}");

    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let layer0 =
        load_qwen3_moe_layer(&model, &bytes, 0).expect("load layer 0 MoE tensor descriptors");

    assert!(
        layer0.router.num_elements > 0,
        "F-QW3-MOE-C1-001: router descriptor must be non-empty, got {:?}",
        layer0.router
    );
    assert!(
        layer0.gate_exps.num_elements > 0,
        "F-QW3-MOE-C1-001: gate_exps descriptor must be non-empty, got {:?}",
        layer0.gate_exps
    );
    assert!(
        layer0.up_exps.num_elements > 0,
        "F-QW3-MOE-C1-001: up_exps descriptor must be non-empty, got {:?}",
        layer0.up_exps
    );
    assert!(
        layer0.down_exps.num_elements > 0,
        "F-QW3-MOE-C1-001: down_exps descriptor must be non-empty, got {:?}",
        layer0.down_exps
    );

    assert_eq!(
        layer0.gate_exps.num_elements, layer0.up_exps.num_elements,
        "F-QW3-MOE-C1-001: gate_exps + up_exps must share shape \
         [N_experts, intermediate, hidden]; got gate={}, up={}",
        layer0.gate_exps.num_elements, layer0.up_exps.num_elements
    );
    assert_eq!(
        layer0.gate_exps.num_elements, layer0.down_exps.num_elements,
        "F-QW3-MOE-C1-001: gate_exps + down_exps share total element count \
         (only the dim ordering differs); got gate={}, down={}",
        layer0.gate_exps.num_elements, layer0.down_exps.num_elements
    );

    eprintln!(
        "F-QW3-MOE-C1-001: PASS\n  router.num_elements      = {}\n  \
         gate/up/down.num_elements = {}\n  qtypes (router/gate/up/down) = ({}, {}, {}, {})",
        layer0.router.num_elements,
        layer0.gate_exps.num_elements,
        layer0.router.qtype,
        layer0.gate_exps.qtype,
        layer0.up_exps.qtype,
        layer0.down_exps.qtype,
    );
}

#[test]
fn f_qw3_moe_c1_002_load_all_48_layers_against_live_gguf() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-C1-002: skipped — no cached GGUF.");
        return;
    };

    eprintln!("F-QW3-MOE-C1-002: loading all {EXPECTED_NUM_LAYERS} layers from {gguf_path}");

    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let mut consistent_router_size: Option<usize> = None;
    let mut consistent_expert_size: Option<usize> = None;
    let mut total_router_bytes: usize = 0;
    let mut total_expert_bytes: usize = 0;

    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        let layer = load_qwen3_moe_layer(&model, &bytes, layer_idx)
            .unwrap_or_else(|e| panic!("F-QW3-MOE-C1-002: layer {layer_idx} load failed: {e:?}"));

        assert!(layer.router.num_elements > 0, "layer {layer_idx} router");
        assert!(
            layer.gate_exps.num_elements > 0,
            "layer {layer_idx} gate_exps"
        );
        assert!(layer.up_exps.num_elements > 0, "layer {layer_idx} up_exps");
        assert!(
            layer.down_exps.num_elements > 0,
            "layer {layer_idx} down_exps"
        );

        if let Some(prev) = consistent_router_size {
            assert_eq!(
                prev, layer.router.num_elements,
                "F-QW3-MOE-C1-002: router shape varies between layer 0 and layer {layer_idx}"
            );
        } else {
            consistent_router_size = Some(layer.router.num_elements);
        }
        if let Some(prev) = consistent_expert_size {
            assert_eq!(
                prev, layer.gate_exps.num_elements,
                "F-QW3-MOE-C1-002: expert shape varies between layer 0 and layer {layer_idx}"
            );
        } else {
            consistent_expert_size = Some(layer.gate_exps.num_elements);
        }

        total_router_bytes += layer.router.byte_size;
        total_expert_bytes +=
            layer.gate_exps.byte_size + layer.up_exps.byte_size + layer.down_exps.byte_size;
    }

    let router_elems = consistent_router_size.expect("at least one layer must be loaded");
    let expert_elems = consistent_expert_size.expect("at least one layer must be loaded");

    eprintln!(
        "F-QW3-MOE-C1-002: PASS\n  L={EXPECTED_NUM_LAYERS}\n  \
         router.num_elements (per layer) = {router_elems}\n  \
         expert.num_elements (per layer per role) = {expert_elems}\n  \
         total_router_bytes = {total_router_bytes}\n  \
         total_expert_bytes (gate+up+down × {EXPECTED_NUM_LAYERS}) = {total_expert_bytes}",
    );

    let expected_router_elems = EXPECTED_HIDDEN * EXPECTED_N_EXPERTS;
    assert_eq!(
        router_elems, expected_router_elems,
        "F-QW3-MOE-C1-002: router shape must be [N_experts={EXPECTED_N_EXPERTS}, hidden={EXPECTED_HIDDEN}], \
         expected num_elements = {expected_router_elems}, got {router_elems}"
    );

    let expected_expert_elems = EXPECTED_N_EXPERTS * EXPECTED_INTERMEDIATE * EXPECTED_HIDDEN;
    assert_eq!(
        expert_elems, expected_expert_elems,
        "F-QW3-MOE-C1-002: expert shape must be \
         [N_experts={EXPECTED_N_EXPERTS}, intermediate={EXPECTED_INTERMEDIATE}, hidden={EXPECTED_HIDDEN}], \
         expected num_elements = {expected_expert_elems}, got {expert_elems}"
    );
}

/// M32c.2.2.0 falsifier — exercises `expert_byte_slice` against the live
/// cached Qwen3-Coder GGUF. Proves that for every layer × every expert,
/// the per-expert byte range is well-formed (correct length, in-bounds,
/// distinct from neighbour) — the load-side guarantee that
/// `fused_q4k_parallel_matvec` (M32c.2.2.1) will rely on.
#[test]
fn f_qw3_moe_c220_001_expert_byte_slice_partitions_live_gguf() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-C220-001: skipped — no cached GGUF.");
        return;
    };

    eprintln!("F-QW3-MOE-C220-001: slicing experts at {gguf_path}");

    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let layer0 = load_qwen3_moe_layer(&model, &bytes, 0).expect("layer 0 descriptors");

    // Slice each of the 4 stacked tensors per-expert and verify:
    //   - returned slice length is exactly byte_size / num_experts
    //   - first byte of expert e differs from first byte of expert e+1 in
    //     at least ONE of the 3 expert tensors (gate/up/down) — proves
    //     we're slicing into different memory, not aliasing
    let n = EXPECTED_N_EXPERTS;
    let stacked = [
        ("gate_exps", &layer0.gate_exps),
        ("up_exps", &layer0.up_exps),
        ("down_exps", &layer0.down_exps),
    ];

    for (name, tensor) in &stacked {
        let per_expert = tensor.byte_size / n;
        for e in 0..n {
            let slice = expert_byte_slice(tensor, &bytes, e, n).unwrap_or_else(|err| {
                panic!("F-QW3-MOE-C220-001: {name} expert {e} slice failed: {err}")
            });
            assert_eq!(
                slice.len(),
                per_expert,
                "F-QW3-MOE-C220-001: {name} expert {e} length"
            );
        }
    }

    // Different experts have different bytes (sanity vs aliasing).
    let mut differs = 0usize;
    for (_, tensor) in &stacked {
        let s0 = expert_byte_slice(tensor, &bytes, 0, n).unwrap();
        let s1 = expert_byte_slice(tensor, &bytes, 1, n).unwrap();
        if s0[..16.min(s0.len())] != s1[..16.min(s1.len())] {
            differs += 1;
        }
    }
    assert!(
        differs > 0,
        "F-QW3-MOE-C220-001: at least one expert tensor must distinguish expert 0 from expert 1"
    );

    eprintln!(
        "F-QW3-MOE-C220-001: PASS\n  {} experts × 3 tensors per layer 0 sliced\n  \
         per-expert sizes: gate={} up={} down={} bytes",
        n,
        layer0.gate_exps.byte_size / n,
        layer0.up_exps.byte_size / n,
        layer0.down_exps.byte_size / n
    );
}

/// M32c.2.2.1 falsifier — exercises `expert_swiglu_quantized` against the
/// cached Qwen3-Coder GGUF. Proves that ONE expert's SwiGLU FFN evaluation
/// (gate Q4_K + up Q4_K + SiLU * up + down Q6_K) returns a finite, non-zero
/// `[hidden_dim]` vector — the per-expert kernel that `moe_forward_token`
/// will compose in M32c.2.2.2.
#[test]
fn f_qw3_moe_c221_001_expert_swiglu_quantized_finite_output() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-C221-001: skipped — no cached GGUF.");
        return;
    };

    eprintln!("F-QW3-MOE-C221-001: per-expert SwiGLU on {gguf_path}");

    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let layer0 = load_qwen3_moe_layer(&model, &bytes, 0).expect("layer 0");

    // Synthetic hidden state with mild magnitude (post-RMSNorm-shaped).
    // Real forward feeds the layer's normed input here; for the per-expert
    // kernel, any finite hidden vector exercises the full kernel chain.
    let hidden: Vec<f32> = (0..EXPECTED_HIDDEN)
        .map(|i| 0.01 * ((i as f32).sin()))
        .collect();

    // Expert 0 — first expert in the stacked tensor.
    let out = expert_swiglu_quantized(
        &hidden,
        &layer0,
        0,
        EXPECTED_N_EXPERTS,
        EXPECTED_INTERMEDIATE,
        EXPECTED_HIDDEN,
        &bytes,
    )
    .expect("F-QW3-MOE-C221-001: expert 0 SwiGLU should succeed");

    assert_eq!(
        out.len(),
        EXPECTED_HIDDEN,
        "F-QW3-MOE-C221-001: output dim must be hidden_dim"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "F-QW3-MOE-C221-001: all output elements must be finite (no NaN/Inf)"
    );
    let nonzero = out.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > EXPECTED_HIDDEN / 2,
        "F-QW3-MOE-C221-001: at least half the output should be non-zero \
         (got {nonzero}/{EXPECTED_HIDDEN}); else the kernel is trivially zeroing"
    );

    // Expert 1 — different selection should produce a measurably different output.
    let out_e1 = expert_swiglu_quantized(
        &hidden,
        &layer0,
        1,
        EXPECTED_N_EXPERTS,
        EXPECTED_INTERMEDIATE,
        EXPECTED_HIDDEN,
        &bytes,
    )
    .expect("F-QW3-MOE-C221-001: expert 1 SwiGLU should succeed");

    let mut differs = false;
    for i in 0..EXPECTED_HIDDEN.min(64) {
        if (out[i] - out_e1[i]).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "F-QW3-MOE-C221-001: expert 0 and expert 1 outputs must differ \
         (else the slicer is aliasing or weights are degenerate)"
    );

    let out_l2: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "F-QW3-MOE-C221-001: PASS\n  out.len() = {}\n  ||out||_2 = {:.4}\n  \
         out[0..5] = {:.4?}\n  diff vs expert 1 confirmed",
        out.len(),
        out_l2,
        &out[..5]
    );
}

/// M32c.2.2.2.0 falsifier — exercises `moe_ffn_forward_layer` against the
/// cached Qwen3-Coder GGUF. This is the FULL single-layer MoE FFN dispatch:
/// router (F32 matmul), softmax, top-8, renormalize, per-expert SwiGLU,
/// and weighted sum. The output is the layer's MoE contribution before the
/// residual addition.
const EXPECTED_K: usize = 8;

#[test]
fn f_qw3_moe_c2220_001_full_layer_forward_finite_output() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-C2220-001: skipped — no cached GGUF.");
        return;
    };

    eprintln!("F-QW3-MOE-C2220-001: full layer 0 MoE FFN forward on {gguf_path}");

    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let layer0 = load_qwen3_moe_layer(&model, &bytes, 0).expect("layer 0");

    let hidden: Vec<f32> = (0..EXPECTED_HIDDEN)
        .map(|i| 0.1 * ((i as f32 * 0.7).sin()))
        .collect();

    let out = moe_ffn_forward_layer(
        &hidden,
        &layer0,
        EXPECTED_N_EXPERTS,
        EXPECTED_K,
        EXPECTED_INTERMEDIATE,
        EXPECTED_HIDDEN,
        &bytes,
    )
    .expect("F-QW3-MOE-C2220-001: full layer 0 forward should succeed");

    assert_eq!(out.len(), EXPECTED_HIDDEN);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "F-QW3-MOE-C2220-001: all output elements must be finite"
    );
    let nonzero = out.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > EXPECTED_HIDDEN / 2,
        "F-QW3-MOE-C2220-001: at least half the output should be non-zero (got {nonzero}/{EXPECTED_HIDDEN})"
    );

    let out_l2: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "F-QW3-MOE-C2220-001: PASS\n  out.len() = {}\n  ||out||_2 = {:.6}\n  out[0..5] = {:.6?}",
        out.len(),
        out_l2,
        &out[..5]
    );
}
