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

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
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
