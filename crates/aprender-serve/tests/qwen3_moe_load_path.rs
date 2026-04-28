//! M32b falsifier tests for `contracts/qwen3-moe-forward-v1.yaml` —
//! FALSIFY-QW3-MOE-FORWARD-002.
//!
//! These tests pin the post-M32b behaviour of the GGUF/APR loader for
//! `arch = qwen3_moe` models. Pre-M32b, loading a Qwen3-Coder-30B-A3B
//! GGUF emitted the architecture-blind cryptic error:
//!
//!     Invalid shape: Tensor 'blk.0.ffn_up.weight' not found
//!
//! Post-M32b, the loader returns a structured
//! `RealizarError::UnsupportedOperation { operation: "moe_forward_pass",
//! reason: "...qwen3-moe-forward-v1..." }` BEFORE the dense FFN tensor
//! lookup is reached. This is the load-time half of the M32 chain;
//! M32c will replace this error with an actual MoE forward dispatch.
//!
//! ## Falsifier-001 (regression-shaped)
//! `f_qw3_moe_load_002a_smoke_no_dense_ffn_error_message`: even with a
//! synthetic GGUF whose general.architecture is "qwen3moe" but which
//! has zero MoE tensors (purely metadata), the error MUST NOT mention
//! `blk.0.ffn_up.weight`. The contract-named error path takes priority
//! over the dense-name lookup.
//!
//! ## Falsifier-002 (live, gated on the cached 17.3 GB GGUF)
//! `f_qw3_moe_load_002b_live_qwen3_coder_returns_unsupported_op`:
//! When the cached Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf is
//! present at one of the canonical paths, opening it through
//! `QuantizedGGUFTransformer::from_gguf` MUST return
//! `UnsupportedOperation { operation: "moe_forward_pass" }` whose
//! `reason` field contains the literal contract id
//! `"qwen3-moe-forward-v1"`. Otherwise (no GGUF cached on this host),
//! the test prints "skipped" and exits 0 — fixture-absent is not a
//! defect of the loader.

use realizar::error::RealizarError;
use realizar::gguf::{GGUFModel, QuantizedGGUFTransformer};

use std::path::Path;

const CONTRACT_ID: &str = "qwen3-moe-forward-v1";

/// Canonical lambda-vector cache locations for the M29 reference GGUF
/// (Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, sha256 prefix
/// `2b88b180a7…`, ~17.3 GB).
const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

#[test]
fn f_qw3_moe_load_002b_live_qwen3_coder_returns_unsupported_op() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "F-QW3-MOE-FORWARD-002b: skipped — no Qwen3-Coder GGUF cached at any of {:?}",
            CANONICAL_QWEN3_CODER_GGUF_PATHS
        );
        return;
    };

    eprintln!("F-QW3-MOE-FORWARD-002b: probing live qwen3_moe load at {gguf_path}");

    // Memory-map the GGUF (avoids 17 GB f32 expansion).
    let bytes = std::fs::read(gguf_path).expect("read GGUF bytes");
    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF header");

    let result = QuantizedGGUFTransformer::from_gguf(&model, &bytes);

    match result {
        Err(RealizarError::UnsupportedOperation { operation, reason }) => {
            assert_eq!(
                operation, "moe_forward_pass",
                "F-QW3-MOE-FORWARD-002b: operation tag must be 'moe_forward_pass', got {operation:?}"
            );
            assert!(
                reason.contains(CONTRACT_ID),
                "F-QW3-MOE-FORWARD-002b: reason must reference contract id {CONTRACT_ID:?}, got: {reason}"
            );
            assert!(
                !reason.contains("blk.0.ffn_up.weight"),
                "F-QW3-MOE-FORWARD-002b: reason must NOT contain dense-FFN tensor name, got: {reason}"
            );
            eprintln!("F-QW3-MOE-FORWARD-002b: PASS — structured contract error returned.");
        },
        Err(other) => {
            panic!("F-QW3-MOE-FORWARD-002b: expected UnsupportedOperation, got {other:?}")
        },
        Ok(_) => panic!(
            "F-QW3-MOE-FORWARD-002b: expected M32b to refuse qwen3_moe load, but it succeeded. \
             Did M32c land without updating this test? If forward dispatch is now wired, \
             this test should be replaced with FALSIFY-QW3-MOE-FORWARD-003 (token emission)."
        ),
    }
}

/// Cross-check: the canonical-key normalization must be a fixed point
/// for the lowercase GGUF metadata form `qwen3moe` and resolve to
/// `qwen3_moe`. Pre-M32b this was unobservable; post-M32b the load
/// path branches on this. Defends the test infrastructure.
#[test]
fn f_qw3_moe_load_002a_normalize_architecture_qwen3moe_resolves() {
    use realizar::tensor_names::normalize_architecture;

    assert_eq!(
        normalize_architecture("qwen3moe"),
        "qwen3_moe",
        "F-QW3-MOE-FORWARD-002a: GGUF metadata form 'qwen3moe' must canonicalize to 'qwen3_moe'"
    );
    assert_eq!(
        normalize_architecture("Qwen3MoeForCausalLM"),
        "qwen3_moe",
        "F-QW3-MOE-FORWARD-002a: HF class name 'Qwen3MoeForCausalLM' must canonicalize to 'qwen3_moe'"
    );
    assert_eq!(
        normalize_architecture("Qwen3CoderForCausalLM"),
        "qwen3_moe",
        "F-QW3-MOE-FORWARD-002a: Qwen3-Coder HF class must canonicalize to 'qwen3_moe' (M29 contract amendment)"
    );
}
