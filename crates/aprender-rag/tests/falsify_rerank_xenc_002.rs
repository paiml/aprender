//! FALSIFY-RERANK-XENC-002 — `aprender-rag::rerank` does not
//! contain a parallel inference stack. A future real cross-encoder
//! MUST route through `aprender-serve` via its HTTP/library API.
//!
//! Contract: `contracts/apr-rerank-v1.yaml` v1.3.0.
//!
//! This is a structural source-code gate (same shape as
//! `FALSIFY-AUTH-003` from HELIX-IDEA-009): runtime-equivalent gates
//! would have to wait until `aprender-serve`'s cross-encoder
//! routing actually exists, but the architectural rule itself is
//! locked in *now* by grepping the source for inference-crate
//! imports and forward-pass patterns. A drive-by refactor that
//! adds a real model into `rerank.rs` without routing through the
//! canonical inference path fails the gate at source level even
//! before the runtime test that exercises it runs.
//!
//! Today's `MockCrossEncoderReranker` uses a term-overlap proxy
//! (no real inference) and trivially complies.

#![allow(clippy::unwrap_used)]

const RERANK_SOURCE: &str = include_str!("../src/rerank.rs");

/// Inference-crate identifiers that, if imported by `rerank.rs`,
/// indicate a parallel inference stack. The gate fails if any of
/// these substrings appears in a `use ...;` line.
const BANNED_IMPORTS: &[&str] = &[
    "use realizar",
    "use candle_core",
    "use candle_nn",
    "use candle_transformers",
    "use tch",
    "use ort",
    "use onnxruntime",
    "use tract",
    "use burn::",
    // aprender-train is for training-time autograd, not for
    // inference at rerank time.
    "use entrenar",
];

/// Forward-pass / model-loading patterns that, if present in
/// `rerank.rs`, indicate inline inference. The gate fails on any
/// of these substrings unless we also see a routing-through-serve
/// marker (see ALLOWED_IF_ROUTED_VIA_SERVE).
const BANNED_PATTERNS: &[&str] =
    &["::from_pretrained(", ".forward(", "load_safetensors(", "load_gguf("];

#[test]
fn rerank_module_does_not_fork_inference_stack() {
    for banned in BANNED_IMPORTS {
        assert!(
            !RERANK_SOURCE.contains(banned),
            "FALSIFY-RERANK-XENC-002: rerank.rs imports {banned:?} — a future real \
             cross-encoder MUST route through aprender-serve, not pull in an \
             inference crate directly. If this is intentional, the architectural \
             rule has changed and the contract needs amending; do NOT just \
             whitelist the import."
        );
    }
}

#[test]
fn rerank_module_does_not_inline_forward_pass() {
    for banned in BANNED_PATTERNS {
        assert!(
            !RERANK_SOURCE.contains(banned),
            "FALSIFY-RERANK-XENC-002: rerank.rs contains forward-pass/model-loading \
             pattern {banned:?}. A real cross-encoder must route through \
             aprender-serve's API rather than running inference inline. The \
             current shipped MockCrossEncoderReranker uses term-overlap and \
             does NOT need this pattern."
        );
    }
}

#[test]
fn rerank_module_path_matches_contract_reference() {
    // If the file moves, this test stops compiling (`include_str!`
    // resolves at compile time) — that's by design. The contract's
    // references: list points at this exact path; a rename without
    // contract update would fail the workspace contract integration
    // test.
    assert!(!RERANK_SOURCE.is_empty(), "rerank.rs source must be non-empty");
    assert!(
        RERANK_SOURCE.contains("Reranker"),
        "rerank.rs must define the Reranker trait — anchoring this gate to the \
         module's actual contents catches accidental file moves"
    );
}

#[test]
fn mock_cross_encoder_uses_term_overlap_not_real_inference() {
    // Specific positive assertion: the shipped Mock implementation
    // computes scores via term-overlap (intersection of tokenised
    // query + doc). This is what makes today's rerank.rs trivially
    // compliant with the gate — no real model is loaded, so no
    // inference stack exists to fork.
    assert!(
        RERANK_SOURCE.contains("intersection"),
        "MockCrossEncoderReranker must score via set-intersection \
         (term overlap), not real inference. If a real cross-encoder \
         lands, route it through aprender-serve and update both the \
         contract description and this assertion."
    );
}
