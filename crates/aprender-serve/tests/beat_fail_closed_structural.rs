//! BEAT-FAIL-CLOSED-STRUCT — Pillar-4 STRUCTURAL correctness beat (PMAT-756).
//!
//! Distinct from the SEMANTIC fail-closed beat (PMAT-744, all-zero/NaN/Inf/
//! extreme-magnitude tensor CONTENTS): this beat is about CROSS-TENSOR DIMENSION
//! INVARIANTS that a real transformer ALWAYS satisfies but that the SafeTensors
//! container format does NOT enforce.
//!
//! The SafeTensors format validates each tensor's shape<->byte-length in
//! isolation — it has NO model-level semantics. The official `safetensors`
//! library (used by HuggingFace Transformers and Ollama's safetensors import)
//! therefore LOADS, with ZERO error, a model whose embedding declares vocab=10
//! rows but whose lm_head declares vocab=8 rows (the two MUST index the same
//! vocabulary), or whose embedding hidden_dim=4 but whose q_proj input dim=6
//! (attention MUST consume the embedding's hidden vector). Such a model then
//! produces out-of-range token lookups / a dimension-mismatched first matmul ->
//! garbage or OOB at inference.
//!
//! INCUMBENT EVIDENCE (measured 2026-06-15, same host):
//!   `uv run --with safetensors --with numpy` -> `safetensors.numpy.load_file`
//!   LOADS both crafted-broken artifacts above with NO error
//!   (-> {'embed': [10,4], 'lm_head': [8,4]} ; {'embed':[10,4],'q_proj':[6,6]}).
//!   (NB: the same `safetensors` lib DOES reject a single-tensor shape<->byte
//!   inconsistency — so this beat is specifically the CROSS-TENSOR class the
//!   format leaves unchecked, and llama.cpp's GGUF arch-dim checks don't cover
//!   for a raw safetensors load.)
//!
//! apr's `validate_cross_tensor_structure` (F-STRUCT-001) REJECTS both, and
//! ACCEPTS a real, consistent model (no false positive). This file is the
//! CI-gated, falsifiable form of that guarantee.
//!
//! Contract: contracts/apr-fail-closed-structural-beat-v1.yaml.

use realizar::safetensors::validation::validate_cross_tensor_structure;

/// A named class of model = a list of `(tensor_name, shape)` pairs.
type ModelClass = (&'static str, Vec<(&'static str, Vec<usize>)>);

/// Build a `(name, shape)` view the gate consumes, mirroring the safetensors
/// tensor-info map (name -> shape) that the parser produces.
fn view<'a>(pairs: &'a [(&'a str, Vec<usize>)]) -> Vec<(&'a str, &'a [usize])> {
    pairs.iter().map(|(n, s)| (*n, s.as_slice())).collect()
}

/// A real, consistent (untied) model: embed [vocab, hidden], lm_head [vocab,
/// hidden], q_proj [q_out, hidden]. MUST PASS (no false positive).
fn healthy_model() -> Vec<(&'static str, Vec<usize>)> {
    let vocab = 32;
    let hidden = 16;
    vec![
        ("model.embed_tokens.weight", vec![vocab, hidden]),
        ("lm_head.weight", vec![vocab, hidden]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![hidden, hidden],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![hidden, hidden],
        ),
        ("model.norm.weight", vec![hidden]),
    ]
}

#[test]
fn beat_apr_rejects_vocab_mismatch_fail_closed() {
    // embed vocab=32, lm_head vocab=24 — DISAGREE. safetensors-lib loads this.
    let pairs = vec![
        ("model.embed_tokens.weight", vec![32usize, 16]),
        ("lm_head.weight", vec![24usize, 16]),
        ("model.layers.0.self_attn.q_proj.weight", vec![16usize, 16]),
    ];
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_err(),
        "FAIL-CLOSED VIOLATION: apr ACCEPTED a vocab-size mismatch (embed 32 rows vs lm_head 24 \
         rows) — this is the structural garbage safetensors/Transformers/Ollama load silently."
    );
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("F-STRUCT-001"),
        "rule id must be present: {msg}"
    );
    assert!(
        msg.contains("Vocab"),
        "error must name the vocab invariant: {msg}"
    );
}

#[test]
fn beat_apr_rejects_hidden_dim_mismatch_fail_closed() {
    // embed hidden=16, q_proj input dim=24 — DISAGREE. safetensors-lib loads this.
    let pairs = vec![
        ("model.embed_tokens.weight", vec![32usize, 16]),
        ("lm_head.weight", vec![32usize, 16]),
        ("model.layers.0.self_attn.q_proj.weight", vec![24usize, 24]),
    ];
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_err(),
        "FAIL-CLOSED VIOLATION: apr ACCEPTED a hidden-dim mismatch (embed hidden 16 vs q_proj \
         input 24) — the first attention matmul would be dimension-mismatched (OOB/garbage)."
    );
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("F-STRUCT-001"),
        "rule id must be present: {msg}"
    );
    assert!(
        msg.contains("Hidden"),
        "error must name the hidden invariant: {msg}"
    );
}

#[test]
fn beat_apr_rejects_llama_naming_vocab_mismatch() {
    // Llama-style names (tok_embeddings / output / attention.wq) — vocab disagree.
    let pairs = vec![
        ("tok_embeddings.weight", vec![100usize, 8]),
        ("output.weight", vec![64usize, 8]),
        ("layers.0.attention.wq.weight", vec![8usize, 8]),
    ];
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_err(),
        "FAIL-CLOSED VIOLATION: apr ACCEPTED a Llama-named vocab mismatch (tok_embeddings 100 vs \
         output 64)."
    );
}

#[test]
fn beat_apr_accepts_healthy_model_no_false_positive() {
    // The dual obligation: fail-closed must NOT reject a valid model.
    let pairs = healthy_model();
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_ok(),
        "FALSE POSITIVE: apr rejected a structurally-consistent model — fail-closed must not block \
         valid models. err={:?}",
        r.err().map(|e| format!("{e}"))
    );
}

#[test]
fn beat_apr_accepts_tied_embedding_no_false_positive() {
    // Tied-embedding model: NO separate lm_head/output tensor. The vocab
    // invariant holds vacuously — must PASS, not be flagged.
    let pairs = vec![
        ("model.embed_tokens.weight", vec![32usize, 16]),
        ("model.layers.0.self_attn.q_proj.weight", vec![16usize, 16]),
        ("model.norm.weight", vec![16usize]),
    ];
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_ok(),
        "FALSE POSITIVE: apr rejected a valid tied-embedding model (no separate lm_head). err={:?}",
        r.err().map(|e| format!("{e}"))
    );
}

#[test]
fn beat_apr_accepts_unknown_naming_no_false_positive() {
    // A model whose tensors this gate cannot positively identify must PASS
    // (the gate makes no assertion it cannot ground).
    let pairs = vec![
        ("some.custom.weight", vec![10usize, 4]),
        ("another.weight", vec![4usize, 4]),
    ];
    let r = validate_cross_tensor_structure(view(&pairs));
    assert!(
        r.is_ok(),
        "FALSE POSITIVE: apr asserted an invariant on unrecognised tensor names. err={:?}",
        r.err().map(|e| format!("{e}"))
    );
}

/// Headline beat assertion: apr rejects ALL structural-mismatch classes the
/// incumbents load silently, AND accepts ALL valid classes (zero false positive).
#[test]
fn beat_struct_summary_apr_strict_incumbents_permissive() {
    // Broken classes apr MUST reject (incumbent safetensors-lib loads all).
    let broken: Vec<ModelClass> = vec![
        (
            "vocab_mismatch",
            vec![
                ("model.embed_tokens.weight", vec![32, 16]),
                ("lm_head.weight", vec![24, 16]),
            ],
        ),
        (
            "hidden_mismatch",
            vec![
                ("model.embed_tokens.weight", vec![32, 16]),
                ("model.layers.0.self_attn.q_proj.weight", vec![24, 24]),
            ],
        ),
    ];
    let mut rejected = 0;
    for (name, pairs) in &broken {
        let v: Vec<(&str, &[usize])> = pairs.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        assert!(
            validate_cross_tensor_structure(v).is_err(),
            "apr must reject broken structural class `{name}`"
        );
        rejected += 1;
    }

    // Valid classes apr MUST accept (no false positive).
    let valid: Vec<ModelClass> = vec![
        ("consistent_untied", healthy_model()),
        (
            "tied",
            vec![
                ("model.embed_tokens.weight", vec![32, 16]),
                ("model.layers.0.self_attn.q_proj.weight", vec![16, 16]),
            ],
        ),
    ];
    let mut accepted = 0;
    for (name, pairs) in &valid {
        let v: Vec<(&str, &[usize])> = pairs.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        assert!(
            validate_cross_tensor_structure(v).is_ok(),
            "apr must accept valid class `{name}` (no false positive)"
        );
        accepted += 1;
    }

    assert_eq!(rejected, broken.len(), "all broken classes rejected");
    assert_eq!(accepted, valid.len(), "all valid classes accepted");
    println!(
        "BEAT-FAIL-CLOSED-STRUCT: apr rejected {rejected}/{} structural-mismatch classes \
         (incumbent safetensors/Transformers/Ollama: 0) and accepted {accepted}/{} valid classes \
         (0 false positive).",
        broken.len(),
        valid.len()
    );
}
