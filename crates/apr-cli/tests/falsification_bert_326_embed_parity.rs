//! FALSIFY-BERT-326-EMBED-PARITY — `apr embed` cosine vs HF reference.
//!
//! Phase 8 of #326. The Phase 4b/4c/4d falsifiers cover `apr rerank`
//! (cross-encoder); this file covers `apr embed` (bi-encoder /
//! sentence-transformers). Together they lock both halves of the RAG
//! retrieve+rerank pipeline against the HuggingFace reference.
//!
//! ## What this asserts
//!
//! For each `(text_a, text_b, expected_cosine)` triple, compute
//! `cos(apr_embed(text_a), apr_embed(text_b))` and assert the result
//! matches the captured `expected_cosine` from
//! `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
//!   .encode(..., normalize_embeddings=True)` to within `COS_TOL`.
//!
//! ## Empirical (lambda-vector RTX 4090, 2026-05-18, post-Phase 6b)
//!
//! | Pair | apr cos | HF cos | diff |
//! |---|---|---|---|
//! | France?/Paris.   | 0.8559  | 0.8561  | 2e-4 |
//! | France?/Berlin   | 0.3939  | 0.3943  | 4e-4 |
//! | France?/Cats     | -0.0744 | -0.0629 | 1e-2 |  ← largest residual
//! | ML/neural        | ~0.56   | 0.5696  | ~1e-2 |
//! | Rust prog/safety | ~0.21   | 0.2155  | ~5e-3 |
//! | identity         | 1.0     | 1.0     | 0    |
//!
//! Residual is larger than rerank-side parity because mean-pooling
//! the encoder hidden states (6 layers × 384 dim × variable seq_len)
//! amplifies the tokenization-edge cases where aprender's WordPiece
//! differs from HF's BertTokenizerFast (e.g. handling of "France?"
//! when the question mark is its own pre-token vs attached). Phase
//! 6b closed the bulk of the gap; the residual is HF-tokenizer-
//! specific edge cases that don't affect ranking.
//!
//! ## How to run
//!
//! Requires:
//! - `apr` binary built from this branch (or later) on PATH
//! - `~/.cache/pacha/models/acbf56fa5791c79b.safetensors` cached via
//!   `apr pull sentence-transformers/all-MiniLM-L6-v2`
//! - `uv` for the HF reference Python script (only needed to update
//!   the captured `EXPECTED_PAIRS`; the test itself doesn't shell out
//!   to Python)
//!
//! ```
//! cargo test --test falsification_bert_326_embed_parity \
//!     -- --ignored --nocapture
//! ```

use std::path::Path;
use std::process::Command;

const ALLMINILM_SAFETENSORS: &str = "/home/noah/.cache/pacha/models/acbf56fa5791c79b.safetensors";
const ALLMINILM_TOKENIZER: &str = "/home/noah/.cache/pacha/models/acbf56fa5791c79b.tokenizer.json";

/// Pairs captured from `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
/// .encode(..., normalize_embeddings=True)` via uv 2026-05-18.
///
/// `cos(a, b)` ∈ [-1, 1] where 1 is identical-direction and -1 is opposite.
const EXPECTED_PAIRS: &[(&str, &str, f32)] = &[
    (
        "what is the capital of France?",
        "Paris is the capital of France.",
        0.856070,
    ),
    (
        "what is the capital of France?",
        "Berlin is the capital of Germany",
        0.394253,
    ),
    (
        "what is the capital of France?",
        "Cats are mammals that purr",
        -0.062919,
    ),
    (
        "machine learning",
        "neural networks are a key ML technique",
        0.569601,
    ),
    (
        "Rust programming",
        "memory safety without garbage collection",
        0.215508,
    ),
    // Identity sanity — normalised encoder output should be unit-norm so
    // cos(x, x) = 1 to fp precision.
    ("hello world", "hello world", 1.000000),
];

/// Tolerance for absolute cosine difference. The mean-pooling +
/// L2-normalisation chain on encoder hidden states amplifies residual
/// WordPiece tokenization differences slightly; ~1e-2 covers the
/// observed worst case while still catching genuine regressions
/// (e.g. wrong pooling, wrong normalization, missing layer).
///
/// Future Phase 6c (full HF BertBasicTokenizer fidelity) is expected
/// to tighten this to ~1e-4.
const COS_TOL: f32 = 1.5e-2;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Shell out to `apr embed --text A --text B ... --vocab ... --json`
/// once per pair, parse the two `embedding` arrays, return cosine.
///
/// Done one-pair-per-call (instead of batching all pairs in one
/// command) so each test pair stands alone and isolates failures to a
/// specific text pair.
fn apr_embed_cosine(
    apr_path: &Path,
    text_a: &str,
    text_b: &str,
    tokenizer: &str,
) -> Result<f32, String> {
    let output = Command::new("apr")
        .args(["embed"])
        .arg(apr_path)
        .args([
            "--text", text_a, "--text", text_b, "--vocab", tokenizer, "--pool", "mean", "--json",
        ])
        .output()
        .map_err(|e| format!("spawn apr embed: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "apr embed failed for ({text_a:?}, {text_b:?}); stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = std::str::from_utf8(&output.stdout).map_err(|e| format!("stdout utf8: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(stdout).map_err(|e| format!("json parse: {e}"))?;
    let results = v
        .get("results")
        .and_then(|r| r.as_array())
        .ok_or("results[] missing")?;
    if results.len() != 2 {
        return Err(format!("expected 2 results, got {}", results.len()));
    }
    let extract = |i: usize| -> Result<Vec<f32>, String> {
        results[i]
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or("embedding missing")
            .map_err(String::from)
            .and_then(|arr| {
                arr.iter()
                    .map(|x| x.as_f64().map(|f| f as f32).ok_or("non-float".to_string()))
                    .collect()
            })
    };
    let ea = extract(0)?;
    let eb = extract(1)?;
    if ea.len() != eb.len() {
        return Err(format!(
            "embedding dim mismatch: {} vs {}",
            ea.len(),
            eb.len()
        ));
    }
    // Unit-norm embeddings (normalize default ON) → cos = dot product.
    Ok(dot(&ea, &eb))
}

#[test]
#[ignore = "requires cached all-MiniLM SafeTensors + apr binary; ~30s"]
fn falsify_bert_326_phase8_embed_hf_parity() {
    if !Path::new(ALLMINILM_SAFETENSORS).exists() {
        eprintln!(
            "FALSIFY-BERT-326-EMBED: skipped — no cached all-MiniLM at {ALLMINILM_SAFETENSORS}.\n\
             Run `apr pull sentence-transformers/all-MiniLM-L6-v2` first."
        );
        return;
    }
    if !Path::new(ALLMINILM_TOKENIZER).exists() {
        eprintln!(
            "FALSIFY-BERT-326-EMBED: skipped — no cached tokenizer.json at {ALLMINILM_TOKENIZER}"
        );
        return;
    }

    // Build a fresh .apr from the cached SafeTensors so the test
    // catches drift in `apr import` too.
    let apr_out = std::env::temp_dir().join("falsify-bert-326-embed-parity.apr");
    let import_status = Command::new("apr")
        .args([
            "import",
            ALLMINILM_SAFETENSORS,
            "--arch",
            "bert",
            "--allow-no-config",
            "-o",
        ])
        .arg(&apr_out)
        .status()
        .expect("spawn apr import");
    assert!(
        import_status.success(),
        "apr import --arch bert must succeed on all-MiniLM-L6-v2"
    );

    let mut failures: Vec<String> = Vec::new();
    for (a, b, expected) in EXPECTED_PAIRS {
        let cos = apr_embed_cosine(&apr_out, a, b, ALLMINILM_TOKENIZER)
            .unwrap_or_else(|e| panic!("apr embed cosine failed for ({a:?}, {b:?}): {e}"));
        let diff = (cos - expected).abs();
        eprintln!(
            "FALSIFY-BERT-326-EMBED: ({a:?}, {b:?}) apr={cos:+.6} hf={expected:+.6} \
             diff={diff:.6e}{}",
            if diff < COS_TOL { "" } else { "  ← FAIL" }
        );
        if diff >= COS_TOL {
            failures.push(format!(
                "({a:?}, {b:?}): apr={cos:+.6} hf={expected:+.6} diff={diff:.6e}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "FALSIFY-BERT-326-EMBED: {} pair(s) failed HF cosine parity:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
