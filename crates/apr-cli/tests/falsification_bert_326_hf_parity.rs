//! FALSIFY-BERT-326-PHASE4B-HF-PARITY — end-to-end HF parity gate.
//!
//! Verifies that `apr rerank` produces the SAME sigmoid-mapped relevance
//! score as the HuggingFace reference `AutoModelForSequenceClassification`
//! for `cross-encoder/ms-marco-MiniLM-L-6-v2` on a fixed set of canonical
//! (query, passage) pairs.
//!
//! ## Empirically established on lambda-vector RTX 4090 (2026-05-17)
//!
//! | Pair | HF score | apr score | input_ids match |
//! |---|---|---|---|
//! | "France" + Paris  | 0.999805 | 0.999805 | ✅ exact |
//! | "France" + Cats   | 0.000015 | 0.000015 | ✅ exact |
//! | ML + neural       | 0.000020 | 0.000020 | ✅ exact |
//!
//! Raw logits agree within ~4e-4 (f32 round-off). Sigmoid maps both to
//! identical 6-decimal scores. WordPiece tokenization is bit-identical to
//! HF BertTokenizer.
//!
//! ## How to run
//!
//! Requires lambda-vector or any host with:
//! - `apr` binary (built from this branch)
//! - `~/.cache/pacha/models/57e6e922118ea840.safetensors` cached via
//!   `apr pull cross-encoder/ms-marco-MiniLM-L-6-v2`
//! - `uv` for the HF reference Python script
//!
//! ```
//! cargo test --test falsification_bert_326_hf_parity \
//!     -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` because it pulls in 87 MB of cached fixtures and invokes
//! `uv run --with transformers --with torch python3` which downloads
//! ~3 GB of pip wheels into the uv cache on first run.

use std::path::Path;
use std::process::Command;

/// A model fixture cached on lambda-vector by `apr pull` (GH-326 Phase 4c).
struct ModelFixture {
    /// Display name for error messages.
    name: &'static str,
    /// Cached SafeTensors path.
    safetensors: &'static str,
    /// Cached HF Tokenizers JSON path.
    tokenizer: &'static str,
    /// Number of BERT layers; passed to `apr rerank --num-layers`.
    num_layers: usize,
    /// `(query, passage, expected_hf_score)` triples captured from
    /// HuggingFace `AutoModelForSequenceClassification`. Aprender must
    /// reproduce these to within `SCORE_TOL`.
    pairs: &'static [(&'static str, &'static str, f32)],
}

const MINILM_L6: ModelFixture = ModelFixture {
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    safetensors: "/home/noah/.cache/pacha/models/57e6e922118ea840.safetensors",
    tokenizer: "/home/noah/.cache/pacha/models/57e6e922118ea840.tokenizer.json",
    num_layers: 6,
    pairs: &[
        (
            "what is the capital of France",
            "Paris is the capital of France",
            0.999805,
        ),
        (
            "what is the capital of France",
            "Cats are mammals that purr",
            0.000015,
        ),
        (
            "machine learning",
            "neural networks are a key ML technique",
            0.000020,
        ),
    ],
};

const MINILM_L12: ModelFixture = ModelFixture {
    name: "cross-encoder/ms-marco-MiniLM-L-12-v2",
    safetensors: "/home/noah/.cache/pacha/models/12445d2fa5ea239d.safetensors",
    tokenizer: "/home/noah/.cache/pacha/models/12445d2fa5ea239d.tokenizer.json",
    num_layers: 12,
    pairs: &[
        (
            "what is the capital of France",
            "Paris is the capital of France",
            0.999919,
        ),
        (
            "what is the capital of France",
            "Berlin is the capital of Germany",
            0.058924,
        ),
        (
            "what is the capital of France",
            "Cats are mammals that purr",
            0.000014,
        ),
    ],
};

/// Tolerance for absolute score difference (`apr` − HF reference). The
/// observed gap is < 5e-5 (sigmoid is monotonic + saturating, so the
/// ~4e-4 raw-logit drift compresses to f32 round-off at the score level).
/// 1e-4 is a generous bound that catches genuine numerical drift but
/// tolerates the existing raw-logit gap.
const SCORE_TOL: f32 = 1e-4;

fn extract_score_from_json(stdout: &str) -> f32 {
    // Parse the JSON output's `scores[0]` field.
    let v: serde_json::Value = serde_json::from_str(stdout).expect("apr rerank JSON parse");
    v.get("scores")
        .and_then(|s| s.get(0))
        .and_then(|s| s.as_f64())
        .expect("scores[0] missing") as f32
}

/// Run one model fixture's HF parity check. Returns `Vec<failure_msg>`
/// for any pair whose score diff exceeds `SCORE_TOL`.
fn run_parity_check(fix: &ModelFixture) -> Vec<String> {
    if !Path::new(fix.safetensors).exists() {
        eprintln!(
            "FALSIFY-BERT-326: skipped {} — no cached SafeTensors at {}.\n\
             Run `apr pull {}` first.",
            fix.name, fix.safetensors, fix.name
        );
        return Vec::new();
    }
    if !Path::new(fix.tokenizer).exists() {
        eprintln!(
            "FALSIFY-BERT-326: skipped {} — no cached tokenizer.json at {}",
            fix.name, fix.tokenizer
        );
        return Vec::new();
    }

    let safe_name = fix
        .name
        .split('/')
        .next_back()
        .unwrap_or(fix.name)
        .replace('.', "-");
    let apr_out = std::env::temp_dir().join(format!("falsify-bert-326-{safe_name}.apr"));
    let import_status = Command::new("apr")
        .args([
            "import",
            fix.safetensors,
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
        "apr import --arch bert must succeed on {}",
        fix.name
    );

    let layers_arg = fix.num_layers.to_string();
    let mut failures: Vec<String> = Vec::new();
    for (q, p, expected) in fix.pairs {
        let output = Command::new("apr")
            .args(["rerank"])
            .arg(&apr_out)
            .args([
                "--query",
                q,
                "--passage",
                p,
                "--vocab",
                fix.tokenizer,
                "--num-layers",
                &layers_arg,
                "--json",
            ])
            .output()
            .expect("spawn apr rerank");
        assert!(
            output.status.success(),
            "apr rerank must succeed for ({q:?}, {p:?}) against {}; stderr:\n{}",
            fix.name,
            String::from_utf8_lossy(&output.stderr)
        );
        let score = extract_score_from_json(
            std::str::from_utf8(&output.stdout).expect("rerank output is UTF-8"),
        );
        let diff = (score - expected).abs();
        eprintln!(
            "FALSIFY-BERT-326: {} q={q:?} p={p:?} apr={score:.6} hf={expected:.6} \
             diff={diff:.6e}{}",
            fix.name,
            if diff < SCORE_TOL { "" } else { "  ← FAIL" }
        );
        if diff >= SCORE_TOL {
            failures.push(format!(
                "{} ({q:?}, {p:?}): apr={score:.6} hf={expected:.6} diff={diff:.6e}",
                fix.name
            ));
        }
    }
    failures
}

/// HF parity for 6-layer MiniLM cross-encoder.
#[test]
#[ignore = "requires cached MiniLM-L-6 SafeTensors + apr binary; takes ~10s"]
fn falsify_bert_326_phase4b_hf_parity_l6() {
    let failures = run_parity_check(&MINILM_L6);
    assert!(
        failures.is_empty(),
        "FALSIFY-BERT-326 L6: {} pair(s) failed HF parity:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// HF parity for 12-layer MiniLM cross-encoder (GH-326 Phase 4c).
/// Validates that the BERT pipeline generalises across depths — the
/// same loader + forward path numerically matches HF for both 6-layer
/// (Phase 4b) and 12-layer architectures.
#[test]
#[ignore = "requires cached MiniLM-L-12 SafeTensors + apr binary; takes ~20s"]
fn falsify_bert_326_phase4c_hf_parity_l12() {
    let failures = run_parity_check(&MINILM_L12);
    assert!(
        failures.is_empty(),
        "FALSIFY-BERT-326 L12: {} pair(s) failed HF parity:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
