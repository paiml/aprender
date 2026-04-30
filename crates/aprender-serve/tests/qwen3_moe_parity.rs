//! M32d.2 — `F-QW3-MOE-PARITY-001`: cosine-similarity parity gate
//! between APR's CPU `forward_qwen3_moe` and HuggingFace FP16 reference logits.
//!
//! Contract: [`contracts/qwen3-moe-forward-v1.yaml`] — `AC_QW3_MOE_005`
//! (formal: `cosine_similarity(apr_logits, hf_fp16_logits) > 0.99`).
//!
//! Falsifier: `FALSIFY-QW3-MOE-FORWARD-004` axis (a). Axis (b) (argmax vs
//! llama.cpp top-1) is the M32d.3 sibling test; landed separately.
//!
//! ## Heavy-test layout
//!
//! Both inputs are large and operator-confirm-gated:
//!
//! 1. The 17.3 GB `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` weights, mmap'd
//!    under `MappedGGUFModel::from_path`. Cached on lambda-vector at the
//!    paths listed in `CANONICAL_QWEN3_CODER_GGUF_PATHS` below.
//! 2. The `qwen3_moe_fp16_logits_pos0.json` fixture, generated once via
//!    `scripts/generate_qwen3_moe_fp16_logits.py` (M32d.1, PR #1129).
//!    This file is committed to `crates/aprender-serve/tests/fixtures/`
//!    after the operator runs the script on a host with disk + VRAM headroom.
//!
//! If either input is missing, the test prints a skip line and returns Ok.
//! Marked `#[ignore]` so it does NOT run in default CI; CI invokes it
//! explicitly via `cargo test ... -- --ignored` once both inputs are present
//! (per the FALSIFY-QW3-MOE-FORWARD-004 `test:` block in the contract).
//!
//! ## What the test does
//!
//! 1. Reads the JSON fixture into [`Fp16Fixture`]: vocab_size, position,
//!    input tokens, full 151936-dim logit vector, argmax token.
//! 2. Sanity-checks vocab_size matches the live model's vocab.
//! 3. Loads the GGUF, builds `OwnedQuantizedModel` + per-layer MoE
//!    descriptors (same path as `qwen3_moe_forward_one_token.rs`).
//! 4. Runs `forward_qwen3_moe(token_ids = fixture.tokens, ...)` once.
//!    The returned logits represent the next-token-after-prompt
//!    distribution at `seq_len-1`.
//! 5. Computes cosine similarity vs the FP16 reference.
//! 6. Asserts `cos_sim > 0.99` (per `AC_QW3_MOE_005`).

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

use std::path::Path;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const FIXTURE_PATH: &str = "crates/aprender-serve/tests/fixtures/qwen3_moe_fp16_logits_pos0.json";

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;
const EXPECTED_VOCAB: usize = 151936;

const COSINE_THRESHOLD: f32 = 0.99;

#[derive(serde::Deserialize)]
struct Fp16Fixture {
    #[serde(default)]
    model_name: String,
    #[serde(default)]
    prompt: String,
    tokens: Vec<u32>,
    vocab_size: usize,
    #[allow(dead_code)]
    position: usize,
    logits: Vec<f32>,
    argmax_token: u32,
    #[serde(default)]
    argmax_text: String,
}

fn load_fixture(path: &Path) -> Option<Fp16Fixture> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine_similarity: vectors must be same length"
    );
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn fixture_path() -> std::path::PathBuf {
    if let Ok(repo_root) = std::env::var("CARGO_MANIFEST_DIR") {
        let manifest = std::path::PathBuf::from(repo_root);
        let abs = manifest.join("tests/fixtures/qwen3_moe_fp16_logits_pos0.json");
        if abs.exists() {
            return abs;
        }
    }
    std::path::PathBuf::from(FIXTURE_PATH)
}

#[test]
#[ignore]
fn f_qw3_moe_parity_001_cosine_vs_hf_fp16() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "F-QW3-MOE-PARITY-001: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    let fx_path = fixture_path();
    let Some(fixture) = load_fixture(&fx_path) else {
        eprintln!(
            "F-QW3-MOE-PARITY-001: skipped — FP16 fixture not found at {} \
             (run scripts/generate_qwen3_moe_fp16_logits.py per M32d.1 to generate it)",
            fx_path.display()
        );
        return;
    };

    eprintln!("F-QW3-MOE-PARITY-001: cosine vs HF FP16");
    eprintln!("  gguf:    {gguf_path}");
    eprintln!(
        "  fixture: {} ({} bytes)",
        fx_path.display(),
        fixture.logits.len() * 4
    );
    eprintln!("  model:   {}", fixture.model_name);
    eprintln!("  prompt:  {:?}", fixture.prompt);
    eprintln!("  tokens:  {} ids", fixture.tokens.len());

    assert_eq!(
        fixture.vocab_size, EXPECTED_VOCAB,
        "fixture vocab_size must match canonical Qwen3-Coder vocab"
    );
    assert_eq!(
        fixture.logits.len(),
        EXPECTED_VOCAB,
        "fixture logits dimensionality must equal vocab_size"
    );
    assert!(
        !fixture.tokens.is_empty(),
        "fixture must contain at least one input token"
    );

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();
    let model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    eprintln!(
        "F-QW3-MOE-PARITY-001: running APR forward on {} prompt tokens (this takes a few minutes)...",
        fixture.tokens.len()
    );
    let start = std::time::Instant::now();
    let apr_logits = model
        .forward_qwen3_moe(
            &fixture.tokens,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("F-QW3-MOE-PARITY-001: forward should succeed");
    let elapsed = start.elapsed();

    assert_eq!(
        apr_logits.len(),
        EXPECTED_VOCAB,
        "APR logits len must equal vocab_size"
    );
    assert!(
        apr_logits.iter().all(|v| v.is_finite()),
        "all APR logits must be finite (no NaN/Inf)"
    );

    let cos = cosine_similarity(&apr_logits, &fixture.logits);

    let (apr_argmax, &apr_max_val) = apr_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("logits non-empty");
    let apr_argmax = apr_argmax as u32;

    eprintln!(
        "F-QW3-MOE-PARITY-001:\n  elapsed       = {elapsed:?}\n  cos_sim       = {cos:.6}\n  threshold     = {COSINE_THRESHOLD}\n  apr_argmax    = {apr_argmax} (val = {apr_max_val:.4})\n  hf_argmax     = {} ({:?})",
        fixture.argmax_token, fixture.argmax_text
    );

    assert!(
        cos > COSINE_THRESHOLD,
        "F-QW3-MOE-PARITY-001 (AC_QW3_MOE_005): \
         cosine_similarity(apr_logits, hf_fp16_logits) = {cos:.6} \
         is NOT > {COSINE_THRESHOLD}. Diagnostic per FALSIFY-QW3-MOE-FORWARD-004 if_fails: \
         numerical divergence in math itself; investigate layer-by-layer \
         (embedding → RMSNorm → QKV → RoPE → attn → MoE router → per-expert SwiGLU → lm_head)."
    );
}

#[test]
fn fixture_loader_handles_missing_path() {
    let result = load_fixture(Path::new(
        "/nonexistent/path/qwen3_moe_fp16_logits_pos0.json",
    ));
    assert!(
        result.is_none(),
        "missing fixture path must return None (not panic)"
    );
}

#[test]
fn cosine_similarity_unit_vectors() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![1.0_f32, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    let c = vec![1.0_f32, 0.0, 0.0];
    let d = vec![0.0_f32, 1.0, 0.0];
    assert!(cosine_similarity(&c, &d).abs() < 1e-6);

    let e = vec![1.0_f32, 0.0, 0.0];
    let f = vec![-1.0_f32, 0.0, 0.0];
    assert!((cosine_similarity(&e, &f) - (-1.0)).abs() < 1e-6);
}

#[test]
fn cosine_similarity_handles_zero_vector() {
    let a = vec![0.0_f32; 8];
    let b = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
    assert_eq!(cosine_similarity(&b, &a), 0.0);
}
