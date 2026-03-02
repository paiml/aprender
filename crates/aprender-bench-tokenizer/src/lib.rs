//! Shared utilities for aprender-bench-tokenizer benchmarks.
//!
//! Provides tokenizer loaders for both aprender and HuggingFace implementations,
//! a graceful-skip macro for environments without `tokenizer.json`, and
//! canonical payloads for head-to-head comparison.
//!
//! # Tokenizer resolution
//!
//! 1. `$TOKENIZER_JSON` environment variable (explicit override)
//! 2. `../../tokenizer.json` (repo root, Qwen2.5 151K vocab)
//! 3. `../../models/qwen3-4b/tokenizer.json` (model-specific)
//!
//! # Performance targets (GH-378)
//!
//! | Operation | Payload | aprender | HF v0.22 | Speedup |
//! |-----------|---------|----------|----------|---------|
//! | Encode | 636-char code | 74us | 155us | 2.08x |
//! | Encode | 2000-char HumanEval | 297us | 546us | 1.84x |
//! | Decode | 200 tokens | 10us | 13us | 1.31x |
//! | Loading | 7MB from_file | 142ms | 204ms | 1.43x |
//! | Loading | 7MB from_json | 136ms | — | — |
//! | Roundtrip | 636-char code | 87us | 161us | 1.84x |
//! | Batch | 100 prompts | 3.3ms | 5.8ms | 1.76x |
//! | ChatML | 3-turn conversation | 37us | 89us | 2.42x |
//!
//! All tokenizer formats (Qwen2, Whisper, GPT-2, LLaMA) share the same
//! optimized `load_from_json` path — results apply to all vocab sizes.

pub mod payloads;

use std::path::PathBuf;

/// Resolve the path to `tokenizer.json` for benchmarks.
///
/// Checks (in order):
/// 1. `$TOKENIZER_JSON` env var
/// 2. `../../tokenizer.json` (repo root)
/// 3. `../../models/qwen3-4b/tokenizer.json`
pub fn find_tokenizer_json() -> Option<PathBuf> {
    // 1. Explicit env override
    if let Ok(path) = std::env::var("TOKENIZER_JSON") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // 2. Repo root (relative to crates/aprender-bench-tokenizer/)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().ok()?;

    let candidates = [
        repo_root.join("tokenizer.json"),
        repo_root.join("models/qwen3-4b/tokenizer.json"),
    ];

    candidates.into_iter().find(|p| p.exists())
}

/// Load the aprender BPE tokenizer from the resolved `tokenizer.json`.
pub fn load_aprender_tokenizer() -> Option<aprender::text::bpe::BpeTokenizer> {
    let path = find_tokenizer_json()?;
    aprender::text::bpe::BpeTokenizer::from_huggingface(&path).ok()
}

/// Load the HuggingFace tokenizer from the resolved `tokenizer.json`.
pub fn load_hf_tokenizer() -> Option<tokenizers::Tokenizer> {
    let path = find_tokenizer_json()?;
    tokenizers::Tokenizer::from_file(&path).ok()
}

/// Read `tokenizer.json` as raw bytes (for loading benchmarks).
pub fn read_tokenizer_json_bytes() -> Option<Vec<u8>> {
    let path = find_tokenizer_json()?;
    std::fs::read(path).ok()
}

/// Read `tokenizer.json` as a string (for `from_json` benchmarks).
pub fn read_tokenizer_json_string() -> Option<String> {
    let path = find_tokenizer_json()?;
    std::fs::read_to_string(path).ok()
}

/// Gracefully skip a benchmark group when `tokenizer.json` is not available.
///
/// Usage:
/// ```ignore
/// let Some(tokenizer) = load_aprender_tokenizer() else {
///     skip_no_tokenizer!("encode");
///     return;
/// };
/// ```
#[macro_export]
macro_rules! skip_no_tokenizer {
    ($name:expr) => {
        eprintln!(
            "SKIP: tokenizer.json not found — skipping {} benchmarks. \
             Set $TOKENIZER_JSON or place tokenizer.json in repo root.",
            $name
        );
    };
}
