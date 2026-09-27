//! #4005: non-ASCII tokenization equals `llama-tokenize` on every tokenizer path apr ships.
//!
//! #3726 (byte-level BPE encoded non-ASCII to id 0) was fixed, and nothing proved it on the
//! release head for EVERY path: GGUF byte-level BPE (Qwen2.5/3/3.5), GGUF SentencePiece
//! (TinyLlama), a `.safetensors` model's HF `tokenizer.json`, and a `.apr`'s embedded
//! tokenizer. The golden ids are `llama-tokenize` on the identical (or, for SafeTensors and
//! `.apr`, the source) GGUF, committed with its build and every file's sha256 in
//! `evidence/tokenizer-parity/nonascii-4005.json`.
//!
//! The host-bound test is `#[ignore]`d because it needs the model files; the release gate
//! `scripts/check_tokenizer_nonascii_parity.sh` runs it explicitly, where a missing model or
//! a sha256 mismatch is a FAIL, never a skip. `violations` is pure and tested here unignored.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Every way apr's ids fail the reference: a mismatch, and (#3726's signature) an id 0 on
/// input that holds no control character. Empty means equal.
fn violations(apr: &[u32], golden: &[u32]) -> Vec<String> {
    let mut v = Vec::new();
    if apr.contains(&0) {
        v.push(format!(
            "apr emitted id 0 on non-ASCII text (#3726): {apr:?}"
        ));
    }
    if apr != golden {
        let at = apr
            .iter()
            .zip(golden)
            .position(|(a, b)| a != b)
            .unwrap_or(apr.len().min(golden.len()));
        v.push(format!(
            "ids differ from llama-tokenize at index {at}: apr {apr:?} vs reference {golden:?}"
        ));
    }
    v
}

fn golden() -> serde_json::Value {
    let p = std::env::var("APR_TOKPARITY_GOLDEN").unwrap_or_else(|_| {
        format!(
            "{}/../../evidence/tokenizer-parity/nonascii-4005.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("golden {p}: {e}")))
        .expect("golden JSON")
}

fn model_dirs() -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    std::env::var("APR_TOKPARITY_MODEL_DIRS")
        .unwrap_or_else(|_| format!("{home}/models:{home}/.apr/models:{home}/.cache/apr/models"))
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn sha256(p: &Path) -> std::io::Result<String> {
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(p)?;
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

/// apr's PRODUCTION encoders, the calls `apr run` makes: GGUF through the mapped model's own
/// tokenizer (infer/mod.rs), SafeTensors and `.apr` through `AprV2Model::encode_text`.
fn apr_encode(kind: &str, path: &Path, text: &str) -> Result<Vec<u32>, String> {
    match kind {
        "gguf_bpe" | "gguf_spm" => realizar::gguf::MappedGGUFModel::from_path(path)
            .map_err(|e| format!("map: {e}"))?
            .model
            .encode(text)
            .ok_or_else(|| "the GGUF tokenizer returned no ids".to_string()),
        "safetensors_hf" | "apr_embedded" => realizar::apr::AprV2Model::encode_text(path, text)
            .ok_or_else(|| "encode_text returned no ids".to_string()),
        other => Err(format!("unknown tokenizer path {other:?}")),
    }
}

#[test]
fn violations_names_a_mismatch_and_an_id_zero() {
    assert!(violations(&[1, 2, 3], &[1, 2, 3]).is_empty());
    let v = violations(&[0, 2, 3], &[1, 2, 3]);
    assert!(v.iter().any(|m| m.contains("id 0")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("at index 0")), "{v:?}");
    let short = violations(&[1, 2], &[1, 2, 3]);
    assert!(short.iter().any(|m| m.contains("at index 2")), "{short:?}");
}

#[test]
#[ignore = "release gate: needs the model files; run by scripts/check_tokenizer_nonascii_parity.sh (#4005)"]
fn nonascii_ids_equal_llama_tokenize_on_every_path() {
    let g = golden();
    let text = g["text"].as_str().expect("text");
    let cases = g["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 6,
        "the golden covers {} case(s); every shipped path is owed",
        cases.len()
    );
    let dirs = model_dirs();
    let mut fails = Vec::new();
    for c in cases {
        let (kind, subject) = (
            c["path"].as_str().unwrap_or(""),
            c["subject"].as_str().unwrap_or(""),
        );
        let label = format!("{} {kind} {subject}", c["family"].as_str().unwrap_or("?"));
        let want: Vec<u32> = c["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect();
        let Some(path) = dirs.iter().map(|d| d.join(subject)).find(|p| p.exists()) else {
            fails.push(format!(
                "{label}: NOT HELD under {dirs:?} -- unmeasured is not passed"
            ));
            continue;
        };
        let hashed = if kind == "safetensors_hf" {
            path.with_file_name("tokenizer.json")
        } else {
            path.clone()
        };
        match sha256(&hashed) {
            Ok(h) if Some(h.as_str()) == c["subject_sha256"].as_str() => {},
            Ok(h) => {
                fails.push(format!("{label}: sha256 {h} is not the golden's -- a different file is a different measurement"));
                continue;
            },
            Err(e) => {
                fails.push(format!("{label}: cannot hash {}: {e}", hashed.display()));
                continue;
            },
        }
        match apr_encode(kind, &path, text) {
            Ok(ids) => fails.extend(
                violations(&ids, &want)
                    .into_iter()
                    .map(|m| format!("{label}: {m}")),
            ),
            Err(e) => fails.push(format!("{label}: apr could not encode: {e}")),
        }
        eprintln!("measured {label}");
    }
    assert!(
        fails.is_empty(),
        "#4005 non-ASCII parity FAILED:\n  {}",
        fails.join("\n  ")
    );
}
