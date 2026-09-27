//! #2815: `<|im_end|>` after sentence punctuation must encode as the single special id,
//! exactly as llama-tokenize does. Special tokens are partitioned out BEFORE the
//! pre-tokenizer runs (#3726/#3993), so `.` / `?` never glue onto `<|`.
//!
//! Host-bound: needs a Qwen2.5 GGUF. Goldens are llama-tokenize (build 11191,
//! commit 4b1a27fa0) `-p <text> --ids --no-bos`; identical for
//! qwen2.5-coder-0.5b-instruct-q4_k_m and qwen2.5-1.5b-instruct-q4_k_m.

use std::path::PathBuf;

const GOLDEN: &[(&str, &[u32])] = &[
    ("number<|im_end|>", &[4082, 151645]),
    ("a<|im_end|>", &[64, 151645]),
    ("<|im_end|>", &[151645]),
    ("number.<|im_end|>", &[4082, 13, 151645]),
    ("2?<|im_end|>", &[17, 30, 151645]),
    (".<|im_end|>", &[13, 151645]),
];

const MODELS: &[&str] = &[
    "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
];

fn model_dir() -> PathBuf {
    std::env::var_os("APR_TOKPARITY_MODEL_DIR").map_or_else(
        || PathBuf::from(std::env::var("HOME").expect("HOME")).join("models"),
        PathBuf::from,
    )
}

#[test]
#[ignore = "host-bound: needs Qwen2.5 GGUFs (APR_TOKPARITY_MODEL_DIR, default ~/models)"]
fn im_end_after_punctuation_matches_llama_tokenize() {
    let mut bad = Vec::new();
    for name in MODELS {
        let path = model_dir().join(name);
        assert!(path.is_file(), "{} NOT HELD", path.display());
        let mapped = realizar::gguf::MappedGGUFModel::from_path(&path).expect("load GGUF");
        for (text, want) in GOLDEN {
            let got = mapped.model.encode(text).expect("encode");
            if got != *want {
                bad.push(format!("{name}: {text:?} -> {got:?}, llama-tokenize {want:?}"));
            }
        }
    }
    assert!(bad.is_empty(), "#2815 parity FAILED:\n{}", bad.join("\n"));
}
