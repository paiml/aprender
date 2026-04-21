//! GGUF → Safetensors output-layout + metadata-translation classifier
//! for `apr convert --format safetensors` (CRUX-B-02).
//!
//! Contract: `contracts/crux-B-02-v1.yaml`.
//!
//! Three pure algorithm-level necessary conditions:
//!
//! 1. `hf_required_files()` is the canonical set of filenames a
//!    converted output directory must contain for HuggingFace
//!    `AutoModelForCausalLM.from_pretrained` to succeed. Without this
//!    exact set, FALSIFY-CRUX-B-02-001 fails at the file-layout layer
//!    before any byte-level load work begins.
//!
//! 2. `translate_gguf_metadata(kv)` is a pure function mapping the
//!    GGUF `llama.*` metadata keys onto HuggingFace `config.json`
//!    fields. If any required field is missing the function returns
//!    `Err(MissingKey)` rather than silently defaulting — a silent
//!    default would let `transformers.from_pretrained` succeed with
//!    the wrong hidden_size / layer count and produce garbage.
//!
//! 3. `peft_target_modules_resolve(tensor_names, target_modules)`
//!    checks that every PEFT target module name (e.g. `q_proj`,
//!    `v_proj`) resolves to at least one tensor in the safetensors
//!    archive. This is a necessary condition for FALSIFY-CRUX-B-02-004
//!    (`peft.get_peft_model` attaches without shape errors): if no
//!    tensor matches the target, the attach fails.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Canonical set of files a HuggingFace transformers-loadable
/// directory must contain.
pub fn hf_required_files() -> BTreeSet<&'static str> {
    ["model.safetensors", "config.json", "tokenizer.json"]
        .into_iter()
        .collect()
}

/// Check that a candidate directory listing contains every required
/// HuggingFace file. Returns the missing files (empty when complete).
pub fn missing_hf_files(listing: &BTreeSet<String>) -> BTreeSet<&'static str> {
    hf_required_files()
        .into_iter()
        .filter(|name| !listing.contains(*name))
        .collect()
}

/// HuggingFace `config.json` fields we translate from GGUF metadata.
/// Kept as an owned struct (not `serde_json::Value`) so the shape is
/// compile-time checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfLlamaConfig {
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
}

/// Error variants for `translate_gguf_metadata`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// A required GGUF key is missing; silent defaults would produce
    /// a loadable-but-wrong HF model.
    MissingKey(&'static str),
    /// The value was present but of an unexpected type.
    WrongType {
        key: &'static str,
        got: &'static str,
    },
}

/// Translate GGUF `llama.*` metadata onto HF `config.json` fields.
///
/// Required keys:
///   - `general.architecture` → `architectures[0]`
///   - `llama.embedding_length` → `hidden_size`
///   - `llama.block_count` → `num_hidden_layers`
///   - `llama.attention.head_count` → `num_attention_heads`
pub fn translate_gguf_metadata(
    kv: &BTreeMap<String, GgufValue>,
) -> Result<HfLlamaConfig, MetadataError> {
    let arch = take_string(kv, "general.architecture")?;
    let hidden_size = take_u32(kv, "llama.embedding_length")?;
    let num_hidden_layers = take_u32(kv, "llama.block_count")?;
    let num_attention_heads = take_u32(kv, "llama.attention.head_count")?;
    Ok(HfLlamaConfig {
        architectures: vec![format!("{}ForCausalLM", capitalize(&arch))],
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
    })
}

/// A tiny GGUF metadata value model — enough to prove the translator
/// is pure without pulling in the real gguf crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufValue {
    Str(String),
    U32(u32),
}

fn take_string(kv: &BTreeMap<String, GgufValue>, key: &'static str) -> Result<String, MetadataError> {
    match kv.get(key) {
        None => Err(MetadataError::MissingKey(key)),
        Some(GgufValue::Str(s)) => Ok(s.clone()),
        Some(GgufValue::U32(_)) => Err(MetadataError::WrongType { key, got: "u32" }),
    }
}

fn take_u32(kv: &BTreeMap<String, GgufValue>, key: &'static str) -> Result<u32, MetadataError> {
    match kv.get(key) {
        None => Err(MetadataError::MissingKey(key)),
        Some(GgufValue::U32(n)) => Ok(*n),
        Some(GgufValue::Str(_)) => Err(MetadataError::WrongType { key, got: "str" }),
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
    }
}

/// Outcome of the PEFT target-module-resolution pre-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeftResolution {
    /// Every target module resolves to at least one tensor.
    AllResolved,
    /// At least one target module has no matching tensor — PEFT
    /// attach would fail.
    Unresolved { missing: Vec<String> },
}

/// Check that every PEFT target module has at least one tensor whose
/// name contains the module string. HuggingFace PEFT uses
/// suffix-matching on module names like `q_proj`, so we mirror that
/// with a substring check — consistent with what PEFT itself does.
pub fn peft_target_modules_resolve(
    tensor_names: &[String],
    target_modules: &[&str],
) -> PeftResolution {
    let missing: Vec<String> = target_modules
        .iter()
        .filter(|m| !tensor_names.iter().any(|t| t.contains(*m)))
        .map(|m| m.to_string())
        .collect();
    if missing.is_empty() {
        PeftResolution::AllResolved
    } else {
        PeftResolution::Unresolved { missing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== hf_required_files / missing_hf_files =====

    #[test]
    fn required_files_set_is_canonical() {
        // FALSIFY-CRUX-B-02-001: the exact trio expected by HF.
        let files = hf_required_files();
        assert!(files.contains("model.safetensors"));
        assert!(files.contains("config.json"));
        assert!(files.contains("tokenizer.json"));
    }

    #[test]
    fn missing_files_empty_on_complete_output() {
        let listing: BTreeSet<String> = ["model.safetensors", "config.json", "tokenizer.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(missing_hf_files(&listing).is_empty());
    }

    #[test]
    fn missing_files_flags_each_omission() {
        let listing: BTreeSet<String> = ["config.json"].iter().map(|s| s.to_string()).collect();
        let missing = missing_hf_files(&listing);
        assert!(missing.contains("model.safetensors"));
        assert!(missing.contains("tokenizer.json"));
        assert!(!missing.contains("config.json"));
    }

    #[test]
    fn missing_files_flags_all_when_empty() {
        let listing: BTreeSet<String> = BTreeSet::new();
        let missing = missing_hf_files(&listing);
        assert_eq!(missing.len(), 3);
    }

    #[test]
    fn required_files_set_is_deterministic() {
        assert_eq!(hf_required_files(), hf_required_files());
    }

    // ===== translate_gguf_metadata =====

    fn minimal_kv() -> BTreeMap<String, GgufValue> {
        let mut kv = BTreeMap::new();
        kv.insert(
            "general.architecture".into(),
            GgufValue::Str("llama".into()),
        );
        kv.insert("llama.embedding_length".into(), GgufValue::U32(4096));
        kv.insert("llama.block_count".into(), GgufValue::U32(32));
        kv.insert("llama.attention.head_count".into(), GgufValue::U32(32));
        kv
    }

    #[test]
    fn translate_full_metadata_succeeds() {
        let cfg = translate_gguf_metadata(&minimal_kv()).unwrap();
        assert_eq!(cfg.architectures, vec!["LlamaForCausalLM".to_string()]);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn translate_missing_architecture_errors() {
        let mut kv = minimal_kv();
        kv.remove("general.architecture");
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap_err(),
            MetadataError::MissingKey("general.architecture")
        );
    }

    #[test]
    fn translate_missing_hidden_size_errors() {
        let mut kv = minimal_kv();
        kv.remove("llama.embedding_length");
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap_err(),
            MetadataError::MissingKey("llama.embedding_length")
        );
    }

    #[test]
    fn translate_missing_layer_count_errors() {
        let mut kv = minimal_kv();
        kv.remove("llama.block_count");
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap_err(),
            MetadataError::MissingKey("llama.block_count")
        );
    }

    #[test]
    fn translate_missing_head_count_errors() {
        let mut kv = minimal_kv();
        kv.remove("llama.attention.head_count");
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap_err(),
            MetadataError::MissingKey("llama.attention.head_count")
        );
    }

    #[test]
    fn translate_wrong_type_errors() {
        // If embedding_length is a string, refuse — silently parsing
        // "4096" would be convenient but would hide a upstream bug.
        let mut kv = minimal_kv();
        kv.insert("llama.embedding_length".into(), GgufValue::Str("x".into()));
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap_err(),
            MetadataError::WrongType {
                key: "llama.embedding_length",
                got: "str",
            }
        );
    }

    #[test]
    fn translate_is_deterministic() {
        let kv = minimal_kv();
        assert_eq!(
            translate_gguf_metadata(&kv).unwrap(),
            translate_gguf_metadata(&kv).unwrap()
        );
    }

    #[test]
    fn translate_config_roundtrips_through_json() {
        let cfg = translate_gguf_metadata(&minimal_kv()).unwrap();
        let s = serde_json::to_string(&cfg).unwrap();
        let parsed: HfLlamaConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(cfg, parsed);
    }

    // ===== peft_target_modules_resolve =====

    fn llama_tensor_names() -> Vec<String> {
        vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.k_proj.weight".into(),
            "model.layers.0.self_attn.v_proj.weight".into(),
            "model.layers.0.self_attn.o_proj.weight".into(),
        ]
    }

    #[test]
    fn peft_default_targets_resolve_on_llama_layout() {
        // FALSIFY-CRUX-B-02-004 necessary condition: q_proj / v_proj
        // both have matching tensors, so PEFT attach can succeed.
        let r = peft_target_modules_resolve(&llama_tensor_names(), &["q_proj", "v_proj"]);
        assert_eq!(r, PeftResolution::AllResolved);
    }

    #[test]
    fn peft_unknown_target_module_flagged() {
        let r = peft_target_modules_resolve(
            &llama_tensor_names(),
            &["q_proj", "nonexistent_proj"],
        );
        match r {
            PeftResolution::Unresolved { missing } => {
                assert_eq!(missing, vec!["nonexistent_proj".to_string()]);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn peft_empty_tensor_list_fails_all_targets() {
        let r = peft_target_modules_resolve(&[], &["q_proj", "v_proj"]);
        match r {
            PeftResolution::Unresolved { missing } => {
                assert_eq!(missing.len(), 2);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn peft_empty_target_list_trivially_resolves() {
        // Vacuous true: no targets required → nothing to resolve.
        let r = peft_target_modules_resolve(&llama_tensor_names(), &[]);
        assert_eq!(r, PeftResolution::AllResolved);
    }

    #[test]
    fn peft_resolution_is_deterministic() {
        let names = llama_tensor_names();
        let targets = &["q_proj", "v_proj"];
        assert_eq!(
            peft_target_modules_resolve(&names, targets),
            peft_target_modules_resolve(&names, targets)
        );
    }
}
