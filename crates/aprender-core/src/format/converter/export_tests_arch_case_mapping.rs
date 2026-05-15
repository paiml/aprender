//! SPEC-SHIP-TWO-001 §81 P0-F — HF arch → GGUF lowercase case mapping.
//!
//! APR metadata uses HuggingFace transformers convention
//! (e.g. `"LlamaForCausalLM"`, `"Qwen2ForCausalLM"`).
//! GGUF / llama.cpp expects lowercase family names
//! (`"llama"`, `"qwen2"`).
//!
//! Empirical surfacing on §78's MODEL-2 checkpoint:
//! - `apr export --format gguf epoch-004.apr -o /tmp/e4.gguf` succeeded
//! - `llama-cli -m /tmp/e4.gguf` refused with
//!   `unknown model architecture: 'LlamaForCausalLM'`
//!
//! Fix surface: `gguf_export_config::normalize_arch_for_gguf` maps
//! HF-style → GGUF-style at the export boundary.

// `normalize_arch_for_gguf` lives in `gguf_export_config.rs` which is
// `include!()`'d via `apr_export_fn.rs` → `export.rs` → `converter/mod.rs`,
// so it is reachable as a sibling under `super` (the `converter` module).
use super::super::normalize_arch_for_gguf;

/// FALSIFY-EXPORT-GGUF-ARCH-001: LlamaForCausalLM must map to lowercase llama.
///
/// The load-bearing case from §81 P0-C — MODEL-2 epoch-004.apr embeds
/// `architecture="LlamaForCausalLM"` (PyTorch/HF style); GGUF readers
/// (llama.cpp, llm-cpp) require lowercase `"llama"`.
#[test]
fn falsify_export_gguf_arch_001_llama_hf_to_gguf_case() {
    assert_eq!(normalize_arch_for_gguf("LlamaForCausalLM"), "llama");
}

#[test]
fn falsify_export_gguf_arch_002_qwen2_hf_to_gguf_case() {
    assert_eq!(normalize_arch_for_gguf("Qwen2ForCausalLM"), "qwen2");
}

#[test]
fn falsify_export_gguf_arch_003_qwen3_hf_to_gguf_case() {
    assert_eq!(normalize_arch_for_gguf("Qwen3ForCausalLM"), "qwen3");
}

#[test]
fn falsify_export_gguf_arch_004_qwen_moe_variants() {
    assert_eq!(normalize_arch_for_gguf("Qwen2MoeForCausalLM"), "qwen2moe");
    assert_eq!(normalize_arch_for_gguf("Qwen3MoeForCausalLM"), "qwen3moe");
}

#[test]
fn falsify_export_gguf_arch_005_mistral_maps_to_llama() {
    // Mistral architectures use the llama family in GGUF — no separate "mistral".
    assert_eq!(normalize_arch_for_gguf("MistralForCausalLM"), "llama");
}

#[test]
fn falsify_export_gguf_arch_006_phi3_gpt2_bert() {
    assert_eq!(normalize_arch_for_gguf("Phi3ForCausalLM"), "phi3");
    assert_eq!(normalize_arch_for_gguf("GPT2LMHeadModel"), "gpt2");
    assert_eq!(normalize_arch_for_gguf("BertForMaskedLM"), "bert");
}

#[test]
fn falsify_export_gguf_arch_007_already_lowercase_passes_through() {
    // Idempotent: feeding in already-normalized GGUF names must not change them.
    assert_eq!(normalize_arch_for_gguf("llama"), "llama");
    assert_eq!(normalize_arch_for_gguf("qwen2"), "qwen2");
    assert_eq!(normalize_arch_for_gguf("phi3"), "phi3");
    assert_eq!(normalize_arch_for_gguf("unknown"), "unknown");
}

#[test]
fn falsify_export_gguf_arch_008_unknown_lowercase_fallback() {
    // Defensive: an unknown HF-style name lowercases to maintain debuggability
    // rather than crashing or silently producing wrong output.
    let result = normalize_arch_for_gguf("SomeNovelArchForCausalLM");
    assert_eq!(result, "somenovelarchforcausallm");
    // The result MUST be lowercase (GGUF convention) even for unknowns.
    assert_eq!(result, result.to_lowercase());
}

#[test]
fn falsify_export_gguf_arch_009_never_emits_hf_uppercase() {
    // Property: for the 6 known HF mappings, the output MUST NOT equal the input
    // (it must have been normalized).
    let hf_names = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "MistralForCausalLM",
        "Phi3ForCausalLM",
        "GPT2LMHeadModel",
        "BertForMaskedLM",
    ];
    for hf in &hf_names {
        let normalized = normalize_arch_for_gguf(hf);
        assert_ne!(
            normalized, *hf,
            "HF name {} was not normalized (still upper-cased)",
            hf
        );
        assert!(
            normalized.chars().all(|c: char| !c.is_uppercase()),
            "normalize_arch_for_gguf({}) → {} contains uppercase chars",
            hf,
            normalized
        );
    }
}
