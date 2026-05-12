// SHIP-TWO-001 — `tensor-names-v1` algorithm-level PARTIAL discharge
// for FALSIFY-TNAME-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/tensor-names-v1.yaml`.

// ===========================================================================
// Reference architecture name normalization (mirrors realizar's logic)
// ===========================================================================

#[must_use]
pub fn normalize_architecture(name: &str) -> &'static str {
    match name {
        "PhiForCausalLM" => "phi2",
        "Phi3ForCausalLM" => "phi",
        "LlamaForCausalLM" | "llama" => "llama",
        "Qwen2ForCausalLM" | "qwen2" | "qwen2.5" | "qwen" => "qwen2",
        "Qwen3ForCausalLM" | "qwen3" => "qwen3",
        "Qwen3MoeForCausalLM" | "qwen3-moe" | "qwen3moe" => "qwen3moe",
        "MistralForCausalLM" | "mistral" => "mistral",
        "GPT2LMHeadModel" | "gpt2" => "gpt2",
        "GPTNeoXForCausalLM" | "gpt_neox" | "gptneox" => "gpt_neox",
        "GemmaForCausalLM" | "gemma" => "gemma",
        "Gemma2ForCausalLM" | "gemma2" => "gemma2",
        _ => "llama", // FALSIFY-TNAME-005: unknown defaults to llama.
    }
}

#[must_use]
pub fn gpt2_global_names_for_embedding() -> Vec<&'static str> {
    vec!["wte.weight", "transformer.wte.weight"]
}

#[must_use]
pub fn fused_qkv_template_count_for_arch(arch: &str) -> (usize, usize) {
    // Returns (fused_count, separate_q_count) for the given arch.
    // GPT-2 / GPT-NeoX use fused; modern LLaMA-style use separate.
    match arch {
        "gpt2" | "gpt_neox" => (1, 0),
        _ => (0, 1),
    }
}

// ===========================================================================
// TNAME-001 — YAML/Rust parity: name lists are equal sets
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_yaml_rust_parity(yaml_names: &[String], rust_names: &[String]) -> Tname001Verdict {
    if yaml_names.is_empty() || rust_names.is_empty() { return Tname001Verdict::Fail; }
    let mut a = yaml_names.to_vec();
    let mut b = rust_names.to_vec();
    a.sort();
    b.sort();
    if a == b { Tname001Verdict::Pass } else { Tname001Verdict::Fail }
}

// ===========================================================================
// TNAME-002 — Architecture map completeness: every referenced arch is mapped
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_arch_map_complete(
    referenced: &[String],
    mapped: &[String],
) -> Tname002Verdict {
    if referenced.is_empty() { return Tname002Verdict::Fail; }
    for arch in referenced {
        if !mapped.iter().any(|m| m == arch) { return Tname002Verdict::Fail; }
    }
    Tname002Verdict::Pass
}

// ===========================================================================
// TNAME-003 — Required role has at least one fallback
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname003Verdict { Pass, Fail }

/// `required_with_fallback_count[i] = (is_required, fallback_count)`.
/// Pass iff every `(true, n)` has `n >= 1`.
#[must_use]
pub fn verdict_from_required_has_fallback(required_with_count: &[(bool, usize)]) -> Tname003Verdict {
    if required_with_count.is_empty() { return Tname003Verdict::Fail; }
    for &(is_required, count) in required_with_count {
        if is_required && count == 0 { return Tname003Verdict::Fail; }
    }
    Tname003Verdict::Pass
}

// ===========================================================================
// TNAME-004 — PhiForCausalLM → "phi2", Phi3ForCausalLM → "phi"
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_phi_normalization() -> Tname004Verdict {
    if normalize_architecture("PhiForCausalLM") != "phi2" { return Tname004Verdict::Fail; }
    if normalize_architecture("Phi3ForCausalLM") != "phi" { return Tname004Verdict::Fail; }
    Tname004Verdict::Pass
}

// ===========================================================================
// TNAME-005 — Unknown architecture → "llama"
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_unknown_arch_default(unknown_name: &str) -> Tname005Verdict {
    if normalize_architecture(unknown_name) == "llama" { Tname005Verdict::Pass } else { Tname005Verdict::Fail }
}

// ===========================================================================
// TNAME-006 — GPT-2 embedding names: ["wte.weight", "transformer.wte.weight"]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gpt2_embedding_names(observed: &[String]) -> Tname006Verdict {
    let canonical = gpt2_global_names_for_embedding();
    if observed.len() != canonical.len() { return Tname006Verdict::Fail; }
    for name in canonical {
        if !observed.iter().any(|o| o == name) { return Tname006Verdict::Fail; }
    }
    // No "model." prefix allowed.
    if observed.iter().any(|o| o.starts_with("model.")) { return Tname006Verdict::Fail; }
    Tname006Verdict::Pass
}

// ===========================================================================
// TNAME-007 — Fused QKV resolution: gpt2 fused non-empty, separate Q empty
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tname007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_fused_qkv_resolution(arch: &str) -> Tname007Verdict {
    let (fused, separate_q) = fused_qkv_template_count_for_arch(arch);
    match arch {
        "gpt2" | "gpt_neox" => {
            // Fused must be > 0; separate Q must be == 0.
            if fused > 0 && separate_q == 0 { Tname007Verdict::Pass } else { Tname007Verdict::Fail }
        }
        _ => {
            // Other archs: separate Q present, fused absent.
            if fused == 0 && separate_q > 0 { Tname007Verdict::Pass } else { Tname007Verdict::Fail }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> { items.iter().map(|i| i.to_string()).collect() }

    // Reference impl spot checks
    #[test] fn ref_phi_normalization() {
        assert_eq!(normalize_architecture("PhiForCausalLM"), "phi2");
        assert_eq!(normalize_architecture("Phi3ForCausalLM"), "phi");
    }
    #[test] fn ref_unknown_to_llama() {
        assert_eq!(normalize_architecture("FutureArch2027"), "llama");
        assert_eq!(normalize_architecture(""), "llama");
        assert_eq!(normalize_architecture("xyz"), "llama");
    }
    #[test] fn ref_qwen_aliases() {
        assert_eq!(normalize_architecture("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(normalize_architecture("qwen2.5"), "qwen2");
        assert_eq!(normalize_architecture("Qwen3ForCausalLM"), "qwen3");
    }

    // TNAME-001
    #[test] fn tname001_pass_match() {
        let yaml = s(&["a", "b", "c"]);
        let rust = s(&["c", "b", "a"]); // same set, different order
        assert_eq!(verdict_from_yaml_rust_parity(&yaml, &rust), Tname001Verdict::Pass);
    }
    #[test] fn tname001_fail_drift() {
        let yaml = s(&["a", "b"]);
        let rust = s(&["a", "c"]);
        assert_eq!(verdict_from_yaml_rust_parity(&yaml, &rust), Tname001Verdict::Fail);
    }
    #[test] fn tname001_fail_empty() {
        assert_eq!(verdict_from_yaml_rust_parity(&[], &s(&["a"])), Tname001Verdict::Fail);
    }

    // TNAME-002
    #[test] fn tname002_pass_complete() {
        let refd = s(&["llama", "qwen2"]);
        let mapped = s(&["llama", "qwen2", "phi"]);
        assert_eq!(verdict_from_arch_map_complete(&refd, &mapped), Tname002Verdict::Pass);
    }
    #[test] fn tname002_fail_missing() {
        let refd = s(&["llama", "newarch"]);
        let mapped = s(&["llama"]);
        assert_eq!(verdict_from_arch_map_complete(&refd, &mapped), Tname002Verdict::Fail);
    }

    // TNAME-003
    #[test] fn tname003_pass_all_required_have_fallback() {
        let pairs = vec![(true, 3), (false, 0), (true, 1)];
        assert_eq!(verdict_from_required_has_fallback(&pairs), Tname003Verdict::Pass);
    }
    #[test] fn tname003_fail_required_no_fallback() {
        let pairs = vec![(true, 3), (true, 0)];
        assert_eq!(verdict_from_required_has_fallback(&pairs), Tname003Verdict::Fail);
    }
    #[test] fn tname003_fail_empty() {
        assert_eq!(verdict_from_required_has_fallback(&[]), Tname003Verdict::Fail);
    }

    // TNAME-004
    #[test] fn tname004_pass() {
        assert_eq!(verdict_from_phi_normalization(), Tname004Verdict::Pass);
    }

    // TNAME-005
    #[test] fn tname005_pass_unknown() {
        assert_eq!(verdict_from_unknown_arch_default("FutureArch2027"), Tname005Verdict::Pass);
    }
    #[test] fn tname005_pass_empty() {
        assert_eq!(verdict_from_unknown_arch_default(""), Tname005Verdict::Pass);
    }
    #[test] fn tname005_fail_known_arch() {
        // qwen2 is a known arch — should NOT default to llama.
        assert_eq!(verdict_from_unknown_arch_default("qwen2"), Tname005Verdict::Fail);
    }

    // TNAME-006
    #[test] fn tname006_pass_canonical() {
        let names = s(&["wte.weight", "transformer.wte.weight"]);
        assert_eq!(verdict_from_gpt2_embedding_names(&names), Tname006Verdict::Pass);
    }
    #[test] fn tname006_pass_reordered() {
        let names = s(&["transformer.wte.weight", "wte.weight"]);
        assert_eq!(verdict_from_gpt2_embedding_names(&names), Tname006Verdict::Pass);
    }
    #[test] fn tname006_fail_with_model_prefix() {
        let names = s(&["model.embed_tokens.weight", "wte.weight"]);
        assert_eq!(verdict_from_gpt2_embedding_names(&names), Tname006Verdict::Fail);
    }
    #[test] fn tname006_fail_short_list() {
        let names = s(&["wte.weight"]);
        assert_eq!(verdict_from_gpt2_embedding_names(&names), Tname006Verdict::Fail);
    }

    // TNAME-007
    #[test] fn tname007_pass_gpt2_fused() {
        assert_eq!(verdict_from_fused_qkv_resolution("gpt2"), Tname007Verdict::Pass);
    }
    #[test] fn tname007_pass_gpt_neox_fused() {
        assert_eq!(verdict_from_fused_qkv_resolution("gpt_neox"), Tname007Verdict::Pass);
    }
    #[test] fn tname007_pass_llama_separate() {
        assert_eq!(verdict_from_fused_qkv_resolution("llama"), Tname007Verdict::Pass);
    }
    #[test] fn tname007_pass_qwen2_separate() {
        assert_eq!(verdict_from_fused_qkv_resolution("qwen2"), Tname007Verdict::Pass);
    }

    // Reference helpers self-consistency
    #[test] fn fused_qkv_helper_canonical() {
        assert_eq!(fused_qkv_template_count_for_arch("gpt2"), (1, 0));
        assert_eq!(fused_qkv_template_count_for_arch("llama"), (0, 1));
        assert_eq!(fused_qkv_template_count_for_arch("qwen2"), (0, 1));
    }
}
