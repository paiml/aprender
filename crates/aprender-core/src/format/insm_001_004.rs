// SHIP-TWO-001 — `apr-inspect-metadata-propagation-v1` algorithm-level
// PARTIAL discharge for FALSIFY-INSPECT-META-001..004.
//
// Contract: `contracts/apr-inspect-metadata-propagation-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four `apr inspect` metadata-propagation gates:
//
// - INSPECT-META-001 (`apr inspect` KV count == `apr hex` KV count).
// - INSPECT-META-002 (`apr inspect` KV count ≥ 20 for Qwen2.5).
// - INSPECT-META-003 (KV keys use `arch.*` or `general.*` prefix —
//   not `n_embd`/`n_heads` shorthand).
// - INSPECT-META-004 (`validate_inspect.rs` does NOT contain
//   `meta_map.insert("n_(embd|heads|layers)"` — the buggy 4-key stub
//   pattern is gone).

/// Minimum KV count for a real Qwen2.5-Coder GGUF (architecture +
/// tokenizer + alignment + version + per-arch hyperparameters).
pub const AC_INSM_002_MIN_KV_FOR_QWEN25: usize = 20;

/// Architecture-prefix scopes that must precede the dotted key name.
pub const AC_INSM_003_VALID_PREFIXES: [&str; 3] = ["qwen2.", "general.", "llama."];

/// Forbidden shorthand keys (the 4-key hand-written stub).
pub const AC_INSM_003_FORBIDDEN_SHORTHAND: [&str; 4] = ["n_embd", "n_heads", "n_layers", "n_kv"];

/// Forbidden source pattern for INSPECT-META-004 (regression guard).
pub const AC_INSM_004_FORBIDDEN_PATTERN_PREFIX: &str = "meta_map.insert(\"n_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsmVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// True iff `key` starts with one of the valid architecture-scoped
/// prefixes (case-sensitive — GGUF keys are lowercase by convention).
#[must_use]
pub fn has_valid_arch_prefix(key: &str) -> bool {
    AC_INSM_003_VALID_PREFIXES
        .iter()
        .any(|p| key.starts_with(p))
}

/// True iff `key` is one of the forbidden shorthand keys.
#[must_use]
pub fn is_forbidden_shorthand(key: &str) -> bool {
    AC_INSM_003_FORBIDDEN_SHORTHAND.contains(&key)
}

// -----------------------------------------------------------------------------
// Verdict 1: INSPECT-META-001 — inspect == hex KV count.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_kv_count_parity(
    inspect_kv_count: usize,
    hex_kv_count: usize,
) -> InsmVerdict {
    if hex_kv_count == 0 {
        return InsmVerdict::Fail;
    }
    if inspect_kv_count == hex_kv_count {
        InsmVerdict::Pass
    } else {
        InsmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: INSPECT-META-002 — KV count ≥ 20 for Qwen2.5.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_kv_count_floor(kv_count: usize) -> InsmVerdict {
    if kv_count >= AC_INSM_002_MIN_KV_FOR_QWEN25 {
        InsmVerdict::Pass
    } else {
        InsmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: INSPECT-META-003 — keys are architecture-scoped.
// -----------------------------------------------------------------------------

/// Pass iff at least one key has a valid arch prefix AND no key is a
/// forbidden shorthand.
#[must_use]
pub fn verdict_from_arch_scoped_keys(keys: &[&str]) -> InsmVerdict {
    if keys.is_empty() {
        return InsmVerdict::Fail;
    }
    let mut found_valid = false;
    for &k in keys {
        if is_forbidden_shorthand(k) {
            return InsmVerdict::Fail;
        }
        if has_valid_arch_prefix(k) {
            found_valid = true;
        }
    }
    if found_valid {
        InsmVerdict::Pass
    } else {
        InsmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: INSPECT-META-004 — no hand-written 4-key stub in source.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_no_meta_map_stub(code_text: &str) -> InsmVerdict {
    // The regex equivalent: meta_map.insert("n_<embd|heads|layers>".
    // We do a substring scan for `meta_map.insert("n_` followed by
    // any of the 3 forbidden suffixes within the next ~10 chars.
    for forbidden in AC_INSM_003_FORBIDDEN_SHORTHAND {
        let pattern = format!("{AC_INSM_004_FORBIDDEN_PATTERN_PREFIX}{}", &forbidden[2..]);
        if code_text.contains(&pattern) {
            return InsmVerdict::Fail;
        }
    }
    InsmVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_kv_qwen25_is_20() {
        assert_eq!(AC_INSM_002_MIN_KV_FOR_QWEN25, 20);
    }

    #[test]
    fn provenance_valid_prefixes() {
        assert!(AC_INSM_003_VALID_PREFIXES.contains(&"qwen2."));
        assert!(AC_INSM_003_VALID_PREFIXES.contains(&"general."));
        assert!(AC_INSM_003_VALID_PREFIXES.contains(&"llama."));
    }

    #[test]
    fn provenance_forbidden_shorthand() {
        assert!(AC_INSM_003_FORBIDDEN_SHORTHAND.contains(&"n_embd"));
        assert!(AC_INSM_003_FORBIDDEN_SHORTHAND.contains(&"n_heads"));
        assert!(AC_INSM_003_FORBIDDEN_SHORTHAND.contains(&"n_layers"));
    }

    // -------------------------------------------------------------------------
    // Section 2: Domain helpers.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_arch_prefix_qwen2() {
        assert!(has_valid_arch_prefix("qwen2.embedding_length"));
        assert!(has_valid_arch_prefix("qwen2.block_count"));
    }

    #[test]
    fn domain_arch_prefix_general() {
        assert!(has_valid_arch_prefix("general.architecture"));
        assert!(has_valid_arch_prefix("general.name"));
    }

    #[test]
    fn domain_arch_prefix_llama() {
        assert!(has_valid_arch_prefix("llama.embedding_length"));
    }

    #[test]
    fn domain_arch_prefix_rejected() {
        assert!(!has_valid_arch_prefix("n_embd"));
        assert!(!has_valid_arch_prefix("embedding_length"));
        assert!(!has_valid_arch_prefix(""));
    }

    #[test]
    fn domain_forbidden_shorthand_recognized() {
        assert!(is_forbidden_shorthand("n_embd"));
        assert!(is_forbidden_shorthand("n_heads"));
        assert!(is_forbidden_shorthand("n_layers"));
        assert!(is_forbidden_shorthand("n_kv"));
        assert!(!is_forbidden_shorthand("qwen2.embedding_length"));
    }

    // -------------------------------------------------------------------------
    // Section 3: INSPECT-META-001 — KV count parity.
    // -------------------------------------------------------------------------
    #[test]
    fn meta001_pass_match() {
        assert_eq!(
            verdict_from_kv_count_parity(28, 28),
            InsmVerdict::Pass
        );
    }

    #[test]
    fn meta001_fail_inspect_truncated() {
        // Bug: inspect shows 4 (stub), hex shows 28.
        assert_eq!(
            verdict_from_kv_count_parity(4, 28),
            InsmVerdict::Fail
        );
    }

    #[test]
    fn meta001_fail_inspect_inflated() {
        assert_eq!(
            verdict_from_kv_count_parity(30, 28),
            InsmVerdict::Fail
        );
    }

    #[test]
    fn meta001_fail_zero_hex_count() {
        assert_eq!(verdict_from_kv_count_parity(0, 0), InsmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: INSPECT-META-002 — ≥ 20 keys.
    // -------------------------------------------------------------------------
    #[test]
    fn meta002_pass_at_floor() {
        assert_eq!(verdict_from_kv_count_floor(20), InsmVerdict::Pass);
    }

    #[test]
    fn meta002_pass_typical_qwen25() {
        // Real Qwen2.5-Coder-7B has ~28 KV.
        assert_eq!(verdict_from_kv_count_floor(28), InsmVerdict::Pass);
    }

    #[test]
    fn meta002_fail_4_key_stub() {
        // The exact regression: hand-written stub returned 4 keys.
        assert_eq!(verdict_from_kv_count_floor(4), InsmVerdict::Fail);
    }

    #[test]
    fn meta002_fail_just_below_floor() {
        assert_eq!(verdict_from_kv_count_floor(19), InsmVerdict::Fail);
    }

    #[test]
    fn meta002_fail_zero() {
        assert_eq!(verdict_from_kv_count_floor(0), InsmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: INSPECT-META-003 — arch-scoped keys.
    // -------------------------------------------------------------------------
    #[test]
    fn meta003_pass_qwen2_keys() {
        let keys = vec![
            "qwen2.embedding_length",
            "qwen2.block_count",
            "general.architecture",
        ];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Pass);
    }

    #[test]
    fn meta003_pass_llama_keys() {
        let keys = vec!["llama.embedding_length", "general.name"];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Pass);
    }

    #[test]
    fn meta003_fail_n_embd_shorthand() {
        // The exact regression: hand-written shorthand instead of authentic.
        let keys = vec!["n_embd", "n_heads", "n_layers", "n_kv"];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Fail);
    }

    #[test]
    fn meta003_fail_one_shorthand_in_otherwise_valid() {
        // Even one shorthand poisons the verdict.
        let keys = vec!["qwen2.embedding_length", "n_heads"];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Fail);
    }

    #[test]
    fn meta003_fail_no_valid_prefix() {
        // Keys with no valid prefix at all (no shorthand either).
        let keys = vec!["embedding_length", "block_count"];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Fail);
    }

    #[test]
    fn meta003_fail_empty() {
        let keys: Vec<&str> = vec![];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: INSPECT-META-004 — no hand-written stub in source.
    // -------------------------------------------------------------------------
    #[test]
    fn meta004_pass_clean_code() {
        let code = r#"
            fn validate_inspect(model: &Model) {
                for (key, value) in &model.metadata.kv_pairs {
                    meta_map.insert(key.clone(), value.clone());
                }
            }
        "#;
        assert_eq!(verdict_from_no_meta_map_stub(code), InsmVerdict::Pass);
    }

    #[test]
    fn meta004_fail_n_embd_stub() {
        let code = r#"
            // Pre-fix stub:
            meta_map.insert("n_embd".to_string(), "4096".to_string());
        "#;
        assert_eq!(verdict_from_no_meta_map_stub(code), InsmVerdict::Fail);
    }

    #[test]
    fn meta004_fail_n_heads_stub() {
        let code = r#"meta_map.insert("n_heads".to_string(), "32".into());"#;
        assert_eq!(verdict_from_no_meta_map_stub(code), InsmVerdict::Fail);
    }

    #[test]
    fn meta004_fail_n_layers_stub() {
        let code = r#"meta_map.insert("n_layers".into(), "28".into());"#;
        assert_eq!(verdict_from_no_meta_map_stub(code), InsmVerdict::Fail);
    }

    #[test]
    fn meta004_pass_empty() {
        assert_eq!(verdict_from_no_meta_map_stub(""), InsmVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full bug regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_4_key_stub_caught() {
        // Pre-fix: validate_inspect.rs returned 4 hand-written keys.
        let stub_keys = vec!["n_embd", "n_heads", "n_layers", "n_kv"];
        // INSPECT-META-001 vs hex (28): mismatch.
        assert_eq!(
            verdict_from_kv_count_parity(stub_keys.len(), 28),
            InsmVerdict::Fail
        );
        // INSPECT-META-002 floor:
        assert_eq!(
            verdict_from_kv_count_floor(stub_keys.len()),
            InsmVerdict::Fail
        );
        // INSPECT-META-003 arch-scoped:
        assert_eq!(
            verdict_from_arch_scoped_keys(&stub_keys),
            InsmVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_full_pipeline_passes_all_4_gates() {
        // Post-fix: real GGUF KV pairs propagated.
        let real_keys = vec![
            "general.architecture",
            "general.name",
            "general.version",
            "general.alignment",
            "general.author",
            "general.description",
            "qwen2.embedding_length",
            "qwen2.block_count",
            "qwen2.context_length",
            "qwen2.attention.head_count",
            "qwen2.attention.head_count_kv",
            "qwen2.attention.layer_norm_rms_epsilon",
            "qwen2.feed_forward_length",
            "qwen2.rope.freq_base",
            "qwen2.rope.dimension_count",
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.merges",
            "tokenizer.chat_template",
            "general.quantization_version",
            "general.file_type",
        ];
        let n = real_keys.len();
        assert_eq!(n, 28);

        // INSPECT-META-001:
        assert_eq!(verdict_from_kv_count_parity(n, n), InsmVerdict::Pass);
        // INSPECT-META-002:
        assert_eq!(verdict_from_kv_count_floor(n), InsmVerdict::Pass);
        // INSPECT-META-003:
        // (note: tokenizer.* doesn't match general/qwen2/llama prefix in
        // our minimal valid_prefixes, but at least one qwen2.* / general.*
        // is present so the predicate passes.)
        assert_eq!(
            verdict_from_arch_scoped_keys(&real_keys),
            InsmVerdict::Pass
        );
        // INSPECT-META-004:
        let clean_code = "fn validate_inspect() { propagate_kv_pairs(); }";
        assert_eq!(
            verdict_from_no_meta_map_stub(clean_code),
            InsmVerdict::Pass
        );
    }

    #[test]
    fn realistic_kv_truncation_caught() {
        // INSPECT-META-001 if_fails: "inspect is truncating metadata —
        // raw KV pairs not propagated".
        assert_eq!(
            verdict_from_kv_count_parity(4, 28),
            InsmVerdict::Fail
        );
    }

    #[test]
    fn realistic_shorthand_fabrications_caught() {
        // INSPECT-META-003 if_fails: "Keys are still n_embd/n_heads
        // fabrications instead of qwen2.embedding_length style".
        let keys = vec!["n_embd", "n_heads", "n_layers"];
        assert_eq!(verdict_from_arch_scoped_keys(&keys), InsmVerdict::Fail);
    }

    #[test]
    fn realistic_root_cause_unfixed_caught() {
        // INSPECT-META-004 if_fails: "Hand-written 4-key stub still in code".
        let buggy_code = r#"
            // bug: still inserting shorthand
            meta_map.insert("n_embd", embedding_length);
            meta_map.insert("n_heads", head_count);
            meta_map.insert("n_layers", block_count);
        "#;
        assert_eq!(verdict_from_no_meta_map_stub(buggy_code), InsmVerdict::Fail);
    }
}
