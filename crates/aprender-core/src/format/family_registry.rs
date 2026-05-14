
impl ModelFamily for DynModelFamily {
    fn family_name(&self) -> &str {
        &self.config.family
    }

    fn display_name(&self) -> &str {
        &self.config.display_name
    }

    fn config(&self) -> &ModelFamilyConfig {
        &self.config
    }

    fn size_config(&self, size: &str) -> Option<&ModelSizeConfig> {
        self.config.size_variants.get(size)
    }

    fn detect_size(&self, hidden_dim: usize, num_layers: usize) -> Option<String> {
        for (name, variant) in &self.config.size_variants {
            if variant.hidden_dim == hidden_dim && variant.num_layers == num_layers {
                return Some(name.clone());
            }
        }
        None
    }

    fn constraints(&self) -> &ModelConstraints {
        &self.config.constraints
    }

    fn expected_tensor_count(&self, size: &str) -> Option<usize> {
        let variant = self.config.size_variants.get(size)?;
        let num_layers = variant.num_layers;

        // Count global tensors
        let mut count = 0usize;
        if !self.config.tensor_template.embedding.is_empty() {
            count += 1;
        }
        if self.config.tensor_template.lm_head.is_some() {
            count += 1;
        }
        if self.config.tensor_template.final_norm.is_some() {
            count += 1;
        }

        // Count per-layer tensors
        let tensors_per_layer = self
            .config
            .tensor_template
            .per_layer
            .values()
            .filter(|v| v.is_some())
            .count();
        count += tensors_per_layer * num_layers;

        Some(count)
    }

    fn validate_tensor_names(
        &self,
        names: &[&str],
        size: &str,
    ) -> std::result::Result<(), ContractError> {
        let variant = self
            .config
            .size_variants
            .get(size)
            .ok_or_else(|| ContractError {
                family: self.config.family.clone(),
                message: format!("Unknown size variant: {size}"),
            })?;

        // Build expected tensor names
        let mut expected: Vec<String> = Vec::new();
        expected.push(self.config.tensor_template.embedding.clone());
        if let Some(lm_head) = &self.config.tensor_template.lm_head {
            expected.push(lm_head.clone());
        }
        if let Some(final_norm) = &self.config.tensor_template.final_norm {
            expected.push(final_norm.clone());
        }

        for layer_idx in 0..variant.num_layers {
            for pat in self.config.tensor_template.per_layer.values().flatten() {
                expected.push(pat.replace("{n}", &layer_idx.to_string()));
            }
        }

        // Check for unexpected tensors (tensor names not in expected list)
        let expected_set: std::collections::HashSet<&str> =
            expected.iter().map(String::as_str).collect();
        let actual_set: std::collections::HashSet<&str> = names.iter().copied().collect();

        let missing: Vec<&str> = expected_set.difference(&actual_set).copied().collect();
        let unexpected: Vec<&str> = actual_set.difference(&expected_set).copied().collect();

        if !missing.is_empty() || !unexpected.is_empty() {
            let mut msg = String::new();
            if !missing.is_empty() {
                msg.push_str(&format!("Missing tensors: {}", missing.join(", ")));
            }
            if !unexpected.is_empty() {
                if !msg.is_empty() {
                    msg.push_str("; ");
                }
                msg.push_str(&format!("Unexpected tensors: {}", unexpected.join(", ")));
            }
            return Err(ContractError {
                family: self.config.family.clone(),
                message: msg,
            });
        }

        Ok(())
    }
}

// ============================================================================
// Family Registry
// ============================================================================

/// Registry of known model families for detection.
#[derive(Debug)]
pub struct FamilyRegistry {
    families: Vec<Box<dyn ModelFamily>>,
    /// HF repo glob patterns aliased to a parent family. e.g. ("codellama/*", "llama").
    aliases: Vec<(String, String)>,
}

/// Discriminator field detected for a family. Used by `detect_from_config_str`
/// to route raw HF `config.json` bodies to the right family without
/// parsing the whole config.
///
/// Order matters in `DISCRIMINATOR_DISPATCH`: more-specific discriminators
/// come first. e.g., Qwen3.5 (`tie_word_embeddings` + `head_dim` +
/// `qwen3_5`) must be checked before Qwen3 (`head_dim` + `qwen3`).
#[derive(Debug, Clone, Copy)]
struct DiscriminatorRule {
    family: &'static str,
    /// Field/marker substrings — all must be present in the config body.
    must_contain: &'static [&'static str],
    /// Substrings whose absence is required (e.g., qwen3 must NOT contain qwen3_5).
    must_not_contain: &'static [&'static str],
}

const DISCRIMINATOR_DISPATCH: &[DiscriminatorRule] = &[
    // Qwen3.5: tie_word_embeddings + head_dim + qwen3_5 marker
    DiscriminatorRule {
        family: "qwen3_5",
        must_contain: &["tie_word_embeddings", "head_dim", "qwen3_5"],
        must_not_contain: &[],
    },
    // Qwen3: head_dim + qwen3, NOT qwen3_5
    DiscriminatorRule {
        family: "qwen3",
        must_contain: &["head_dim", "qwen3"],
        must_not_contain: &["qwen3_5"],
    },
    // Qwen2: qwen2 model_type + rope_theta
    DiscriminatorRule {
        family: "qwen2",
        must_contain: &["qwen2", "rope_theta"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "phi",
        must_contain: &["qkv_proj_fused"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "gemma",
        must_contain: &["query_pre_attn_scalar"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "gptneox",
        must_contain: &["use_parallel_residual"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "opt",
        must_contain: &["do_layer_norm_before"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "gpt2",
        must_contain: &["\"n_embd\""],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "openelm",
        must_contain: &["ffn_multipliers", "num_query_heads"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "deepseek",
        must_contain: &["n_routed_experts"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "falcon_h1",
        must_contain: &["mamba_d_state", "mamba_expand", "falcon_h1"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "rwkv7",
        must_contain: &["time_mix_extra_dim"],
        must_not_contain: &[],
    },
    // MAMBA: state_size + conv_kernel without attention (pure SSM)
    DiscriminatorRule {
        family: "mamba",
        must_contain: &["state_size", "conv_kernel"],
        must_not_contain: &["num_attention_heads"],
    },
    DiscriminatorRule {
        family: "bert",
        must_contain: &["type_vocab_size"],
        must_not_contain: &[],
    },
    // Mistral: sliding_window + MistralForCausalLM
    DiscriminatorRule {
        family: "mistral",
        must_contain: &["sliding_window", "MistralForCausalLM"],
        must_not_contain: &[],
    },
    // Whisper / Moonshine: speech families identified by architecture string
    DiscriminatorRule {
        family: "whisper",
        must_contain: &["WhisperForConditionalGeneration"],
        must_not_contain: &[],
    },
    DiscriminatorRule {
        family: "moonshine",
        must_contain: &["MoonshineForConditionalGeneration"],
        must_not_contain: &[],
    },
    // Llama: catch-all for transformer configs that didn't match anything more specific.
    // Must be checked LAST.
    DiscriminatorRule {
        family: "llama",
        must_contain: &["LlamaForCausalLM"],
        must_not_contain: &[],
    },
];

impl FamilyRegistry {
    /// Create an empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Register a model family
    pub fn register(&mut self, family: Box<dyn ModelFamily>) {
        self.families.push(family);
    }

    /// Register an HF repo pattern as an alias for an existing family.
    ///
    /// Used to unblock derived models that share the parent family's
    /// architecture but have different vocab/repo patterns. e.g.,
    /// `register_alias("codellama/*", "llama")` lets codellama checkpoints
    /// dispatch through the llama loader.
    ///
    /// The pattern is a simple glob with `*` as suffix wildcard. Returns
    /// `Err` if the parent family is not registered.
    pub fn register_alias(
        &mut self,
        hf_pattern: &str,
        parent_family: &str,
    ) -> std::result::Result<(), String> {
        if self.get(parent_family).is_none() {
            return Err(format!(
                "cannot alias '{hf_pattern}' to unregistered family '{parent_family}'"
            ));
        }
        self.aliases
            .push((hf_pattern.to_string(), parent_family.to_string()));
        Ok(())
    }

    /// Resolve an HF repo identifier through registered aliases. Returns
    /// the parent family for the first matching alias, or None.
    #[must_use]
    pub fn resolve_alias(&self, hf_repo: &str) -> Option<&str> {
        for (pattern, parent) in &self.aliases {
            if alias_matches(pattern, hf_repo) {
                return Some(parent.as_str());
            }
        }
        None
    }

    /// Number of registered aliases.
    #[must_use]
    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    /// Get all registered family names
    #[must_use]
    pub fn family_names(&self) -> Vec<&str> {
        self.families.iter().map(|f| f.family_name()).collect()
    }

    /// Look up a family by name
    #[must_use]
    pub fn get(&self, family_name: &str) -> Option<&dyn ModelFamily> {
        self.families
            .iter()
            .find(|f| f.family_name() == family_name)
            .map(|f| f.as_ref())
    }

    /// Detect model family from tensor names using best-match scoring.
    ///
    /// Scores each family by counting how many of its expected tensor patterns
    /// (embedding + per-layer for layer 0) match the given tensor names.
    /// Returns the family with the highest score, which disambiguates families
    /// with overlapping naming conventions (e.g., Qwen2's bias tensors
    /// distinguish it from LLaMA/DeepSeek/Mistral which share the same base
    /// naming but lack bias patterns).
    #[must_use]
    pub fn detect_family(&self, tensor_names: &[&str]) -> Option<&dyn ModelFamily> {
        let mut best: Option<(usize, &dyn ModelFamily)> = None;

        for family in &self.families {
            let config = family.config();

            // Must have the embedding tensor
            if !tensor_names.contains(&config.tensor_template.embedding.as_str()) {
                continue;
            }

            // Score: 1 point for embedding match + 1 for each per-layer pattern match
            let mut score = 1usize;
            for pattern in config.tensor_template.per_layer.values().flatten() {
                let layer0 = pattern.replace("{n}", "0");
                if tensor_names.contains(&layer0.as_str()) {
                    score += 1;
                }
            }

            // Need at least one per-layer match (score > 1)
            if score <= 1 {
                continue;
            }

            match best {
                None => best = Some((score, family.as_ref())),
                Some((best_score, _)) if score > best_score => {
                    best = Some((score, family.as_ref()));
                }
                _ => {}
            }
        }

        best.map(|(_, family)| family)
    }

    /// Detect model family from HuggingFace `model_type` string.
    #[must_use]
    pub fn detect_from_model_type(&self, model_type: &str) -> Option<&dyn ModelFamily> {
        let model_type_lower = model_type.to_lowercase();

        // Pass 1: exact family name match (prevents "qwen3_5" matching "qwen3")
        for family in &self.families {
            if family.config().family == model_type_lower {
                return Some(family.as_ref());
            }
        }

        // Pass 2: architecture class or substring match
        for family in &self.families {
            let config = family.config();
            for arch in &config.architectures {
                if arch.to_lowercase().contains(&model_type_lower)
                    || model_type_lower.contains(&config.family)
                {
                    return Some(family.as_ref());
                }
            }
        }
        None
    }

    /// Number of registered families
    #[must_use]
    pub fn len(&self) -> usize {
        self.families.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.families.is_empty()
    }

    /// Detect model family from a raw HF `config.json` body string.
    ///
    /// Uses discriminator-field-based dispatch in priority order, mirroring
    /// the `apr-cookbook` architecture-demos detector recipe. Falls through
    /// to `LlamaForCausalLM` as the catch-all for transformer configs that
    /// don't match anything more specific.
    ///
    /// Returns the family name (e.g. "llama", "qwen2", "phi") if a
    /// discriminator matches, or `None` if the config is unrecognized.
    /// Use `get(name)` to obtain the registered family object after detection.
    ///
    /// This is the high-level entry point most callers want — `detect_family`
    /// requires already-parsed tensor names; `detect_from_config_str` works
    /// directly off the JSON body and surfaces all 18 supported families.
    #[must_use]
    pub fn detect_from_config_str(body: &str) -> Option<&'static str> {
        for rule in DISCRIMINATOR_DISPATCH {
            let all_present = rule.must_contain.iter().all(|m| body.contains(m));
            let none_excluded = rule.must_not_contain.iter().all(|m| !body.contains(m));
            if all_present && none_excluded {
                return Some(rule.family);
            }
        }
        None
    }

    /// All 18 supported family names tracked by the detector dispatch.
    #[must_use]
    pub fn supported_families() -> Vec<&'static str> {
        DISCRIMINATOR_DISPATCH.iter().map(|r| r.family).collect()
    }
}

/// Match an HF repo identifier against a glob pattern with `*` as suffix
/// wildcard. e.g., "codellama/*" matches "codellama/CodeLlama-7b-hf".
fn alias_matches(pattern: &str, hf_repo: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        hf_repo.starts_with(prefix)
    } else {
        pattern == hf_repo
    }
}

impl Default for FamilyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Build-Time Generated Code (PMAT-250)
// ============================================================================
//
// This include! pulls in code generated by build.rs from
// contracts/model-families/*.yaml. It provides:
// - KNOWN_FAMILIES: &[&str] — list of family names
// - Per-family const definitions (e.g., QWEN2_0_5B_HIDDEN_DIM)
// - build_default_registry() → FamilyRegistry with all families

include!(concat!(env!("OUT_DIR"), "/model_families_generated.rs"));

#[cfg(test)]
mod detector_tests {
    use super::*;

    #[test]
    fn detect_llama_from_minimal_config() {
        let body = r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}"#;
        assert_eq!(FamilyRegistry::detect_from_config_str(body), Some("llama"));
    }

    #[test]
    fn detect_mistral_via_sliding_window() {
        let body = r#"{"architectures": ["MistralForCausalLM"], "sliding_window": 4096}"#;
        assert_eq!(
            FamilyRegistry::detect_from_config_str(body),
            Some("mistral")
        );
    }

    #[test]
    fn detect_qwen3_5_takes_priority_over_qwen3() {
        let body = r#"{"model_type": "qwen3_5", "head_dim": 64, "tie_word_embeddings": true}"#;
        assert_eq!(
            FamilyRegistry::detect_from_config_str(body),
            Some("qwen3_5")
        );
    }

    #[test]
    fn detect_phi_via_qkv_fused() {
        let body = r#"{"qkv_proj_fused": true}"#;
        assert_eq!(FamilyRegistry::detect_from_config_str(body), Some("phi"));
    }

    #[test]
    fn detect_bert_via_type_vocab_size() {
        let body = r#"{"architectures": ["BertForMaskedLM"], "type_vocab_size": 2}"#;
        assert_eq!(FamilyRegistry::detect_from_config_str(body), Some("bert"));
    }

    #[test]
    fn detect_mamba_pure_ssm_no_attention() {
        let body = r#"{"state_size": 16, "conv_kernel": 4}"#;
        assert_eq!(FamilyRegistry::detect_from_config_str(body), Some("mamba"));
    }

    #[test]
    fn unknown_config_returns_none() {
        let body = r#"{"model_type": "completely_unknown_arch"}"#;
        assert_eq!(FamilyRegistry::detect_from_config_str(body), None);
    }

    #[test]
    fn supported_families_count_matches_dispatch_table() {
        // 18 supported (16 in-progress + whisper + moonshine)
        assert_eq!(FamilyRegistry::supported_families().len(), 18);
    }

    #[test]
    fn detect_is_deterministic() {
        let body = r#"{"architectures": ["LlamaForCausalLM"], "head_dim": 64, "qwen3": true}"#;
        let a = FamilyRegistry::detect_from_config_str(body);
        let b = FamilyRegistry::detect_from_config_str(body);
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    #[test]
    fn alias_matches_glob_prefix() {
        assert!(alias_matches("codellama/*", "codellama/CodeLlama-7b-hf"));
        assert!(!alias_matches("codellama/*", "meta-llama/Llama-3-8b"));
    }

    #[test]
    fn alias_matches_exact() {
        assert!(alias_matches(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ));
        assert!(!alias_matches(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "other/checkpoint"
        ));
    }

    #[test]
    fn register_alias_rejects_unregistered_parent() {
        let mut registry = FamilyRegistry::new();
        let result = registry.register_alias("codellama/*", "llama");
        assert!(
            result.is_err(),
            "should reject alias to unregistered family"
        );
    }

    #[test]
    fn alias_count_starts_at_zero() {
        let registry = FamilyRegistry::new();
        assert_eq!(registry.alias_count(), 0);
    }

    #[test]
    fn resolve_alias_returns_none_for_unknown_repo() {
        let registry = FamilyRegistry::new();
        assert_eq!(registry.resolve_alias("openai/gpt-3"), None);
    }
}
