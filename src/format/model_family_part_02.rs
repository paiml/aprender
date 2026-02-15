
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
}

impl FamilyRegistry {
    /// Create an empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
        }
    }

    /// Register a model family
    pub fn register(&mut self, family: Box<dyn ModelFamily>) {
        self.families.push(family);
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
