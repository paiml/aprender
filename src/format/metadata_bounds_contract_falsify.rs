// ============================================================================
// Model Metadata Bounds Contract Falsification (FALSIFY-MB-001..005)
//
// Verifies that:
// 1. The YAML contract matches the Rust validation constants in realizar
// 2. All model-family configs satisfy the bounds
// 3. Upper bounds have sufficient headroom over real maxima
//
// Contract: contracts/model-metadata-bounds-v1.yaml
// Rust enforcement: realizar/src/gguf/config.rs::validate_metadata_bounds()
// ============================================================================

#[cfg(test)]
mod metadata_bounds_contract {
    use std::collections::HashMap;
    use std::path::Path;

    use crate::format::model_family::ModelFamilyConfig;
    use crate::format::model_family_loader::load_family_yaml;

    // ========================================================================
    // YAML parser for metadata-bounds contract
    // ========================================================================

    /// Parsed upper bound entry from YAML.
    #[derive(Debug)]
    struct UpperBound {
        field: String,
        max: u64,
    }

    /// Parsed range bound entry from YAML.
    #[derive(Debug)]
    struct RangeBound {
        field: String,
        min: f64,
        max: f64,
    }

    /// Read the bounds contract YAML.
    fn read_bounds_content() -> String {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/model-metadata-bounds-v1.yaml");
        assert!(
            path.exists(),
            "contracts/model-metadata-bounds-v1.yaml must exist (Gap 2)"
        );
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read model-metadata-bounds: {e}"))
    }

    /// Parse the upper_bounds section from YAML.
    fn parse_upper_bounds(content: &str) -> Vec<UpperBound> {
        let mut bounds = Vec::new();
        let mut in_section = false;
        let mut current_field = String::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Detect section start
            if trimmed == "upper_bounds:" {
                in_section = true;
                continue;
            }
            // Detect section end (new top-level key)
            if in_section && !line.starts_with(' ') && !line.starts_with('#') && !trimmed.is_empty()
            {
                break;
            }
            if !in_section {
                continue;
            }

            let indent = line.len() - line.trim_start().len();
            if indent >= 4 {
                if let Some(rest) = trimmed.strip_prefix("field:") {
                    current_field = rest.split('#').next().unwrap_or("").trim().to_string();
                } else if let Some(rest) = trimmed.strip_prefix("max:") {
                    let val_str = rest.split('#').next().unwrap_or("").trim();
                    if let Ok(max) = val_str.parse::<u64>() {
                        if !current_field.is_empty() {
                            bounds.push(UpperBound {
                                field: current_field.clone(),
                                max,
                            });
                        }
                    }
                }
            }
        }

        bounds
    }

    /// Parse the range_bounds section from YAML.
    fn parse_range_bounds(content: &str) -> Vec<RangeBound> {
        let mut bounds = Vec::new();
        let mut in_section = false;
        let mut current_field = String::new();
        let mut current_min = f64::NAN;
        let mut current_max = f64::NAN;

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed == "range_bounds:" {
                in_section = true;
                continue;
            }
            if in_section && !line.starts_with(' ') && !line.starts_with('#') && !trimmed.is_empty()
            {
                break;
            }
            if !in_section {
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            // New entry at list-item level
            if indent == 2 && trimmed.starts_with("- id:") {
                // Save previous entry
                if !current_field.is_empty() && !current_min.is_nan() && !current_max.is_nan() {
                    bounds.push(RangeBound {
                        field: current_field.clone(),
                        min: current_min,
                        max: current_max,
                    });
                }
                current_field.clear();
                current_min = f64::NAN;
                current_max = f64::NAN;
                continue;
            }

            if indent >= 4 {
                if let Some(rest) = trimmed.strip_prefix("field:") {
                    current_field = rest.split('#').next().unwrap_or("").trim().to_string();
                } else if let Some(rest) = trimmed.strip_prefix("min:") {
                    let val_str = rest.split('#').next().unwrap_or("").trim();
                    if let Ok(v) = val_str.parse::<f64>() {
                        current_min = v;
                    }
                } else if let Some(rest) = trimmed.strip_prefix("max:") {
                    let val_str = rest.split('#').next().unwrap_or("").trim();
                    if let Ok(v) = val_str.parse::<f64>() {
                        current_max = v;
                    }
                }
            }
        }

        // Save last entry
        if !current_field.is_empty() && !current_min.is_nan() && !current_max.is_nan() {
            bounds.push(RangeBound {
                field: current_field,
                min: current_min,
                max: current_max,
            });
        }

        bounds
    }

    /// Load all model-family YAML configs.
    fn load_all_family_configs() -> Vec<(String, ModelFamilyConfig)> {
        let families_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/model-families");
        assert!(
            families_dir.exists(),
            "contracts/model-families/ must exist"
        );

        let mut families = Vec::new();
        let entries = std::fs::read_dir(&families_dir).expect("read model-families dir");

        for entry in entries {
            let entry = entry.expect("read dir entry");
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let ext_is_yaml = path.extension().is_some_and(|ext| {
                ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml")
            });
            if !ext_is_yaml || file_name.starts_with('_') {
                continue;
            }

            let config = load_family_yaml(&path)
                .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()));
            let name = file_name
                .trim_end_matches(".yaml")
                .trim_end_matches(".yml")
                .to_string();
            families.push((name, config));
        }

        families
    }

    /// Get the max for a given field from parsed upper bounds.
    fn get_upper_bound(bounds: &[UpperBound], field: &str) -> Option<u64> {
        bounds.iter().find(|b| b.field == field).map(|b| b.max)
    }

    /// Get the range for a given field from parsed range bounds.
    fn get_range_bound(bounds: &[RangeBound], field: &str) -> Option<(f64, f64)> {
        bounds
            .iter()
            .find(|b| b.field == field)
            .map(|b| (b.min, b.max))
    }

    // ========================================================================
    // FALSIFY-MB-001: YAML upper_bounds match Rust constants
    // ========================================================================
    #[test]
    fn falsify_mb_001_upper_bounds_match_rust() {
        let content = read_bounds_content();
        let bounds = parse_upper_bounds(&content);

        // These are the hardcoded constants from realizar/src/gguf/config.rs
        // validate_metadata_bounds() → check_usize_max() calls.
        let rust_bounds: HashMap<&str, u64> = [
            ("hidden_dim", 65_536),
            ("num_layers", 256),
            ("num_heads", 256),
            ("num_kv_heads", 256),
            ("vocab_size", 1_000_000),
            ("intermediate_dim", 262_144),
            ("context_length", 2_097_152),
        ]
        .into_iter()
        .collect();

        let mut violations = Vec::new();

        // Check that every Rust bound has a YAML counterpart
        for (field, rust_max) in &rust_bounds {
            match get_upper_bound(&bounds, field) {
                Some(yaml_max) => {
                    if yaml_max != *rust_max {
                        violations
                            .push(format!("{field}: YAML max={yaml_max}, Rust max={rust_max}"));
                    }
                }
                None => {
                    violations.push(format!(
                        "{field}: Rust has check_usize_max({rust_max}) but no YAML entry"
                    ));
                }
            }
        }

        // Check that every YAML bound has a Rust counterpart
        for bound in &bounds {
            if !rust_bounds.contains_key(bound.field.as_str()) {
                violations.push(format!(
                    "{}: YAML has upper bound {} but no Rust check",
                    bound.field, bound.max
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MB-001: YAML ↔ Rust upper bounds divergence:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MB-002: YAML range_bounds match Rust constants
    // ========================================================================
    #[test]
    fn falsify_mb_002_range_bounds_match_rust() {
        let content = read_bounds_content();
        let ranges = parse_range_bounds(&content);

        // Hardcoded from realizar/src/gguf/config.rs validate_metadata_bounds()
        let rust_ranges: HashMap<&str, (f64, f64)> =
            [("rope_theta", (1.0, 100_000_000.0)), ("eps", (1e-10, 0.01))]
                .into_iter()
                .collect();

        let mut violations = Vec::new();

        for (field, (rust_min, rust_max)) in &rust_ranges {
            match get_range_bound(&ranges, field) {
                Some((yaml_min, yaml_max)) => {
                    if (yaml_min - rust_min).abs() > 1e-15 {
                        violations
                            .push(format!("{field}: YAML min={yaml_min}, Rust min={rust_min}"));
                    }
                    if (yaml_max - rust_max).abs() > 1e-6 {
                        violations
                            .push(format!("{field}: YAML max={yaml_max}, Rust max={rust_max}"));
                    }
                }
                None => {
                    violations.push(format!(
                        "{field}: Rust has range [{rust_min}, {rust_max}] but no YAML entry"
                    ));
                }
            }
        }

        for range in &ranges {
            if !rust_ranges.contains_key(range.field.as_str()) {
                violations.push(format!(
                    "{}: YAML has range [{}, {}] but no Rust check",
                    range.field, range.min, range.max
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MB-002: YAML ↔ Rust range bounds divergence:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MB-003: All model-family configs satisfy upper bounds
    // ========================================================================
    #[test]
    fn falsify_mb_003_family_configs_within_upper_bounds() {
        let content = read_bounds_content();
        let bounds = parse_upper_bounds(&content);
        let families = load_all_family_configs();

        assert!(!families.is_empty(), "No model families loaded");
        assert!(!bounds.is_empty(), "No upper bounds parsed");

        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for size in config.size_variants.values() {
                let label = format!("{family_name}/{}", size.parameters);

                let checks: &[(&str, u64)] = &[
                    ("hidden_dim", size.hidden_dim as u64),
                    ("num_layers", size.num_layers as u64),
                    ("num_heads", size.num_heads as u64),
                    ("num_kv_heads", size.num_kv_heads as u64),
                    ("vocab_size", size.vocab_size as u64),
                    ("intermediate_dim", size.intermediate_dim as u64),
                ];

                for &(field, val) in checks {
                    if let Some(max) = get_upper_bound(&bounds, field) {
                        if val > max {
                            violations.push(format!("{label}.{field}: {val} > max {max}"));
                        }
                    }
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MB-003: Family configs exceed upper bounds:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MB-004: All model-family configs satisfy structural invariants
    // ========================================================================
    #[test]
    fn falsify_mb_004_family_configs_structural_invariants() {
        let families = load_all_family_configs();
        assert!(!families.is_empty(), "No model families loaded");

        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for size in config.size_variants.values() {
                let label = format!("{family_name}/{}", size.parameters);

                // SI-001..006: positive dimensions
                let positive_checks: &[(&str, usize)] = &[
                    ("hidden_dim", size.hidden_dim),
                    ("num_layers", size.num_layers),
                    ("vocab_size", size.vocab_size),
                    ("num_heads", size.num_heads),
                    ("num_kv_heads", size.num_kv_heads),
                    ("intermediate_dim", size.intermediate_dim),
                ];

                for &(field, val) in positive_checks {
                    if val == 0 {
                        violations.push(format!("{label}.{field}: is 0 (SI violation)"));
                    }
                }

                // SI-007: head_dim > 0
                if size.head_dim == 0 {
                    violations.push(format!("{label}.head_dim: is 0 (SI-007)"));
                }

                // SI-009: GQA divisibility
                if size.num_kv_heads > 0 && size.num_heads % size.num_kv_heads != 0 {
                    violations.push(format!(
                        "{label}: num_heads ({}) not divisible by num_kv_heads ({}) (SI-009)",
                        size.num_heads, size.num_kv_heads
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MB-004: Structural invariant violations:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MB-005: Upper bounds have headroom over real maxima
    // ========================================================================
    #[test]
    fn falsify_mb_005_upper_bounds_headroom() {
        let content = read_bounds_content();
        let bounds = parse_upper_bounds(&content);
        let families = load_all_family_configs();

        assert!(!families.is_empty(), "No model families loaded");

        // Collect max observed value for each field across all families
        let mut max_observed: HashMap<&str, u64> = HashMap::new();
        let fields = [
            "hidden_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "vocab_size",
            "intermediate_dim",
        ];

        for (_, config) in &families {
            for size in config.size_variants.values() {
                let values: &[(&str, u64)] = &[
                    ("hidden_dim", size.hidden_dim as u64),
                    ("num_layers", size.num_layers as u64),
                    ("num_heads", size.num_heads as u64),
                    ("num_kv_heads", size.num_kv_heads as u64),
                    ("vocab_size", size.vocab_size as u64),
                    ("intermediate_dim", size.intermediate_dim as u64),
                ];

                for &(field, val) in values {
                    let entry = max_observed.entry(field).or_insert(0);
                    if val > *entry {
                        *entry = val;
                    }
                }
            }
        }

        let mut warnings = Vec::new();

        for field in &fields {
            let Some(bound_max) = get_upper_bound(&bounds, field) else {
                continue;
            };
            let observed = max_observed.get(field).copied().unwrap_or(0);
            if observed == 0 {
                continue;
            }

            // Check that max observed is less than 50% of the upper bound.
            // If it's more, the bound may be too tight for future models.
            let headroom_ratio = observed as f64 / bound_max as f64;
            if headroom_ratio > 0.5 {
                warnings.push(format!(
                    "{field}: max observed = {observed}, upper bound = {bound_max} \
                     (ratio {headroom_ratio:.2} > 0.50 — consider raising)"
                ));
            }
        }

        assert!(
            warnings.is_empty(),
            "FALSIFY-MB-005: Insufficient headroom (observed > 50%% of bound):\n{}",
            warnings.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MB-006: range_bounds float fields within bounds for real models
    // ========================================================================
    #[test]
    fn falsify_mb_006_family_float_ranges() {
        let content = read_bounds_content();
        let ranges = parse_range_bounds(&content);
        let families = load_all_family_configs();

        assert!(!families.is_empty(), "No model families loaded");

        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for size in config.size_variants.values() {
                let label = format!("{family_name}/{}", size.parameters);

                // rope_theta
                if size.rope_theta > 0.0 {
                    if let Some((min, max)) = get_range_bound(&ranges, "rope_theta") {
                        if size.rope_theta < min {
                            violations.push(format!(
                                "{label}.rope_theta: {} < min {min}",
                                size.rope_theta
                            ));
                        }
                        if size.rope_theta > max {
                            violations.push(format!(
                                "{label}.rope_theta: {} > max {max}",
                                size.rope_theta
                            ));
                        }
                    }
                }

                // norm_eps
                if size.norm_eps > 0.0 {
                    if let Some((min, max)) = get_range_bound(&ranges, "eps") {
                        if size.norm_eps < min {
                            violations
                                .push(format!("{label}.norm_eps: {} < min {min}", size.norm_eps));
                        }
                        if size.norm_eps > max {
                            violations
                                .push(format!("{label}.norm_eps: {} > max {max}", size.norm_eps));
                        }
                    }
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MB-006: Float ranges violated:\n{}",
            violations.join("\n")
        );
    }
}
