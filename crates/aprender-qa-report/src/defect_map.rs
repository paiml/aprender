//! Defect-to-Fixture Mapping (§3.5)
//!
//! Maps `ConversionFailureType` to upstream fixture builders and ticket templates.
//! Enables auto-ticket generation with structured reproduction steps.

use apr_qa_runner::ConversionFailureType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in the defect-fixture map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectFixtureEntry {
    /// Path to the upstream fixture (e.g., "fixtures/tensor_name_mismatch.py")
    pub upstream_fixture: String,
    /// Builder function name for pygmy test generation
    pub pygmy_builder: String,
    /// Markdown template for ticket body (supports `{field}` placeholders)
    pub ticket_template: String,
}

/// Load the defect-fixture map from the embedded YAML
///
/// # Errors
///
/// Returns an error if the YAML fails to parse (should never happen
/// since it's an embedded static file).
pub fn load_defect_fixture_map() -> Result<HashMap<String, DefectFixtureEntry>, String> {
    let yaml = include_str!("defect_fixture_map.yaml");
    serde_yaml::from_str(yaml).map_err(|e| format!("Failed to parse defect fixture map: {e}"))
}

/// Map a `ConversionFailureType` to its defect-fixture map key
#[must_use]
pub fn failure_type_to_key(ft: &ConversionFailureType) -> &'static str {
    ft.key()
}

/// Render a ticket template by substituting `{field}` placeholders
#[must_use]
pub fn render_ticket_template<S: ::std::hash::BuildHasher>(
    template: &str,
    fields: &HashMap<String, String, S>,
) -> String {
    let mut result = template.to_string();
    for (key, value) in fields {
        let placeholder = format!("{{{key}}}");
        result = result.replace(&placeholder, value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the embedded defect fixture map loads with all expected keys
    #[test]
    fn test_load_defect_fixture_map() {
        let map = load_defect_fixture_map().expect("should load embedded map");
        assert!(map.contains_key("tensor_name_mismatch"));
        assert!(map.contains_key("dequantization_failure"));
        assert!(map.contains_key("config_metadata_mismatch"));
        assert!(map.contains_key("missing_artifact"));
        assert!(map.contains_key("inference_failure"));
        assert_eq!(map.len(), 5);
    }

    /// Verify tensor_name_mismatch entry has correct fixture and builder
    #[test]
    fn test_entry_fields_populated() {
        let map = load_defect_fixture_map().expect("load map");
        let entry = map.get("tensor_name_mismatch").expect("should exist");
        assert_eq!(entry.upstream_fixture, "fixtures/tensor_name_mismatch.py");
        assert_eq!(entry.pygmy_builder, "test_tensor_naming_convention");
        assert!(entry.ticket_template.contains("Tensor Name Mismatch"));
    }

    /// Verify dequantization_failure entry has correct fixture path
    #[test]
    fn test_dequantization_entry() {
        let map = load_defect_fixture_map().expect("load map");
        let entry = map.get("dequantization_failure").expect("should exist");
        assert_eq!(
            entry.upstream_fixture,
            "fixtures/dequantization_roundtrip.py"
        );
        assert_eq!(entry.pygmy_builder, "test_dequantization_roundtrip");
        assert!(entry.ticket_template.contains("Dequantization Failure"));
    }

    /// Verify all ConversionFailureType variants map to correct keys
    #[test]
    fn test_failure_type_to_key_mapping() {
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::TensorNameMismatch),
            "tensor_name_mismatch"
        );
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::DequantizationFailure),
            "dequantization_failure"
        );
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::ConfigMetadataMismatch),
            "config_metadata_mismatch"
        );
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::MissingArtifact),
            "missing_artifact"
        );
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::InferenceFailure),
            "inference_failure"
        );
        assert_eq!(
            failure_type_to_key(&ConversionFailureType::Unknown),
            "unknown"
        );
    }

    /// Verify basic template rendering substitutes all placeholders
    #[test]
    fn test_render_ticket_template_basic() {
        let template = "Model: {model_id}, Error: {error}";
        let mut fields = HashMap::new();
        fields.insert("model_id".to_string(), "test/model".to_string());
        fields.insert("error".to_string(), "tensor mismatch".to_string());

        let rendered = render_ticket_template(template, &fields);
        assert_eq!(rendered, "Model: test/model, Error: tensor mismatch");
    }

    /// Verify unknown placeholders are preserved in rendered output
    #[test]
    fn test_render_ticket_template_missing_field() {
        let template = "Model: {model_id}, Missing: {unknown}";
        let mut fields = HashMap::new();
        fields.insert("model_id".to_string(), "test/model".to_string());

        let rendered = render_ticket_template(template, &fields);
        // Unknown fields left as-is
        assert_eq!(rendered, "Model: test/model, Missing: {unknown}");
    }

    /// Verify full template rendering with real defect fixture map entry
    #[test]
    fn test_render_ticket_template_full() {
        let map = load_defect_fixture_map().expect("load map");
        let entry = map.get("tensor_name_mismatch").expect("should exist");

        let mut fields = HashMap::new();
        fields.insert("source_naming".to_string(), "HuggingFace".to_string());
        fields.insert("target_naming".to_string(), "GGUF".to_string());
        fields.insert("affected_count".to_string(), "42".to_string());
        fields.insert("model_id".to_string(), "test/model".to_string());
        fields.insert("model_path".to_string(), "/models/test".to_string());

        let rendered = render_ticket_template(&entry.ticket_template, &fields);
        assert!(rendered.contains("HuggingFace"));
        assert!(rendered.contains("GGUF"));
        assert!(rendered.contains("42"));
        assert!(rendered.contains("test/model"));
    }

    /// Verify DefectFixtureEntry survives JSON serialization roundtrip
    #[test]
    fn test_defect_fixture_entry_serde_roundtrip() {
        let entry = DefectFixtureEntry {
            upstream_fixture: "fixtures/test.py".to_string(),
            pygmy_builder: "test_builder".to_string(),
            ticket_template: "## Title\n\nBody {field}".to_string(),
        };

        let json = serde_json::to_string(&entry).expect("serialize");
        let deserialized: DefectFixtureEntry = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.upstream_fixture, entry.upstream_fixture);
        assert_eq!(deserialized.pygmy_builder, entry.pygmy_builder);
        assert_eq!(deserialized.ticket_template, entry.ticket_template);
    }
}
