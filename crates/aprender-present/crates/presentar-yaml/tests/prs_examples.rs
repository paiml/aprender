//! Integration tests for .prs example files.
//!
//! This test validates all .prs files in the examples/prs directory,
//! ensuring they parse correctly and pass validation.

#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)]

use presentar_yaml::Scene;
use std::fs;
use std::path::Path;

/// Load and validate all .prs files in the examples/prs directory.
#[test]
fn test_all_prs_examples_valid() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent dir")
        .parent()
        .expect("workspace dir")
        .join("examples")
        .join("prs");

    assert!(
        examples_dir.exists(),
        "Examples directory not found: {}",
        examples_dir.display()
    );

    let mut tested = 0;
    let mut errors = Vec::new();

    for entry in fs::read_dir(&examples_dir).expect("Failed to read examples directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "prs") {
            let filename = path.file_name().expect("filename").to_string_lossy();
            let content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {filename}: {e}"));

            match Scene::from_yaml(&content) {
                Ok(scene) => {
                    // Additional validation checks
                    assert!(
                        !scene.metadata.name.is_empty(),
                        "{filename}: metadata.name is empty"
                    );
                    assert!(
                        !scene.prs_version.is_empty(),
                        "{filename}: prs_version is empty"
                    );
                    tested += 1;
                    println!("OK: {filename}");
                }
                Err(e) => {
                    errors.push(format!("{filename}: {e}"));
                }
            }
        }
    }

    assert!(
        errors.is_empty(),
        "Failed to parse {} .prs file(s):\n{}",
        errors.len(),
        errors.join("\n")
    );

    assert!(tested > 0, "No .prs files found in examples/prs directory");
    println!("\nValidated {tested} .prs example files");
}

/// Test individual example files for specific properties.
mod individual_examples {
    use super::*;

    fn load_example(name: &str) -> Scene {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("parent dir")
            .parent()
            .expect("workspace dir")
            .join("examples")
            .join("prs")
            .join(name);

        let content =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {name}: {e}"));

        Scene::from_yaml(&content).unwrap_or_else(|e| panic!("Failed to parse {name}: {e}"))
    }

    #[test]
    fn test_minimal_example() {
        let scene = load_example("minimal.prs");
        assert_eq!(scene.metadata.name, "hello-world");
        assert_eq!(scene.widgets.len(), 1);
    }

    #[test]
    fn test_sentiment_demo_has_resources() {
        let scene = load_example("sentiment-demo.prs");
        assert_eq!(scene.metadata.name, "sentiment-analysis-demo");
        assert!(!scene.resources.models.is_empty());
        assert!(scene.resources.models.contains_key("sentiment_model"));
    }

    #[test]
    fn test_image_classifier_has_bindings() {
        let scene = load_example("image-classifier.prs");
        assert_eq!(scene.metadata.name, "image-classifier");
        assert!(!scene.bindings.is_empty());
    }

    #[test]
    fn test_data_explorer_has_dataset() {
        let scene = load_example("data-explorer.prs");
        assert_eq!(scene.metadata.name, "data-explorer");
        assert!(scene.resources.datasets.contains_key("sales"));
    }

    #[test]
    fn test_parameter_tuner_has_sliders() {
        let scene = load_example("parameter-tuner.prs");
        assert_eq!(scene.metadata.name, "parameter-tuner");

        // Check that we have slider widgets
        let slider_count = scene
            .widgets
            .iter()
            .filter(|w| w.widget_type == presentar_yaml::WidgetType::Slider)
            .count();

        assert_eq!(slider_count, 3, "Expected 3 slider widgets");
    }
}

/// QA Checklist validation for .prs files.
mod qa_checklist {
    use super::*;

    /// Validates .prs files against QA checklist criteria.
    fn validate_qa_checklist(scene: &Scene, filename: &str) -> Vec<String> {
        let mut issues = Vec::new();

        // 1. Version format
        if !scene.prs_version.contains('.') {
            issues.push(format!("{filename}: prs_version should be semver format"));
        }

        // 2. Metadata name is kebab-case
        if scene.metadata.name.chars().any(|c| c.is_ascii_uppercase()) {
            issues.push(format!("{filename}: metadata.name should be kebab-case"));
        }

        // 3. Widget IDs are unique (already validated by parser, but double-check)
        let mut ids = std::collections::HashSet::new();
        for widget in &scene.widgets {
            if !ids.insert(&widget.id) {
                issues.push(format!("{filename}: duplicate widget id '{}'", widget.id));
            }
        }

        // 4. Remote resources have hashes
        for (name, model) in &scene.resources.models {
            if model.source.primary().starts_with("https://") && model.hash.is_none() {
                issues.push(format!("{filename}: remote model '{name}' missing hash"));
            }
        }

        // 5. Theme preset is valid if specified
        if let Some(theme) = &scene.theme {
            if let Some(preset) = &theme.preset {
                if !["light", "dark"].contains(&preset.as_str()) {
                    issues.push(format!("{filename}: unknown theme preset '{preset}'"));
                }
            }
        }

        issues
    }

    #[test]
    fn test_all_examples_pass_qa_checklist() {
        let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("parent dir")
            .parent()
            .expect("workspace dir")
            .join("examples")
            .join("prs");

        let mut all_issues = Vec::new();

        for entry in fs::read_dir(&examples_dir).expect("Failed to read examples directory") {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "prs") {
                let filename = path.file_name().expect("filename").to_string_lossy();
                let content = fs::read_to_string(&path).expect("Failed to read file");

                if let Ok(scene) = Scene::from_yaml(&content) {
                    let issues = validate_qa_checklist(&scene, &filename);
                    all_issues.extend(issues);
                }
            }
        }

        assert!(
            all_issues.is_empty(),
            "QA checklist failures:\n{}",
            all_issues.join("\n")
        );
    }
}
