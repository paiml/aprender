//! Tests for HuggingFace Ecosystem Catalog types
//!
//! Extreme TDD with 95%+ Coverage

use super::*;

// ========================================================================
// HF-QUERY-001-001: HfComponentCategory Tests
// ========================================================================

#[test]
fn test_HF_QUERY_001_001_category_all() {
    let categories = HfComponentCategory::all();
    assert_eq!(categories.len(), 6);
    assert!(categories.contains(&HfComponentCategory::Hub));
    assert!(categories.contains(&HfComponentCategory::Deployment));
    assert!(categories.contains(&HfComponentCategory::Library));
    assert!(categories.contains(&HfComponentCategory::Training));
    assert!(categories.contains(&HfComponentCategory::Collaboration));
    assert!(categories.contains(&HfComponentCategory::Community));
}

#[test]
fn test_HF_QUERY_001_002_category_display_name() {
    assert_eq!(HfComponentCategory::Hub.display_name(), "Hub & Client Libraries");
    assert_eq!(HfComponentCategory::Deployment.display_name(), "Deployment & Inference");
    assert_eq!(HfComponentCategory::Library.display_name(), "Core ML Libraries");
    assert_eq!(HfComponentCategory::Training.display_name(), "Training & Optimization");
    assert_eq!(HfComponentCategory::Collaboration.display_name(), "Collaboration & Extras");
    assert_eq!(HfComponentCategory::Community.display_name(), "Community Resources");
}

#[test]
fn test_HF_QUERY_001_003_category_display() {
    assert_eq!(format!("{}", HfComponentCategory::Hub), "Hub & Client Libraries");
}

// ========================================================================
// HF-QUERY-001-010: CourseAlignment Tests
// ========================================================================

#[test]
fn test_HF_QUERY_001_010_course_alignment_new() {
    let ca = CourseAlignment::new(1, 2);
    assert_eq!(ca.course, 1);
    assert_eq!(ca.week, 2);
    assert!(ca.lessons.is_empty());
    assert!(ca.asset_types.is_empty());
}

#[test]
fn test_HF_QUERY_001_011_course_alignment_with_lessons() {
    let ca = CourseAlignment::new(1, 1).with_lessons(&["1.1", "1.2"]);
    assert_eq!(ca.lessons.len(), 2);
    assert_eq!(ca.lessons[0], "1.1");
}

#[test]
fn test_HF_QUERY_001_012_course_alignment_with_assets() {
    let ca = CourseAlignment::new(1, 1).with_assets(&[AssetType::Video, AssetType::Lab]);
    assert_eq!(ca.asset_types.len(), 2);
    assert!(ca.asset_types.contains(&AssetType::Video));
}

// ========================================================================
// HF-QUERY-001-020: CatalogComponent Tests
// ========================================================================

#[test]
fn test_HF_QUERY_001_020_component_new() {
    let comp = CatalogComponent::new("transformers", "Transformers", HfComponentCategory::Library);
    assert_eq!(comp.id, "transformers");
    assert_eq!(comp.name, "Transformers");
    assert_eq!(comp.category, HfComponentCategory::Library);
}

#[test]
fn test_HF_QUERY_001_021_component_with_description() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_description("A test component");
    assert_eq!(comp.description, "A test component");
}

#[test]
fn test_HF_QUERY_001_022_component_with_docs() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_docs("https://example.com");
    assert_eq!(comp.docs_url, "https://example.com");
}

#[test]
fn test_HF_QUERY_001_023_component_with_repo() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_repo("https://github.com/test");
    assert_eq!(comp.repo_url, Some("https://github.com/test".to_string()));
}

#[test]
fn test_HF_QUERY_001_024_component_with_pypi() {
    let comp =
        CatalogComponent::new("test", "Test", HfComponentCategory::Hub).with_pypi("test-package");
    assert_eq!(comp.pypi_name, Some("test-package".to_string()));
}

#[test]
fn test_HF_QUERY_001_025_component_with_npm() {
    let comp =
        CatalogComponent::new("test", "Test", HfComponentCategory::Hub).with_npm("@test/package");
    assert_eq!(comp.npm_name, Some("@test/package".to_string()));
}

#[test]
fn test_HF_QUERY_001_026_component_with_deps() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_deps(&["dep1", "dep2"]);
    assert_eq!(comp.dependencies.len(), 2);
}

#[test]
fn test_HF_QUERY_001_027_component_with_course() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_course(CourseAlignment::new(1, 1));
    assert_eq!(comp.courses.len(), 1);
}

#[test]
fn test_HF_QUERY_001_028_component_with_related() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_related(&["related1", "related2"]);
    assert_eq!(comp.related.len(), 2);
}

#[test]
fn test_HF_QUERY_001_029_component_with_tags() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_tags(&["tag1", "tag2"]);
    assert_eq!(comp.tags.len(), 2);
}

// ========================================================================
// HF-QUERY-001-050: Serialization Tests
// ========================================================================

#[test]
fn test_HF_QUERY_001_050_component_serialize_json() {
    let comp = CatalogComponent::new("test", "Test", HfComponentCategory::Hub)
        .with_description("Test component");
    let json = serde_json::to_string(&comp).unwrap();
    assert!(json.contains("\"id\":\"test\""));
    assert!(json.contains("\"category\":\"hub\""));
}

#[test]
fn test_HF_QUERY_001_051_component_deserialize_json() {
    let json = r#"{"id":"test","name":"Test","category":"hub","description":"","docs_url":"","repo_url":null,"pypi_name":null,"npm_name":null,"dependencies":[],"courses":[],"related":[],"tags":[]}"#;
    let comp: CatalogComponent = serde_json::from_str(json).unwrap();
    assert_eq!(comp.id, "test");
    assert_eq!(comp.category, HfComponentCategory::Hub);
}

#[test]
fn test_HF_QUERY_001_052_course_alignment_serialize() {
    let ca = CourseAlignment::new(1, 2).with_lessons(&["1.1"]).with_assets(&[AssetType::Video]);
    let json = serde_json::to_string(&ca).unwrap();
    assert!(json.contains("\"course\":1"));
    assert!(json.contains("\"week\":2"));
}
