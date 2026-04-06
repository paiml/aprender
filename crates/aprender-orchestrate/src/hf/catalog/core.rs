//! Core catalog operations
//!
//! Implements HF-QUERY-001, HF-QUERY-004, HF-QUERY-005, HF-QUERY-006
//!
//! ## Observability (HF-OBS-003)
//!
//! Key catalog operations are instrumented with tracing spans:
//! - `hf.catalog.search` - Component search operations
//! - `hf.catalog.by_course` - Course-filtered queries
//! - `hf.catalog.by_category` - Category-filtered queries

use std::collections::HashMap;
use tracing::{debug, info, instrument};

use super::types::{AssetType, CatalogComponent, HfComponentCategory};

// ============================================================================
// HF-QUERY-001: Catalog
// ============================================================================

/// The complete HuggingFace ecosystem catalog
#[derive(Debug, Clone, Default)]
pub struct HfCatalog {
    pub(crate) components: HashMap<String, CatalogComponent>,
}

impl HfCatalog {
    /// Create empty catalog
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the standard catalog with all 50+ components
    pub fn standard() -> Self {
        let mut catalog = Self::new();
        catalog.register_hub_components();
        catalog.register_deployment_components();
        catalog.register_library_components();
        catalog.register_training_components();
        catalog.register_collaboration_components();
        catalog.register_community_components();
        catalog.register_integration_components();
        catalog
    }

    /// Add a component to the catalog
    pub fn add(&mut self, component: CatalogComponent) {
        self.components.insert(component.id.clone(), component);
    }

    /// Get component by ID
    #[instrument(name = "hf.catalog.get", skip(self), fields(found = tracing::field::Empty))]
    pub fn get(&self, id: &str) -> Option<&CatalogComponent> {
        let result = self.components.get(id);
        tracing::Span::current().record("found", result.is_some());
        if result.is_some() {
            debug!(component_id = id, "Retrieved catalog component");
        }
        result
    }

    /// Get all component IDs
    pub fn list(&self) -> Vec<&str> {
        let mut ids: Vec<_> = self.components.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }

    /// Get total component count
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Get all components as an iterator
    pub fn all(&self) -> impl Iterator<Item = &CatalogComponent> {
        self.components.values()
    }

    /// Get components by tag
    #[instrument(name = "hf.catalog.by_tag", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_tag(&self, tag: &str) -> Vec<&CatalogComponent> {
        let tag_lower = tag.to_lowercase();
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.tags.iter().any(|t| t.to_lowercase() == tag_lower))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(tag = tag, count = results.len(), "Tag query completed");
        results
    }

    /// Get components by category
    #[instrument(name = "hf.catalog.by_category", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_category(&self, category: HfComponentCategory) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self.components.values().filter(|c| c.category == category).collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            category = ?category,
            count = results.len(),
            "Category query completed"
        );
        results
    }

    /// Search components by query (matches id, name, description, tags)
    #[instrument(name = "hf.catalog.search", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn search(&self, query: &str) -> Vec<&CatalogComponent> {
        let query_lower = query.to_lowercase();
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| {
                c.id.to_lowercase().contains(&query_lower)
                    || c.name.to_lowercase().contains(&query_lower)
                    || c.description.to_lowercase().contains(&query_lower)
                    || c.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect();
        tracing::Span::current().record("result_count", results.len());
        info!(query = query, count = results.len(), "Catalog search completed");
        results
    }

    // ========================================================================
    // HF-QUERY-004: Course Alignment
    // ========================================================================

    /// Get components for a specific course
    #[instrument(name = "hf.catalog.by_course", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_course(&self, course: u8) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.courses.iter().any(|ca| ca.course == course))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        info!(course = course, count = results.len(), "Course query completed");
        results
    }

    /// Get components for a specific course and week
    #[instrument(name = "hf.catalog.by_course_week", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_course_week(&self, course: u8, week: u8) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.courses.iter().any(|ca| ca.course == course && ca.week == week))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(course = course, week = week, count = results.len(), "Course-week query completed");
        results
    }

    /// Get components by asset type (labs, videos, etc.)
    #[instrument(name = "hf.catalog.by_asset_type", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn by_asset_type(&self, asset: AssetType) -> Vec<&CatalogComponent> {
        let results: Vec<_> = self
            .components
            .values()
            .filter(|c| c.courses.iter().any(|ca| ca.asset_types.contains(&asset)))
            .collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(asset = ?asset, count = results.len(), "Asset type query completed");
        results
    }

    // ========================================================================
    // HF-QUERY-005: Dependency Graph
    // ========================================================================

    /// Get dependencies of a component
    #[instrument(name = "hf.catalog.deps", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn deps(&self, id: &str) -> Vec<&CatalogComponent> {
        let results = self
            .get(id)
            .map(|c| {
                c.dependencies
                    .iter()
                    .filter_map(|dep_id| self.components.get(dep_id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        tracing::Span::current().record("result_count", results.len());
        debug!(component_id = id, dep_count = results.len(), "Dependency lookup completed");
        results
    }

    /// Get reverse dependencies (what depends on this component)
    #[instrument(name = "hf.catalog.rdeps", skip(self), fields(result_count = tracing::field::Empty))]
    pub fn rdeps(&self, id: &str) -> Vec<&CatalogComponent> {
        let results: Vec<_> =
            self.components.values().filter(|c| c.dependencies.contains(&id.to_string())).collect();
        tracing::Span::current().record("result_count", results.len());
        debug!(
            component_id = id,
            rdep_count = results.len(),
            "Reverse dependency lookup completed"
        );
        results
    }

    /// Check if two components are compatible (no conflicts)
    pub fn compatible(&self, id1: &str, id2: &str) -> bool {
        // Simple compatibility: both exist and share common deps or are unrelated
        self.get(id1).is_some() && self.get(id2).is_some()
    }

    // ========================================================================
    // HF-QUERY-006: Documentation Links
    // ========================================================================

    /// Get documentation URL for a component
    pub fn docs_url(&self, id: &str) -> Option<&str> {
        self.get(id).map(|c| c.docs_url.as_str())
    }

    /// Get API reference URL (docs + /api)
    pub fn api_url(&self, id: &str) -> Option<String> {
        self.docs_url(id).map(|url| format!("{}/api", url.trim_end_matches('/')))
    }

    /// Get tutorials URL
    pub fn tutorials_url(&self, id: &str) -> Option<String> {
        self.docs_url(id).map(|url| format!("{}/tutorials", url.trim_end_matches('/')))
    }
}
