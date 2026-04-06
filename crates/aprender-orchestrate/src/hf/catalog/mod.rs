//! HuggingFace Ecosystem Catalog
//!
//! Implements HF-QUERY-001, HF-QUERY-004, HF-QUERY-005, HF-QUERY-006
//!
//! Provides:
//! - Complete 50+ component registry
//! - Course alignment for Coursera specialization
//! - Dependency graph between components
//! - Documentation links
//!
//! ## Observability (HF-OBS-003)
//!
//! Key catalog operations are instrumented with tracing spans:
//! - `hf.catalog.search` - Component search operations
//! - `hf.catalog.by_course` - Course-filtered queries
//! - `hf.catalog.by_category` - Category-filtered queries

// Allow dead_code for methods that are tested but not yet exposed via CLI
#![allow(dead_code, unused_imports)]

mod core;
mod registry_core;
mod registry_extended;
mod types;

#[cfg(test)]
#[allow(non_snake_case)]
mod tests_catalog;

#[cfg(test)]
#[allow(non_snake_case)]
mod tests_types;

// Re-export all public types
pub use core::HfCatalog;
pub use types::{AssetType, CatalogComponent, CourseAlignment, HfComponentCategory};
