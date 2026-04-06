//! WASM Serve Module - Browser-based data serving and sharing
//!
//! This module provides functionality for serving datasets, courses, and other
//! content types through WASM-based browser applications with optional P2P
//! sharing.
//!
//! # Design Principles
//!
//! 1. **Browser-First** - Full functionality in WASM without server
//!    dependencies
//! 2. **Plugin Architecture** - Extensible type system for arbitrary content
//! 3. **Zero-Server Option** - P2P sharing via WebRTC
//! 4. **Schema-Driven** - YAML/JSON configuration for type definitions
//!
//! # Example
//!
//! ```ignore
//! use alimentar::serve::{PluginRegistry, ContentTypeId};
//!
//! let registry = PluginRegistry::new();
//! let plugin = registry.get(&ContentTypeId::dataset()).unwrap();
//! ```

// Allow builder patterns without must_use (common pattern for fluent APIs)
#![allow(clippy::return_self_not_must_use)]

mod content;
mod plugin;
mod raw_source;
mod schema;

pub use content::{ContentMetadata, ContentTypeId, ServeableContent, ValidationReport};
pub use plugin::{ContentPlugin, DatasetPlugin, PluginRegistry, RenderHints};
pub use raw_source::{RawSource, RawSourceConfig, SourceType};
pub use schema::{Constraint, ContentSchema, FieldDefinition, FieldType, ValidatorDefinition};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();
        assert!(registry.get(&ContentTypeId::dataset()).is_some());
    }

    #[test]
    fn test_content_type_id_constants() {
        assert_eq!(ContentTypeId::dataset().as_str(), "alimentar.dataset");
        assert_eq!(ContentTypeId::course().as_str(), "assetgen.course");
        assert_eq!(ContentTypeId::model().as_str(), "aprender.model");
        assert_eq!(ContentTypeId::registry().as_str(), "alimentar.registry");
        assert_eq!(ContentTypeId::raw().as_str(), "alimentar.raw");
    }
}
