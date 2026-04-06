//! Knowledge Graph for Sovereign AI Stack
//!
//! Provides component registry, capability indexing, and relationship mapping
//! between all stack components.

mod components;
mod domain_mappings;
mod integrations;
mod types;

#[cfg(test)]
mod tests;

pub use types::KnowledgeGraph;
