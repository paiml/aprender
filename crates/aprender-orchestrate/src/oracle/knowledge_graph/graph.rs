//! Core Knowledge Graph structure and basic operations
//!
//! Contains the main KnowledgeGraph struct and initialization logic.

use super::super::types::*;
use std::collections::HashMap;

/// Knowledge graph containing all stack components and their relationships
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// All registered components
    pub(crate) components: HashMap<String, StackComponent>,
    /// Capability to component index
    pub(crate) capability_index: HashMap<String, Vec<String>>,
    /// Problem domain to capability mapping
    pub(crate) domain_capabilities: HashMap<ProblemDomain, Vec<String>>,
    /// Integration patterns between components
    pub(crate) integrations: Vec<IntegrationPattern>,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        let mut graph = Self {
            components: HashMap::new(),
            capability_index: HashMap::new(),
            domain_capabilities: HashMap::new(),
            integrations: Vec::new(),
        };
        graph.initialize_domain_mappings();
        graph
    }

    /// Create a knowledge graph pre-populated with the Sovereign AI Stack
    pub fn sovereign_stack() -> Self {
        let mut graph = Self::new();
        graph.register_sovereign_stack();
        graph.register_integration_patterns();
        graph
    }

    /// Register a component and update indices
    pub fn register_component(&mut self, component: StackComponent) {
        let name = component.name.clone();

        // Update capability index
        for cap in &component.capabilities {
            self.capability_index
                .entry(cap.name.clone())
                .or_default()
                .push(name.clone());
        }

        self.components.insert(name, component);
    }
}
