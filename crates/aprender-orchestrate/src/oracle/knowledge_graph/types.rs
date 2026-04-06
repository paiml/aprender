//! Core types and methods for the knowledge graph.

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
            self.capability_index.entry(cap.name.clone()).or_default().push(name.clone());
        }

        self.components.insert(name, component);
    }

    /// Get a component by name
    pub fn get_component(&self, name: &str) -> Option<&StackComponent> {
        self.components.get(name)
    }

    /// Get all components
    pub fn components(&self) -> impl Iterator<Item = &StackComponent> {
        self.components.values()
    }

    /// Get all component names
    pub fn component_names(&self) -> impl Iterator<Item = &String> {
        self.components.keys()
    }

    /// Get components in a specific layer
    pub fn components_in_layer(&self, layer: StackLayer) -> Vec<&StackComponent> {
        self.components.values().filter(|c| c.layer == layer).collect()
    }

    /// Find components with a specific capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<&StackComponent> {
        self.capability_index
            .get(capability)
            .map(|names| names.iter().filter_map(|n| self.components.get(n)).collect())
            .unwrap_or_default()
    }

    /// Find components for a problem domain
    pub fn find_by_domain(&self, domain: ProblemDomain) -> Vec<&StackComponent> {
        self.domain_capabilities
            .get(&domain)
            .map(|caps| {
                caps.iter().flat_map(|cap| self.find_by_capability(cap)).collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Get integration patterns from a component
    pub fn integrations_from(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations.iter().filter(|p| p.from == component).collect()
    }

    /// Get integration patterns to a component
    pub fn integrations_to(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations.iter().filter(|p| p.to == component).collect()
    }

    /// Get integration pattern between two components
    pub fn get_integration(&self, from: &str, to: &str) -> Option<&IntegrationPattern> {
        self.integrations.iter().find(|p| p.from == from && p.to == to)
    }

    /// Get all capabilities in the graph
    pub fn all_capabilities(&self) -> impl Iterator<Item = &String> {
        self.capability_index.keys()
    }

    /// Get total number of components
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get total number of capabilities
    pub fn capability_count(&self) -> usize {
        self.capability_index.len()
    }

    /// Get total number of integration patterns
    pub fn integration_count(&self) -> usize {
        self.integrations.len()
    }
}
