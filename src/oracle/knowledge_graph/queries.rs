//! Query operations for the Knowledge Graph
//!
//! Methods for searching and retrieving components, capabilities, and integrations.

use super::super::types::*;
use super::graph::KnowledgeGraph;

impl KnowledgeGraph {
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
        self.components
            .values()
            .filter(|c| c.layer == layer)
            .collect()
    }

    /// Find components with a specific capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<&StackComponent> {
        self.capability_index
            .get(capability)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|n| self.components.get(n))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find components for a problem domain
    pub fn find_by_domain(&self, domain: ProblemDomain) -> Vec<&StackComponent> {
        self.domain_capabilities
            .get(&domain)
            .map(|caps| {
                caps.iter()
                    .flat_map(|cap| self.find_by_capability(cap))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Get integration patterns from a component
    pub fn integrations_from(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations
            .iter()
            .filter(|p| p.from == component)
            .collect()
    }

    /// Get integration patterns to a component
    pub fn integrations_to(&self, component: &str) -> Vec<&IntegrationPattern> {
        self.integrations
            .iter()
            .filter(|p| p.to == component)
            .collect()
    }

    /// Get integration pattern between two components
    pub fn get_integration(&self, from: &str, to: &str) -> Option<&IntegrationPattern> {
        self.integrations
            .iter()
            .find(|p| p.from == from && p.to == to)
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
