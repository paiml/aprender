//! RO-Crate metadata descriptor.

use super::entity::{EntityType, RoCrateEntity};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// JSON-LD context for RO-Crate 1.1
pub const RO_CRATE_CONTEXT: &str = "https://w3id.org/ro/crate/1.1/context";

/// RO-Crate metadata descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoCrateDescriptor {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: String,
    /// Graph of entities
    #[serde(rename = "@graph")]
    pub graph: Vec<RoCrateEntity>,
}

impl Default for RoCrateDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl RoCrateDescriptor {
    /// Create a new RO-Crate descriptor
    pub fn new() -> Self {
        // Start with the metadata file entity pointing to root
        let metadata_entity =
            RoCrateEntity::new("ro-crate-metadata.json", EntityType::CreativeWork)
                .with_property("conformsTo", json!({ "@id": "https://w3id.org/ro/crate/1.1" }))
                .with_reference("about", "./");

        Self { context: RO_CRATE_CONTEXT.to_string(), graph: vec![metadata_entity] }
    }

    /// Add an entity to the graph
    pub fn add_entity(&mut self, entity: RoCrateEntity) {
        self.graph.push(entity);
    }

    /// Get the root dataset entity
    pub fn root_dataset(&self) -> Option<&RoCrateEntity> {
        self.graph.iter().find(|e| e.id == "./")
    }

    /// Get the root dataset entity mutably
    pub fn root_dataset_mut(&mut self) -> Option<&mut RoCrateEntity> {
        self.graph.iter_mut().find(|e| e.id == "./")
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_err| "{}".to_string())
    }
}
