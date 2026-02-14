//! Model Catalog - Registry of available models across the federation
//!
//! The catalog tracks which models are available, where they're deployed,
//! and what capabilities they support.

use super::traits::*;
use std::collections::HashMap;
use std::sync::RwLock;

// ============================================================================
// Model Entry
// ============================================================================

/// Entry for a registered model in the catalog
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub model_id: ModelId,
    pub metadata: ModelMetadata,
    pub deployments: Vec<ModelDeployment>,
}

/// A specific deployment of a model
#[derive(Debug, Clone)]
pub struct ModelDeployment {
    pub node_id: NodeId,
    pub region_id: RegionId,
    pub endpoint: String,
    pub status: DeploymentStatus,
}

/// Deployment status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentStatus {
    /// Model is loading
    Loading,
    /// Model is ready for inference
    Ready,
    /// Model is draining (no new requests)
    Draining,
    /// Model has been removed
    Removed,
}

// ============================================================================
// In-Memory Catalog Implementation
// ============================================================================

/// In-memory model catalog (production would use etcd, Redis, etc.)
pub struct ModelCatalog {
    entries: RwLock<HashMap<ModelId, ModelEntry>>,
    by_capability: RwLock<HashMap<String, Vec<ModelId>>>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            by_capability: RwLock::new(HashMap::new()),
        }
    }

    /// Get all entries (for debugging/admin)
    pub fn all_entries(&self) -> Vec<ModelEntry> {
        self.entries
            .read()
            .expect("catalog lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Get entry by ID
    pub fn get(&self, model_id: &ModelId) -> Option<ModelEntry> {
        self.entries
            .read()
            .expect("catalog lock poisoned")
            .get(model_id)
            .cloned()
    }

    fn capability_key(cap: &Capability) -> String {
        match cap {
            Capability::Transcribe => "transcribe".to_string(),
            Capability::Synthesize => "synthesize".to_string(),
            Capability::Generate => "generate".to_string(),
            Capability::Code => "code".to_string(),
            Capability::Embed => "embed".to_string(),
            Capability::ImageGen => "image_gen".to_string(),
            Capability::Custom(s) => format!("custom:{}", s),
        }
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelCatalogTrait for ModelCatalog {
    fn register(
        &self,
        model_id: ModelId,
        node_id: NodeId,
        region_id: RegionId,
        capabilities: Vec<Capability>,
    ) -> BoxFuture<'_, FederationResult<()>> {
        Box::pin(async move {
            let mut entries = self.entries.write().expect("catalog lock poisoned");
            let mut by_cap = self.by_capability.write().expect("catalog lock poisoned");

            let deployment = ModelDeployment {
                node_id,
                region_id,
                endpoint: String::new(), // Would be set by registration protocol
                status: DeploymentStatus::Ready,
            };

            if let Some(entry) = entries.get_mut(&model_id) {
                // Add deployment to existing model
                entry.deployments.push(deployment);
            } else {
                // New model registration
                let metadata = ModelMetadata {
                    model_id: model_id.clone(),
                    name: model_id.0.clone(),
                    version: "1.0.0".to_string(),
                    capabilities: capabilities.clone(),
                    parameters: 0,
                    quantization: None,
                };

                let entry = ModelEntry {
                    model_id: model_id.clone(),
                    metadata,
                    deployments: vec![deployment],
                };

                entries.insert(model_id.clone(), entry);

                // Index by capability
                for cap in &capabilities {
                    let key = Self::capability_key(cap);
                    by_cap.entry(key).or_default().push(model_id.clone());
                }
            }

            Ok(())
        })
    }

    fn deregister(
        &self,
        model_id: ModelId,
        node_id: NodeId,
    ) -> BoxFuture<'_, FederationResult<()>> {
        Box::pin(async move {
            let mut entries = self.entries.write().expect("catalog lock poisoned");

            if let Some(entry) = entries.get_mut(&model_id) {
                entry.deployments.retain(|d| d.node_id != node_id);

                // Remove entry entirely if no deployments remain
                if entry.deployments.is_empty() {
                    entries.remove(&model_id);
                }
            }

            Ok(())
        })
    }

    fn find_by_capability(
        &self,
        capability: &Capability,
    ) -> BoxFuture<'_, FederationResult<Vec<(NodeId, RegionId)>>> {
        let key = Self::capability_key(capability);

        Box::pin(async move {
            let entries = self.entries.read().expect("catalog lock poisoned");
            let by_cap = self.by_capability.read().expect("catalog lock poisoned");

            let mut results = Vec::new();

            if let Some(model_ids) = by_cap.get(&key) {
                for model_id in model_ids {
                    if let Some(entry) = entries.get(model_id) {
                        for deployment in &entry.deployments {
                            if deployment.status == DeploymentStatus::Ready {
                                results.push((
                                    deployment.node_id.clone(),
                                    deployment.region_id.clone(),
                                ));
                            }
                        }
                    }
                }
            }

            Ok(results)
        })
    }

    fn list_all(&self) -> BoxFuture<'_, FederationResult<Vec<ModelId>>> {
        Box::pin(async move {
            let entries = self.entries.read().expect("catalog lock poisoned");
            Ok(entries.keys().cloned().collect())
        })
    }

    fn get_metadata(&self, model_id: &ModelId) -> BoxFuture<'_, FederationResult<ModelMetadata>> {
        let model_id = model_id.clone();

        Box::pin(async move {
            let entries = self.entries.read().expect("catalog lock poisoned");

            entries
                .get(&model_id)
                .map(|e| e.metadata.clone())
                .ok_or_else(|| {
                    FederationError::Internal(format!("Model not found: {:?}", model_id))
                })
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "catalog_tests.rs"]
mod tests;
