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
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_find() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("whisper-large".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("registration failed");

        let nodes = catalog
            .find_by_capability(&Capability::Transcribe)
            .await
            .expect("find failed");

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, NodeId("node-1".to_string()));
    }

    #[tokio::test]
    async fn test_deregister() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("llama-7b".to_string()),
                NodeId("node-1".to_string()),
                RegionId("eu-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        catalog
            .deregister(
                ModelId("llama-7b".to_string()),
                NodeId("node-1".to_string()),
            )
            .await
            .expect("deregistration failed");

        let models = catalog.list_all().await.expect("list failed");
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_deployments() {
        let catalog = ModelCatalog::new();

        // Same model on two nodes
        catalog
            .register(
                ModelId("whisper-base".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-east".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("registration failed");

        catalog
            .register(
                ModelId("whisper-base".to_string()),
                NodeId("node-2".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("registration failed");

        let nodes = catalog
            .find_by_capability(&Capability::Transcribe)
            .await
            .expect("find failed");

        assert_eq!(nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_custom_capability() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("sentiment-bert".to_string()),
                NodeId("node-1".to_string()),
                RegionId("ap-south".to_string()),
                vec![Capability::Custom("sentiment".to_string())],
            )
            .await
            .expect("registration failed");

        let nodes = catalog
            .find_by_capability(&Capability::Custom("sentiment".to_string()))
            .await
            .expect("find failed");

        assert_eq!(nodes.len(), 1);

        // Different custom capability should return empty
        let empty = catalog
            .find_by_capability(&Capability::Custom("other".to_string()))
            .await
            .expect("find failed");

        assert!(empty.is_empty());
    }

    // =========================================================================
    // ModelCatalog::get tests
    // =========================================================================

    #[tokio::test]
    async fn test_get_existing_model() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("whisper".to_string()),
                NodeId("n1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("registration failed");

        let entry = catalog.get(&ModelId("whisper".to_string()));
        assert!(entry.is_some());
        let entry = entry.expect("entry should exist");
        assert_eq!(entry.model_id, ModelId("whisper".to_string()));
        assert_eq!(entry.deployments.len(), 1);
    }

    #[test]
    fn test_get_nonexistent_model() {
        let catalog = ModelCatalog::new();
        let entry = catalog.get(&ModelId("nonexistent".to_string()));
        assert!(entry.is_none());
    }

    // =========================================================================
    // get_metadata tests
    // =========================================================================

    #[tokio::test]
    async fn test_get_metadata_existing() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("llama".to_string()),
                NodeId("n1".to_string()),
                RegionId("us-east".to_string()),
                vec![Capability::Generate, Capability::Code],
            )
            .await
            .expect("registration failed");

        let meta = catalog
            .get_metadata(&ModelId("llama".to_string()))
            .await
            .expect("metadata failed");

        assert_eq!(meta.model_id, ModelId("llama".to_string()));
        assert_eq!(meta.name, "llama");
        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.capabilities.len(), 2);
    }

    #[tokio::test]
    async fn test_get_metadata_nonexistent() {
        let catalog = ModelCatalog::new();

        let result = catalog.get_metadata(&ModelId("missing".to_string())).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FederationError::Internal(_)));
    }

    // =========================================================================
    // all_entries tests
    // =========================================================================

    #[test]
    fn test_all_entries_empty() {
        let catalog = ModelCatalog::new();
        assert!(catalog.all_entries().is_empty());
    }

    #[tokio::test]
    async fn test_all_entries_multiple() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n1".to_string()),
                RegionId("r1".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("failed");

        catalog
            .register(
                ModelId("m2".to_string()),
                NodeId("n2".to_string()),
                RegionId("r2".to_string()),
                vec![Capability::Embed],
            )
            .await
            .expect("failed");

        let entries = catalog.all_entries();
        assert_eq!(entries.len(), 2);
    }

    // =========================================================================
    // deregister edge cases
    // =========================================================================

    #[tokio::test]
    async fn test_deregister_nonexistent_model() {
        let catalog = ModelCatalog::new();

        // Deregistering a non-existent model should succeed (no-op)
        let result = catalog
            .deregister(ModelId("missing".to_string()), NodeId("n1".to_string()))
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deregister_nonexistent_node() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n1".to_string()),
                RegionId("r1".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("failed");

        // Deregister a different node -> model should still exist
        catalog
            .deregister(ModelId("m1".to_string()), NodeId("n2".to_string()))
            .await
            .expect("deregister failed");

        let models = catalog.list_all().await.expect("list failed");
        assert_eq!(models.len(), 1);
    }

    #[tokio::test]
    async fn test_deregister_partial_keeps_remaining() {
        let catalog = ModelCatalog::new();

        // Same model on two nodes
        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n1".to_string()),
                RegionId("r1".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("failed");

        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n2".to_string()),
                RegionId("r2".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("failed");

        // Deregister one node
        catalog
            .deregister(ModelId("m1".to_string()), NodeId("n1".to_string()))
            .await
            .expect("deregister failed");

        // Model should still exist with 1 deployment
        let entry = catalog.get(&ModelId("m1".to_string()));
        assert!(entry.is_some());
        assert_eq!(entry.expect("should exist").deployments.len(), 1);
    }

    // =========================================================================
    // capability_key coverage
    // =========================================================================

    #[tokio::test]
    async fn test_all_capability_keys_via_registration() {
        let catalog = ModelCatalog::new();

        // Register one model for each capability variant
        let capabilities = vec![
            (ModelId("t1".to_string()), Capability::Transcribe),
            (ModelId("t2".to_string()), Capability::Synthesize),
            (ModelId("t3".to_string()), Capability::Generate),
            (ModelId("t4".to_string()), Capability::Code),
            (ModelId("t5".to_string()), Capability::Embed),
            (ModelId("t6".to_string()), Capability::ImageGen),
            (
                ModelId("t7".to_string()),
                Capability::Custom("custom_task".to_string()),
            ),
        ];

        for (model_id, cap) in &capabilities {
            catalog
                .register(
                    model_id.clone(),
                    NodeId("n1".to_string()),
                    RegionId("r1".to_string()),
                    vec![cap.clone()],
                )
                .await
                .expect("registration failed");
        }

        // Verify each can be found
        for (_, cap) in &capabilities {
            let nodes = catalog.find_by_capability(cap).await.expect("find failed");
            assert_eq!(nodes.len(), 1, "Should find 1 node for {:?}", cap);
        }
    }

    // =========================================================================
    // DeploymentStatus tests
    // =========================================================================

    #[test]
    fn test_deployment_status_equality() {
        assert_eq!(DeploymentStatus::Ready, DeploymentStatus::Ready);
        assert_ne!(DeploymentStatus::Ready, DeploymentStatus::Loading);
        assert_ne!(DeploymentStatus::Draining, DeploymentStatus::Removed);
    }

    #[test]
    fn test_deployment_status_all_variants() {
        let statuses = [
            DeploymentStatus::Loading,
            DeploymentStatus::Ready,
            DeploymentStatus::Draining,
            DeploymentStatus::Removed,
        ];
        // All distinct
        for (i, a) in statuses.iter().enumerate() {
            for (j, b) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_deployment_status_copy() {
        let status = DeploymentStatus::Draining;
        let copied = status;
        assert_eq!(status, copied);
    }

    // =========================================================================
    // ModelEntry/ModelDeployment construction tests
    // =========================================================================

    #[test]
    fn test_model_entry_clone() {
        let entry = ModelEntry {
            model_id: ModelId("test".to_string()),
            metadata: ModelMetadata {
                model_id: ModelId("test".to_string()),
                name: "Test Model".to_string(),
                version: "1.0".to_string(),
                capabilities: vec![Capability::Generate],
                parameters: 7_000_000_000,
                quantization: Some("Q4_K".to_string()),
            },
            deployments: vec![ModelDeployment {
                node_id: NodeId("n1".to_string()),
                region_id: RegionId("us-west".to_string()),
                endpoint: "http://n1:8080".to_string(),
                status: DeploymentStatus::Ready,
            }],
        };

        let cloned = entry.clone();
        assert_eq!(cloned.model_id, ModelId("test".to_string()));
        assert_eq!(cloned.deployments.len(), 1);
    }

    #[test]
    fn test_model_deployment_construction() {
        let dep = ModelDeployment {
            node_id: NodeId("gpu-node".to_string()),
            region_id: RegionId("eu-west".to_string()),
            endpoint: "https://gpu-node.eu-west:443".to_string(),
            status: DeploymentStatus::Loading,
        };
        assert_eq!(dep.node_id, NodeId("gpu-node".to_string()));
        assert_eq!(dep.status, DeploymentStatus::Loading);
    }

    // =========================================================================
    // ModelCatalog::default tests
    // =========================================================================

    #[test]
    fn test_model_catalog_default() {
        let catalog = ModelCatalog::default();
        assert!(catalog.all_entries().is_empty());
    }

    // =========================================================================
    // find_by_capability with non-Ready deployments
    // =========================================================================

    #[tokio::test]
    async fn test_find_by_capability_empty() {
        let catalog = ModelCatalog::new();
        let nodes = catalog
            .find_by_capability(&Capability::Generate)
            .await
            .expect("find failed");
        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_find_by_capability_no_match() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("whisper".to_string()),
                NodeId("n1".to_string()),
                RegionId("r1".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("failed");

        // Search for different capability
        let nodes = catalog
            .find_by_capability(&Capability::Generate)
            .await
            .expect("find failed");
        assert!(nodes.is_empty());
    }

    // =========================================================================
    // list_all tests
    // =========================================================================

    #[tokio::test]
    async fn test_list_all_empty() {
        let catalog = ModelCatalog::new();
        let models = catalog.list_all().await.expect("list failed");
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_list_all_after_deregister_all() {
        let catalog = ModelCatalog::new();

        catalog
            .register(
                ModelId("m1".to_string()),
                NodeId("n1".to_string()),
                RegionId("r1".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("failed");

        catalog
            .deregister(ModelId("m1".to_string()), NodeId("n1".to_string()))
            .await
            .expect("failed");

        let models = catalog.list_all().await.expect("list failed");
        assert!(models.is_empty());
    }
}
