//! Federated Split Coordination for Privacy-Preserving ML
//!
//! This module enables distributed/federated ML workflows where data stays
//! local on each node (sovereignty) and only metadata/sketches cross
//! boundaries.
//!
//! # Architecture
//!
//! ```text
//! Node A (EU):     data_eu.ald      → local train_eu.ald, test_eu.ald
//! Node B (US):     data_us.ald      → local train_us.ald, test_us.ald
//! Node C (APAC):   data_apac.ald    → local train_apac.ald, test_apac.ald
//!                        ↓
//!              Coordinator (sees only manifests)
//!                        ↓
//!              Global split verification
//! ```

// Federated coordination uses ratio calculations where precision loss is acceptable
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::{Error, Result},
    split::DatasetSplit,
};

/// Federated split coordination (no raw data leaves nodes)
#[derive(Debug, Clone)]
pub struct FederatedSplitCoordinator {
    /// Strategy for distributed splitting
    strategy: FederatedSplitStrategy,
}

/// Strategy for federated/distributed splitting
#[derive(Debug, Clone, PartialEq)]
pub enum FederatedSplitStrategy {
    /// Each node splits locally with same seed (simple, no coordination)
    LocalWithSeed {
        /// Random seed for reproducibility
        seed: u64,
        /// Training set ratio (0.0 to 1.0)
        train_ratio: f64,
    },

    /// Stratified across nodes - coordinator sees only label distributions
    GlobalStratified {
        /// Column containing class labels
        label_column: String,
        /// Target distribution (coordinator computes from sketches)
        target_distribution: HashMap<String, f64>,
    },

    /// IID sampling - each node contributes proportionally
    ProportionalIID {
        /// Training set ratio
        train_ratio: f64,
    },
}

/// Per-node split manifest (shared with coordinator, no raw data)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSplitManifest {
    /// Unique node identifier
    pub node_id: String,
    /// Total rows in dataset
    pub total_rows: u64,
    /// Rows in training split
    pub train_rows: u64,
    /// Rows in test split
    pub test_rows: u64,
    /// Rows in validation split (optional)
    pub validation_rows: Option<u64>,
    /// Label distribution (for stratification verification)
    pub label_distribution: Option<HashMap<String, u64>>,
    /// Hash of split indices (for reproducibility verification)
    pub split_hash: [u8; 32],
}

/// Instructions for a node to execute its split
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSplitInstruction {
    /// Node this instruction is for
    pub node_id: String,
    /// Random seed to use
    pub seed: u64,
    /// Training ratio
    pub train_ratio: f64,
    /// Test ratio
    pub test_ratio: f64,
    /// Validation ratio (optional)
    pub validation_ratio: Option<f64>,
    /// Column to stratify by (optional)
    pub stratify_column: Option<String>,
}

/// Report on global split quality across all nodes
#[derive(Debug, Clone)]
pub struct GlobalSplitReport {
    /// Total rows across all nodes
    pub total_rows: u64,
    /// Total training rows
    pub total_train_rows: u64,
    /// Total test rows
    pub total_test_rows: u64,
    /// Total validation rows
    pub total_validation_rows: Option<u64>,
    /// Effective train/test/val ratios
    pub effective_train_ratio: f64,
    /// Effective test ratio
    pub effective_test_ratio: f64,
    /// Effective validation ratio
    pub effective_validation_ratio: Option<f64>,
    /// Per-node summaries
    pub node_summaries: Vec<NodeSummary>,
    /// Global label distribution (if stratified)
    pub global_label_distribution: Option<HashMap<String, u64>>,
    /// Whether split quality meets requirements
    pub quality_passed: bool,
    /// Quality issues found
    pub issues: Vec<SplitQualityIssue>,
}

/// Summary for a single node
#[derive(Debug, Clone)]
pub struct NodeSummary {
    /// Node ID
    pub node_id: String,
    /// Contribution ratio (this node's rows / total rows)
    pub contribution_ratio: f64,
    /// Train ratio for this node
    pub train_ratio: f64,
    /// Test ratio for this node
    pub test_ratio: f64,
}

/// Quality issues that can be detected in federated splits
#[derive(Debug, Clone, PartialEq)]
pub enum SplitQualityIssue {
    /// Train/test ratio differs too much from target
    RatioDeviation {
        /// Node with the issue
        node_id: String,
        /// Expected ratio
        expected: f64,
        /// Actual ratio
        actual: f64,
    },
    /// Label distribution imbalance across nodes
    DistributionImbalance {
        /// Label with imbalance
        label: String,
        /// Nodes with significantly different distributions
        nodes: Vec<String>,
    },
    /// Node has too few samples
    InsufficientSamples {
        /// Node ID
        node_id: String,
        /// Number of samples
        samples: u64,
        /// Minimum required
        minimum: u64,
    },
    /// Split hashes don't match expected (reproducibility issue)
    HashMismatch {
        /// Node ID
        node_id: String,
    },
}

impl FederatedSplitCoordinator {
    /// Create a new coordinator with the given strategy
    #[must_use]
    pub fn new(strategy: FederatedSplitStrategy) -> Self {
        Self { strategy }
    }

    /// Get the current strategy
    #[must_use]
    pub fn strategy(&self) -> &FederatedSplitStrategy {
        &self.strategy
    }

    /// Compute split instructions for each node (runs on coordinator)
    ///
    /// The coordinator only sees manifests (metadata), never raw data.
    pub fn compute_split_plan(
        &self,
        manifests: &[NodeSplitManifest],
    ) -> Result<Vec<NodeSplitInstruction>> {
        if manifests.is_empty() {
            return Err(Error::invalid_config(
                "Cannot compute plan for empty manifest list",
            ));
        }

        match &self.strategy {
            FederatedSplitStrategy::LocalWithSeed { seed, train_ratio } => Ok(
                Self::compute_local_seed_plan(manifests, *seed, *train_ratio),
            ),
            FederatedSplitStrategy::GlobalStratified {
                label_column,
                target_distribution,
            } => Ok(Self::compute_stratified_plan(
                manifests,
                label_column,
                target_distribution,
            )),
            FederatedSplitStrategy::ProportionalIID { train_ratio } => {
                Ok(Self::compute_proportional_plan(manifests, *train_ratio))
            }
        }
    }

    /// Compute plan for LocalWithSeed strategy
    fn compute_local_seed_plan(
        manifests: &[NodeSplitManifest],
        seed: u64,
        train_ratio: f64,
    ) -> Vec<NodeSplitInstruction> {
        let test_ratio = 1.0 - train_ratio;

        manifests
            .iter()
            .map(|m| NodeSplitInstruction {
                node_id: m.node_id.clone(),
                seed,
                train_ratio,
                test_ratio,
                validation_ratio: None,
                stratify_column: None,
            })
            .collect()
    }

    /// Compute plan for GlobalStratified strategy
    fn compute_stratified_plan(
        manifests: &[NodeSplitManifest],
        label_column: &str,
        _target_distribution: &HashMap<String, f64>,
    ) -> Vec<NodeSplitInstruction> {
        // For stratified splits, each node uses a deterministic seed based on node_id
        // and stratifies by the label column
        let base_seed = 42u64; // Fixed base for reproducibility

        // Default to 80/20 split if not specified in target distribution
        // Future: use target_distribution to adjust per-node ratios
        let train_ratio = 0.8;
        let test_ratio = 0.2;

        manifests
            .iter()
            .enumerate()
            .map(|(i, m)| {
                // Derive node-specific seed from base + index
                let node_seed = base_seed.wrapping_add(i as u64);

                NodeSplitInstruction {
                    node_id: m.node_id.clone(),
                    seed: node_seed,
                    train_ratio,
                    test_ratio,
                    validation_ratio: None,
                    stratify_column: Some(label_column.to_string()),
                }
            })
            .collect()
    }

    /// Compute plan for ProportionalIID strategy
    fn compute_proportional_plan(
        manifests: &[NodeSplitManifest],
        train_ratio: f64,
    ) -> Vec<NodeSplitInstruction> {
        let test_ratio = 1.0 - train_ratio;

        // Each node gets a unique seed based on position
        manifests
            .iter()
            .enumerate()
            .map(|(i, m)| NodeSplitInstruction {
                node_id: m.node_id.clone(),
                seed: i as u64,
                train_ratio,
                test_ratio,
                validation_ratio: None,
                stratify_column: None,
            })
            .collect()
    }

    /// Execute split locally (runs on each node)
    ///
    /// This function runs on the data-owning node - raw data never leaves.
    pub fn execute_local_split(
        dataset: &ArrowDataset,
        instruction: &NodeSplitInstruction,
    ) -> Result<DatasetSplit> {
        let val_ratio = instruction.validation_ratio;

        if let Some(ref column) = instruction.stratify_column {
            DatasetSplit::stratified(
                dataset,
                column,
                instruction.train_ratio,
                instruction.test_ratio,
                val_ratio,
                Some(instruction.seed),
            )
        } else {
            DatasetSplit::from_ratios(
                dataset,
                instruction.train_ratio,
                instruction.test_ratio,
                val_ratio,
                Some(instruction.seed),
            )
        }
    }

    /// Verify global split quality (runs on coordinator)
    ///
    /// Only examines manifests - no access to raw data.
    pub fn verify_global_split(manifests: &[NodeSplitManifest]) -> Result<GlobalSplitReport> {
        if manifests.is_empty() {
            return Err(Error::invalid_config("Cannot verify empty manifest list"));
        }

        let total_rows: u64 = manifests.iter().map(|m| m.total_rows).sum();
        let total_train_rows: u64 = manifests.iter().map(|m| m.train_rows).sum();
        let total_test_rows: u64 = manifests.iter().map(|m| m.test_rows).sum();
        let total_validation_rows: Option<u64> =
            if manifests.iter().any(|m| m.validation_rows.is_some()) {
                Some(manifests.iter().filter_map(|m| m.validation_rows).sum())
            } else {
                None
            };

        let effective_train_ratio = if total_rows > 0 {
            total_train_rows as f64 / total_rows as f64
        } else {
            0.0
        };

        let effective_test_ratio = if total_rows > 0 {
            total_test_rows as f64 / total_rows as f64
        } else {
            0.0
        };

        let effective_validation_ratio = total_validation_rows.map(|v| {
            if total_rows > 0 {
                v as f64 / total_rows as f64
            } else {
                0.0
            }
        });

        // Build node summaries
        let node_summaries: Vec<NodeSummary> = manifests
            .iter()
            .map(|m| {
                let contribution_ratio = if total_rows > 0 {
                    m.total_rows as f64 / total_rows as f64
                } else {
                    0.0
                };

                let train_ratio = if m.total_rows > 0 {
                    m.train_rows as f64 / m.total_rows as f64
                } else {
                    0.0
                };

                let test_ratio = if m.total_rows > 0 {
                    m.test_rows as f64 / m.total_rows as f64
                } else {
                    0.0
                };

                NodeSummary {
                    node_id: m.node_id.clone(),
                    contribution_ratio,
                    train_ratio,
                    test_ratio,
                }
            })
            .collect();

        // Aggregate global label distribution
        let global_label_distribution = Self::aggregate_label_distributions(manifests);

        // Check for quality issues
        let issues = Self::detect_quality_issues(manifests, &node_summaries);

        let quality_passed = issues.is_empty();

        Ok(GlobalSplitReport {
            total_rows,
            total_train_rows,
            total_test_rows,
            total_validation_rows,
            effective_train_ratio,
            effective_test_ratio,
            effective_validation_ratio,
            node_summaries,
            global_label_distribution,
            quality_passed,
            issues,
        })
    }

    /// Aggregate label distributions from all nodes
    fn aggregate_label_distributions(
        manifests: &[NodeSplitManifest],
    ) -> Option<HashMap<String, u64>> {
        let mut global_dist: HashMap<String, u64> = HashMap::new();
        let mut any_has_distribution = false;

        for manifest in manifests {
            if let Some(ref dist) = manifest.label_distribution {
                any_has_distribution = true;
                for (label, count) in dist {
                    *global_dist.entry(label.clone()).or_insert(0) += count;
                }
            }
        }

        if any_has_distribution {
            Some(global_dist)
        } else {
            None
        }
    }

    /// Detect quality issues in the split
    fn detect_quality_issues(
        manifests: &[NodeSplitManifest],
        summaries: &[NodeSummary],
    ) -> Vec<SplitQualityIssue> {
        // Minimum samples threshold for valid split
        const MIN_SAMPLES: u64 = 10;

        let mut issues = Vec::new();

        // Check for insufficient samples
        for manifest in manifests {
            if manifest.train_rows < MIN_SAMPLES || manifest.test_rows < MIN_SAMPLES {
                issues.push(SplitQualityIssue::InsufficientSamples {
                    node_id: manifest.node_id.clone(),
                    samples: manifest.train_rows.min(manifest.test_rows),
                    minimum: MIN_SAMPLES,
                });
            }
        }

        // Check for ratio deviation (more than 10% from mean)
        if !summaries.is_empty() {
            let mean_train_ratio: f64 =
                summaries.iter().map(|s| s.train_ratio).sum::<f64>() / summaries.len() as f64;

            for summary in summaries {
                let deviation = (summary.train_ratio - mean_train_ratio).abs();
                if deviation > 0.1 {
                    issues.push(SplitQualityIssue::RatioDeviation {
                        node_id: summary.node_id.clone(),
                        expected: mean_train_ratio,
                        actual: summary.train_ratio,
                    });
                }
            }
        }

        issues
    }
}

impl NodeSplitManifest {
    /// Create a new manifest from split results
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        total_rows: u64,
        train_rows: u64,
        test_rows: u64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            total_rows,
            train_rows,
            test_rows,
            validation_rows: None,
            label_distribution: None,
            split_hash: [0u8; 32],
        }
    }

    /// Set validation rows
    #[must_use]
    pub fn with_validation(mut self, rows: u64) -> Self {
        self.validation_rows = Some(rows);
        self
    }

    /// Set label distribution
    #[must_use]
    pub fn with_label_distribution(mut self, distribution: HashMap<String, u64>) -> Self {
        self.label_distribution = Some(distribution);
        self
    }

    /// Set split hash
    #[must_use]
    pub fn with_split_hash(mut self, hash: [u8; 32]) -> Self {
        self.split_hash = hash;
        self
    }

    /// Create manifest from a dataset split
    #[must_use]
    pub fn from_split(node_id: impl Into<String>, split: &DatasetSplit) -> Self {
        let train_rows = split.train.len() as u64;
        let test_rows = split.test.len() as u64;
        let validation_rows = split.validation.as_ref().map(|v| v.len() as u64);

        let mut manifest = Self::new(
            node_id,
            train_rows + test_rows + validation_rows.unwrap_or(0),
            train_rows,
            test_rows,
        );

        if let Some(v) = validation_rows {
            manifest = manifest.with_validation(v);
        }

        manifest
    }

    /// Serialize to JSON bytes
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::Format(e.to_string()))
    }

    /// Deserialize from JSON bytes
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| Error::Format(e.to_string()))
    }
}

impl NodeSplitInstruction {
    /// Serialize to JSON bytes
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::Format(e.to_string()))
    }

    /// Deserialize from JSON bytes
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| Error::Format(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // FederatedSplitStrategy tests
    // ============================================================

    #[test]
    fn test_strategy_local_with_seed() {
        let strategy = FederatedSplitStrategy::LocalWithSeed {
            seed: 42,
            train_ratio: 0.8,
        };

        match strategy {
            FederatedSplitStrategy::LocalWithSeed { seed, train_ratio } => {
                assert_eq!(seed, 42);
                assert!((train_ratio - 0.8).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_strategy_global_stratified() {
        let mut target = HashMap::new();
        target.insert("class_a".to_string(), 0.5);
        target.insert("class_b".to_string(), 0.5);

        let strategy = FederatedSplitStrategy::GlobalStratified {
            label_column: "label".to_string(),
            target_distribution: target.clone(),
        };

        match strategy {
            FederatedSplitStrategy::GlobalStratified {
                label_column,
                target_distribution,
            } => {
                assert_eq!(label_column, "label");
                assert_eq!(target_distribution, target);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_strategy_proportional_iid() {
        let strategy = FederatedSplitStrategy::ProportionalIID { train_ratio: 0.7 };

        match strategy {
            FederatedSplitStrategy::ProportionalIID { train_ratio } => {
                assert!((train_ratio - 0.7).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_strategy_clone_and_debug() {
        let strategy = FederatedSplitStrategy::LocalWithSeed {
            seed: 42,
            train_ratio: 0.8,
        };

        let cloned = strategy.clone();
        assert_eq!(strategy, cloned);

        let debug = format!("{:?}", strategy);
        assert!(debug.contains("LocalWithSeed"));
        assert!(debug.contains("42"));
    }

    // ============================================================
    // NodeSplitManifest tests
    // ============================================================

    #[test]
    fn test_manifest_new() {
        let manifest = NodeSplitManifest::new("node_a", 1000, 800, 200);

        assert_eq!(manifest.node_id, "node_a");
        assert_eq!(manifest.total_rows, 1000);
        assert_eq!(manifest.train_rows, 800);
        assert_eq!(manifest.test_rows, 200);
        assert!(manifest.validation_rows.is_none());
        assert!(manifest.label_distribution.is_none());
    }

    #[test]
    fn test_manifest_with_validation() {
        let manifest = NodeSplitManifest::new("node_a", 1000, 700, 200).with_validation(100);

        assert_eq!(manifest.validation_rows, Some(100));
    }

    #[test]
    fn test_manifest_with_label_distribution() {
        let mut dist = HashMap::new();
        dist.insert("cat".to_string(), 500);
        dist.insert("dog".to_string(), 500);

        let manifest =
            NodeSplitManifest::new("node_a", 1000, 800, 200).with_label_distribution(dist.clone());

        assert_eq!(manifest.label_distribution, Some(dist));
    }

    #[test]
    fn test_manifest_with_split_hash() {
        let hash = [1u8; 32];
        let manifest = NodeSplitManifest::new("node_a", 1000, 800, 200).with_split_hash(hash);

        assert_eq!(manifest.split_hash, hash);
    }

    #[test]
    fn test_manifest_serialization() {
        let manifest = NodeSplitManifest::new("node_a", 1000, 800, 200);

        let json = manifest.to_json().expect("serialization failed");
        let parsed = NodeSplitManifest::from_json(&json).expect("deserialization failed");

        assert_eq!(manifest, parsed);
    }

    #[test]
    fn test_manifest_full_serialization() {
        let mut dist = HashMap::new();
        dist.insert("a".to_string(), 400);
        dist.insert("b".to_string(), 600);

        let manifest = NodeSplitManifest::new("node_eu", 1000, 700, 200)
            .with_validation(100)
            .with_label_distribution(dist)
            .with_split_hash([42u8; 32]);

        let json = manifest.to_json().expect("serialization failed");
        let parsed = NodeSplitManifest::from_json(&json).expect("deserialization failed");

        assert_eq!(manifest, parsed);
    }

    // ============================================================
    // NodeSplitInstruction tests
    // ============================================================

    #[test]
    fn test_instruction_serialization() {
        let instruction = NodeSplitInstruction {
            node_id: "node_a".to_string(),
            seed: 42,
            train_ratio: 0.8,
            test_ratio: 0.2,
            validation_ratio: None,
            stratify_column: None,
        };

        let json = instruction.to_json().expect("serialization failed");
        let parsed = NodeSplitInstruction::from_json(&json).expect("deserialization failed");

        assert_eq!(instruction, parsed);
    }

    #[test]
    fn test_instruction_with_stratification() {
        let instruction = NodeSplitInstruction {
            node_id: "node_b".to_string(),
            seed: 123,
            train_ratio: 0.7,
            test_ratio: 0.15,
            validation_ratio: Some(0.15),
            stratify_column: Some("label".to_string()),
        };

        let json = instruction.to_json().expect("serialization failed");
        let parsed = NodeSplitInstruction::from_json(&json).expect("deserialization failed");

        assert_eq!(instruction, parsed);
    }

    // ============================================================
    // FederatedSplitCoordinator tests
    // ============================================================

    #[test]
    fn test_coordinator_new() {
        let strategy = FederatedSplitStrategy::LocalWithSeed {
            seed: 42,
            train_ratio: 0.8,
        };
        let coordinator = FederatedSplitCoordinator::new(strategy.clone());

        assert_eq!(coordinator.strategy(), &strategy);
    }

    #[test]
    fn test_coordinator_empty_manifests_error() {
        let coordinator = FederatedSplitCoordinator::new(FederatedSplitStrategy::LocalWithSeed {
            seed: 42,
            train_ratio: 0.8,
        });

        let result = coordinator.compute_split_plan(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_coordinator_local_seed_plan() {
        let coordinator = FederatedSplitCoordinator::new(FederatedSplitStrategy::LocalWithSeed {
            seed: 42,
            train_ratio: 0.8,
        });

        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200),
            NodeSplitManifest::new("node_b", 2000, 1600, 400),
        ];

        let plan = coordinator
            .compute_split_plan(&manifests)
            .expect("plan failed");

        assert_eq!(plan.len(), 2);

        // All nodes get same seed
        assert_eq!(plan[0].seed, 42);
        assert_eq!(plan[1].seed, 42);

        // All nodes get same ratios
        assert!((plan[0].train_ratio - 0.8).abs() < f64::EPSILON);
        assert!((plan[1].train_ratio - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coordinator_stratified_plan() {
        let mut target = HashMap::new();
        target.insert("a".to_string(), 0.5);
        target.insert("b".to_string(), 0.5);

        let coordinator =
            FederatedSplitCoordinator::new(FederatedSplitStrategy::GlobalStratified {
                label_column: "label".to_string(),
                target_distribution: target,
            });

        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200),
            NodeSplitManifest::new("node_b", 2000, 1600, 400),
        ];

        let plan = coordinator
            .compute_split_plan(&manifests)
            .expect("plan failed");

        assert_eq!(plan.len(), 2);

        // Each node has stratify column set
        assert_eq!(plan[0].stratify_column, Some("label".to_string()));
        assert_eq!(plan[1].stratify_column, Some("label".to_string()));

        // Nodes have different seeds (derived from position)
        assert_ne!(plan[0].seed, plan[1].seed);
    }

    #[test]
    fn test_coordinator_proportional_plan() {
        let coordinator = FederatedSplitCoordinator::new(FederatedSplitStrategy::ProportionalIID {
            train_ratio: 0.7,
        });

        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 700, 300),
            NodeSplitManifest::new("node_b", 2000, 1400, 600),
            NodeSplitManifest::new("node_c", 500, 350, 150),
        ];

        let plan = coordinator
            .compute_split_plan(&manifests)
            .expect("plan failed");

        assert_eq!(plan.len(), 3);

        // All nodes get same ratio
        for instruction in &plan {
            assert!((instruction.train_ratio - 0.7).abs() < f64::EPSILON);
            assert!((instruction.test_ratio - 0.3).abs() < f64::EPSILON);
        }

        // Each node has unique seed
        assert_eq!(plan[0].seed, 0);
        assert_eq!(plan[1].seed, 1);
        assert_eq!(plan[2].seed, 2);
    }

    // ============================================================
    // GlobalSplitReport tests
    // ============================================================

    #[test]
    fn test_verify_global_split_empty_error() {
        let result = FederatedSplitCoordinator::verify_global_split(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_global_split_single_node() {
        let manifests = vec![NodeSplitManifest::new("node_a", 1000, 800, 200)];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        assert_eq!(report.total_rows, 1000);
        assert_eq!(report.total_train_rows, 800);
        assert_eq!(report.total_test_rows, 200);
        assert!((report.effective_train_ratio - 0.8).abs() < f64::EPSILON);
        assert!((report.effective_test_ratio - 0.2).abs() < f64::EPSILON);
        assert!(report.quality_passed);
    }

    #[test]
    fn test_verify_global_split_multiple_nodes() {
        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200),
            NodeSplitManifest::new("node_b", 2000, 1600, 400),
            NodeSplitManifest::new("node_c", 1000, 800, 200),
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        assert_eq!(report.total_rows, 4000);
        assert_eq!(report.total_train_rows, 3200);
        assert_eq!(report.total_test_rows, 800);
        assert!((report.effective_train_ratio - 0.8).abs() < f64::EPSILON);

        assert_eq!(report.node_summaries.len(), 3);
        assert!(report.quality_passed);
    }

    #[test]
    fn test_verify_global_split_with_validation() {
        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 700, 200).with_validation(100),
            NodeSplitManifest::new("node_b", 2000, 1400, 400).with_validation(200),
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        assert_eq!(report.total_validation_rows, Some(300));
        assert!((report.effective_validation_ratio.unwrap() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verify_global_split_aggregates_labels() {
        let mut dist_a = HashMap::new();
        dist_a.insert("cat".to_string(), 600);
        dist_a.insert("dog".to_string(), 400);

        let mut dist_b = HashMap::new();
        dist_b.insert("cat".to_string(), 800);
        dist_b.insert("dog".to_string(), 1200);

        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200).with_label_distribution(dist_a),
            NodeSplitManifest::new("node_b", 2000, 1600, 400).with_label_distribution(dist_b),
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        let global_dist = report
            .global_label_distribution
            .expect("should have distribution");
        assert_eq!(global_dist.get("cat"), Some(&1400));
        assert_eq!(global_dist.get("dog"), Some(&1600));
    }

    #[test]
    fn test_verify_detects_insufficient_samples() {
        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200),
            NodeSplitManifest::new("node_b", 15, 10, 5), // Too few test samples
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        assert!(!report.quality_passed);
        assert!(!report.issues.is_empty());

        let has_insufficient = report.issues.iter().any(|i| {
            matches!(i, SplitQualityIssue::InsufficientSamples { node_id, .. } if node_id == "node_b")
        });
        assert!(has_insufficient);
    }

    #[test]
    fn test_verify_detects_ratio_deviation() {
        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200), // 80/20
            NodeSplitManifest::new("node_b", 1000, 500, 500), // 50/50 - big deviation
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        assert!(!report.quality_passed);

        let has_deviation = report
            .issues
            .iter()
            .any(|i| matches!(i, SplitQualityIssue::RatioDeviation { .. }));
        assert!(has_deviation);
    }

    #[test]
    fn test_node_summary_contribution_ratio() {
        let manifests = vec![
            NodeSplitManifest::new("node_a", 1000, 800, 200),
            NodeSplitManifest::new("node_b", 3000, 2400, 600),
        ];

        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        // node_a has 1000/4000 = 0.25
        assert!((report.node_summaries[0].contribution_ratio - 0.25).abs() < f64::EPSILON);

        // node_b has 3000/4000 = 0.75
        assert!((report.node_summaries[1].contribution_ratio - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_split_quality_issue_variants() {
        let ratio_issue = SplitQualityIssue::RatioDeviation {
            node_id: "node_a".to_string(),
            expected: 0.8,
            actual: 0.5,
        };
        assert!(format!("{:?}", ratio_issue).contains("RatioDeviation"));

        let dist_issue = SplitQualityIssue::DistributionImbalance {
            label: "cat".to_string(),
            nodes: vec!["node_a".to_string(), "node_b".to_string()],
        };
        assert!(format!("{:?}", dist_issue).contains("DistributionImbalance"));

        let sample_issue = SplitQualityIssue::InsufficientSamples {
            node_id: "node_a".to_string(),
            samples: 5,
            minimum: 10,
        };
        assert!(format!("{:?}", sample_issue).contains("InsufficientSamples"));

        let hash_issue = SplitQualityIssue::HashMismatch {
            node_id: "node_a".to_string(),
        };
        assert!(format!("{:?}", hash_issue).contains("HashMismatch"));
    }

    #[test]
    fn test_global_split_report_debug() {
        let manifests = vec![NodeSplitManifest::new("node_a", 100, 80, 20)];
        let report =
            FederatedSplitCoordinator::verify_global_split(&manifests).expect("verify failed");

        let debug = format!("{:?}", report);
        assert!(debug.contains("GlobalSplitReport"));
        assert!(debug.contains("total_rows"));
    }
}
