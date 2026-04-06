#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::redundant_clone,
    clippy::doc_markdown,
    clippy::unnecessary_debug_formatting
)]
//! Federated Split Coordination Example
//!
//! Demonstrates privacy-preserving distributed data splitting:
//! - Node manifest generation (no raw data shared)
//! - Centralized split planning
//! - Coordinated train/test splits
//! - Different splitting strategies
//!
//! Run with: cargo run --example federated_split

use std::{collections::HashMap, sync::Arc};

use alimentar::{
    ArrowDataset, Dataset, FederatedSplitCoordinator, FederatedSplitStrategy, NodeSplitManifest,
};
use arrow::{
    array::{Float64Array, Int32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_node_dataset(
    node_id: &str,
    num_samples: usize,
    class_bias: i32,
) -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_1", DataType::Float64, false),
        Field::new("feature_2", DataType::Float64, false),
        Field::new("label", DataType::Int32, false),
    ]));

    // Create data with some class imbalance based on node
    let features_1: Vec<f64> = (0..num_samples).map(|i| i as f64 * 0.1).collect();
    let features_2: Vec<f64> = (0..num_samples).map(|i| (i as f64).sin()).collect();

    // Introduce class bias per node to simulate non-IID data
    let labels: Vec<i32> = (0..num_samples)
        .map(|i| (i as i32 + class_bias) % 3)
        .collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(features_1)),
            Arc::new(Float64Array::from(features_2)),
            Arc::new(Int32Array::from(labels)),
        ],
    )?;

    println!("   {} created with {} samples", node_id, num_samples);
    ArrowDataset::from_batch(batch)
}

/// Create a split hash from the split parameters
fn create_split_hash(node_id: &str, seed: u64) -> [u8; 32] {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node_id.hash(&mut hasher);
    seed.hash(&mut hasher);
    let hash = hasher.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&hash.to_le_bytes());
    result
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Federated Split Example ===\n");

    // Simulate 3 data nodes with different amounts of data
    println!("1. Creating node datasets (simulating distributed data)");
    let node_a_data = create_node_dataset("Node A", 1000, 0)?;
    let node_b_data = create_node_dataset("Node B", 800, 1)?;
    let node_c_data = create_node_dataset("Node C", 1200, 2)?;

    // 2. Generate node manifests (this is what nodes share - NO raw data!)
    println!("\n2. Generating node manifests (privacy-preserving)");

    // Create label distributions for each node
    let mut dist_a: HashMap<String, u64> = HashMap::new();
    dist_a.insert("0".to_string(), 334);
    dist_a.insert("1".to_string(), 333);
    dist_a.insert("2".to_string(), 333);

    let manifest_a = NodeSplitManifest {
        node_id: "node_a".to_string(),
        total_rows: node_a_data.len() as u64,
        train_rows: 700,
        test_rows: 300,
        validation_rows: None,
        label_distribution: Some(dist_a),
        split_hash: create_split_hash("node_a", 42),
    };
    println!("   Node A: {} total rows", manifest_a.total_rows);

    let mut dist_b: HashMap<String, u64> = HashMap::new();
    dist_b.insert("0".to_string(), 266);
    dist_b.insert("1".to_string(), 267);
    dist_b.insert("2".to_string(), 267);

    let manifest_b = NodeSplitManifest {
        node_id: "node_b".to_string(),
        total_rows: node_b_data.len() as u64,
        train_rows: 560,
        test_rows: 240,
        validation_rows: None,
        label_distribution: Some(dist_b),
        split_hash: create_split_hash("node_b", 42),
    };
    println!("   Node B: {} total rows", manifest_b.total_rows);

    let mut dist_c: HashMap<String, u64> = HashMap::new();
    dist_c.insert("0".to_string(), 400);
    dist_c.insert("1".to_string(), 400);
    dist_c.insert("2".to_string(), 400);

    let manifest_c = NodeSplitManifest {
        node_id: "node_c".to_string(),
        total_rows: node_c_data.len() as u64,
        train_rows: 840,
        test_rows: 360,
        validation_rows: None,
        label_distribution: Some(dist_c),
        split_hash: create_split_hash("node_c", 42),
    };
    println!("   Node C: {} total rows", manifest_c.total_rows);

    let manifests = vec![manifest_a.clone(), manifest_b.clone(), manifest_c.clone()];
    println!("\n   Created manifests for {} nodes", manifests.len());

    // 3. Coordinator with LocalWithSeed strategy
    println!("\n3. Using LocalWithSeed strategy");

    let strategy = FederatedSplitStrategy::LocalWithSeed {
        seed: 42,
        train_ratio: 0.7,
    };

    let coordinator = FederatedSplitCoordinator::new(strategy);
    let instructions = coordinator.compute_split_plan(&manifests)?;

    println!("   Strategy: LocalWithSeed (70% train, 30% test)");
    println!("   Generated {} split instructions", instructions.len());

    for instruction in &instructions {
        println!(
            "\n   {}: seed={}, train_ratio={:.2}, test_ratio={:.2}",
            instruction.node_id, instruction.seed, instruction.train_ratio, instruction.test_ratio
        );
    }

    // 4. ProportionalIID strategy
    println!("\n4. Using ProportionalIID strategy");

    let proportional_strategy = FederatedSplitStrategy::ProportionalIID { train_ratio: 0.8 };

    let proportional_coordinator = FederatedSplitCoordinator::new(proportional_strategy);
    let proportional_instructions = proportional_coordinator.compute_split_plan(&manifests)?;

    println!("   Strategy: ProportionalIID (80% train, 20% test)");
    for instruction in &proportional_instructions {
        println!(
            "   {}: train_ratio={:.2}, test_ratio={:.2}",
            instruction.node_id, instruction.train_ratio, instruction.test_ratio
        );
    }

    // 5. GlobalStratified strategy
    println!("\n5. Using GlobalStratified strategy");

    let mut target_distribution: HashMap<String, f64> = HashMap::new();
    target_distribution.insert("0".to_string(), 0.33);
    target_distribution.insert("1".to_string(), 0.33);
    target_distribution.insert("2".to_string(), 0.34);

    let stratified_strategy = FederatedSplitStrategy::GlobalStratified {
        label_column: "label".to_string(),
        target_distribution,
    };

    let stratified_coordinator = FederatedSplitCoordinator::new(stratified_strategy);
    let stratified_instructions = stratified_coordinator.compute_split_plan(&manifests)?;

    println!("   Strategy: GlobalStratified (balanced classes)");
    for instruction in &stratified_instructions {
        println!(
            "   {}: train_ratio={:.2}, test_ratio={:.2}",
            instruction.node_id, instruction.train_ratio, instruction.test_ratio
        );
    }

    // 6. Verify split calculations
    println!("\n6. Verifying split calculations");

    let total_rows: u64 = manifests.iter().map(|m| m.total_rows).sum();
    let total_train: u64 = manifests.iter().map(|m| m.train_rows).sum();
    let total_test: u64 = manifests.iter().map(|m| m.test_rows).sum();

    println!("   Total rows across nodes: {}", total_rows);
    println!(
        "   Total train rows: {} ({:.1}%)",
        total_train,
        total_train as f64 / total_rows as f64 * 100.0
    );
    println!(
        "   Total test rows: {} ({:.1}%)",
        total_test,
        total_test as f64 / total_rows as f64 * 100.0
    );

    // 7. Node contribution analysis
    println!("\n7. Node contribution analysis");
    for manifest in &manifests {
        let contribution = manifest.total_rows as f64 / total_rows as f64 * 100.0;
        println!(
            "   {}: {:.1}% of total data",
            manifest.node_id, contribution
        );
    }

    // 8. Label distribution across nodes
    println!("\n8. Label distribution across nodes");
    for manifest in &manifests {
        if let Some(dist) = &manifest.label_distribution {
            println!("   {}:", manifest.node_id);
            for (label, count) in dist {
                let pct = *count as f64 / manifest.total_rows as f64 * 100.0;
                println!("     Label {}: {} ({:.1}%)", label, count, pct);
            }
        }
    }

    // 9. Deterministic splits with same seed
    println!("\n9. Verifying deterministic splits");
    let strategy1 = FederatedSplitStrategy::LocalWithSeed {
        seed: 123,
        train_ratio: 0.8,
    };
    let strategy2 = FederatedSplitStrategy::LocalWithSeed {
        seed: 123,
        train_ratio: 0.8,
    };

    let coord1 = FederatedSplitCoordinator::new(strategy1);
    let coord2 = FederatedSplitCoordinator::new(strategy2);

    let instructions1 = coord1.compute_split_plan(&manifests)?;
    let instructions2 = coord2.compute_split_plan(&manifests)?;

    let same_split = instructions1[0].seed == instructions2[0].seed
        && instructions1[0].train_ratio == instructions2[0].train_ratio;
    println!("   Same seed produces same instructions: {}", same_split);

    // 10. Summary
    println!("\n10. Summary");
    println!("   Total nodes: {}", manifests.len());
    println!("   Total samples: {}", total_rows);
    println!("   Strategies demonstrated: LocalWithSeed, ProportionalIID, GlobalStratified");

    println!("\n=== Example Complete ===");
    Ok(())
}
