# Federated & Splitting (Examples 66-75)

This section covers dataset splitting for ML and federated learning.

## Examples 66-67: Train/Test and Stratified Split

```rust
use alimentar::{ArrowDataset, DatasetSplit};

let dataset = ArrowDataset::from_parquet("data.parquet")?;

// Basic 80/20 split
let split = DatasetSplit::from_ratios(
    &dataset,
    0.8,      // train
    0.2,      // test
    None,     // no validation
    Some(42)  // seed for reproducibility
)?;

// Stratified by label column
let split = DatasetSplit::stratified(
    &dataset,
    "label",  // stratify column
    0.8, 0.2, None,
    Some(42)
)?;

assert_eq!(split.train().len() + split.test().len(), dataset.len());
```

## Examples 68-69: K-Fold and Leave-One-Out

```rust
use alimentar::DatasetSplit;

// 5-fold cross-validation
let folds = DatasetSplit::kfold(&dataset, 5, Some(42))?;
for (i, (train, test)) in folds.iter().enumerate() {
    println!("Fold {}: train={}, test={}", i, train.len(), test.len());
}

// Leave-one-out
let loo = DatasetSplit::leave_one_out(&dataset)?;
```

## Examples 70-71: Node Manifest and Coordinator

```rust
use alimentar::{DatasetSplit, NodeSplitManifest, FederatedCoordinator};

let split = DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(42))?;
let manifest = NodeSplitManifest::from_split("node1", &split);

println!("Node: {}", manifest.node_id);
println!("Train rows: {}", manifest.train_rows);
println!("Test rows: {}", manifest.test_rows);

// Coordinator aggregates manifests
let coordinator = FederatedCoordinator::new();
coordinator.register_node(manifest)?;
```

## Examples 72-74: IID/Non-IID/Dirichlet Strategies

```rust
use alimentar::{FederatedSplit, PartitionStrategy};

// IID (random) partitioning
let splits = FederatedSplit::partition(
    &dataset,
    10, // 10 nodes
    PartitionStrategy::IID,
    Some(42)
)?;

// Non-IID (label-skewed)
let splits = FederatedSplit::partition(
    &dataset,
    10,
    PartitionStrategy::NonIID { skew: 0.5 },
    Some(42)
)?;

// Dirichlet distribution
let splits = FederatedSplit::partition(
    &dataset,
    10,
    PartitionStrategy::Dirichlet { alpha: 0.5 },
    Some(42)
)?;
```

## Example 75: Multi-Node Simulation

```rust
use alimentar::{FederatedSplit, FederatedCoordinator};

let coordinator = FederatedCoordinator::new();

// Distribute to 10 simulated nodes
let splits = FederatedSplit::partition(&dataset, 10,
    PartitionStrategy::IID, Some(42))?;

for (i, split) in splits.iter().enumerate() {
    let manifest = NodeSplitManifest::from_split(
        &format!("node_{}", i),
        split
    );
    coordinator.register_node(manifest)?;
}

// Verify distribution
let stats = coordinator.distribution_stats()?;
println!("Total: {} rows across {} nodes", stats.total_rows, stats.node_count);
```

## CLI Usage

```bash
# Basic split
alimentar fed split data.parquet --train 0.8 --test 0.2

# Stratified split
alimentar fed split data.parquet --stratify label --train 0.8 --test 0.2

# Create node manifest
alimentar fed manifest data.parquet --node-id node1

# Plan federated distribution
alimentar fed plan --nodes 10 --strategy iid data.parquet

# Verify manifests
alimentar fed verify manifest1.json manifest2.json
```

## Key Concepts

- **Reproducibility**: Seed ensures same split
- **Stratification**: Preserves class distribution
- **Manifest**: Metadata about node's data
- **Coordinator**: Central aggregation point
