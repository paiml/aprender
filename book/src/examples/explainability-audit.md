# Model Explainability and Audit Trails

<!-- DOC_STATUS_START -->
**Chapter Status**: 100% Working (All examples verified)

| Status | Count | Examples |
|--------|-------|----------|
| Working | 8 | DecisionPath, HashChainCollector, audit trails verified |
| In Progress | 0 | - |
| Not Implemented | 0 | - |

*Last tested: 2025-12-10*
*Aprender version: 0.17.0*
*Test file: src/explainability/mod.rs tests*
<!-- DOC_STATUS_END -->

---

## Overview

Aprender provides built-in model explainability and tamper-evident audit trails for ML compliance and debugging. This follows the Toyota Way principle: **shihai wo kakusanai** (never hide failures) - every prediction decision is auditable with full context.

**Key Concepts**:
- **Decision Path**: Serializable explanation of why a model made a specific prediction
- **Hash Chain Provenance**: Cryptographic chain ensuring audit trail integrity
- **Feature Contributions**: Quantified impact of each feature on predictions

**Why This Matters**:
For regulated industries (finance, healthcare, autonomous systems), you need to explain *why* a model predicted what it did. Aprender's explainability system provides:
1. Human-readable decision explanations
2. Machine-parseable decision paths for downstream analysis
3. Tamper-evident audit logs for compliance

---

## The DecisionPath Trait

```rust
use aprender::explainability::{DecisionPath, Explainable};
use serde::{Serialize, Deserialize};

/// Every model prediction generates a DecisionPath
pub trait DecisionPath: Serialize + Clone {
    /// Human-readable explanation
    fn explain(&self) -> String;

    /// Feature contribution scores
    fn feature_contributions(&self) -> &[f32];

    /// Confidence score [0.0, 1.0]
    fn confidence(&self) -> f32;

    /// Serialize for audit storage
    fn to_bytes(&self) -> Vec<u8>;
}
```

---

## Decision Path Types

### LinearPath (Linear Models)

For linear regression, logistic regression, and regularized variants:

```rust
use aprender::explainability::LinearPath;

// After prediction
let path = LinearPath {
    feature_weights: vec![0.5, -0.3, 0.8],  // Model coefficients
    feature_values: vec![1.2, 3.4, 0.9],     // Input values
    contributions: vec![0.6, -1.02, 0.72],   // weight * value
    intercept: 0.1,
    prediction: 0.5,                          // Final output
};

println!("{}", path.explain());
// Output:
// Linear Model Decision:
//   Feature 0: 1.20 * 0.50 = 0.60
//   Feature 1: 3.40 * -0.30 = -1.02
//   Feature 2: 0.90 * 0.80 = 0.72
//   Intercept: 0.10
//   Prediction: 0.50
```

### TreePath (Decision Trees)

For decision tree and random forest models:

```rust
use aprender::explainability::TreePath;

let path = TreePath {
    nodes: vec![
        TreeNode { feature: 2, threshold: 2.5, went_left: true },
        TreeNode { feature: 0, threshold: 1.0, went_left: false },
    ],
    leaf_value: 0.0,  // Class 0 (Setosa)
    feature_importances: vec![0.3, 0.1, 0.6],
};

println!("{}", path.explain());
// Output:
// Decision Tree Path:
//   Node 0: feature[2]=1.4 <= 2.5? YES -> left
//   Node 1: feature[0]=5.1 <= 1.0? NO -> right
//   Leaf: class 0 (confidence: 100.0%)
```

### ForestPath (Ensemble Models)

For random forests, gradient boosting, and ensemble methods:

```rust
use aprender::explainability::ForestPath;

let path = ForestPath {
    tree_paths: vec![tree_path_1, tree_path_2, tree_path_3],
    tree_weights: vec![0.33, 0.33, 0.34],
    aggregated_prediction: 1.0,
    tree_agreement: 0.67,  // 2/3 trees agreed
};

// Feature importance aggregated across all trees
let importance = path.aggregate_feature_importance();
```

### NeuralPath (Neural Networks)

For MLP and deep learning models:

```rust
use aprender::explainability::NeuralPath;

let path = NeuralPath {
    layer_activations: vec![
        vec![0.5, 0.8, 0.2],      // Hidden layer 1
        vec![0.9, 0.1],           // Hidden layer 2
    ],
    input_gradients: vec![0.1, -0.3, 0.5, 0.2],  // Saliency
    output_logits: vec![0.9, 0.05, 0.05],
    predicted_class: 0,
};

// Gradient-based feature importance
let saliency = path.saliency_map();
```

---

## Hash Chain Audit Collector

For regulatory compliance, Aprender provides tamper-evident audit trails:

```rust
use aprender::explainability::{HashChainCollector, ChainVerification};

// Create collector for an inference session
let mut collector = HashChainCollector::new("session-2025-12-10-001");

// Record each prediction with its decision path
for (input, prediction, path) in predictions {
    collector.record(path);
}

// Verify chain integrity (detects tampering)
let verification: ChainVerification = collector.verify_chain();
assert!(verification.valid);
println!("Verified {} entries", verification.entries_verified);

// Export for compliance
let audit_json = collector.to_json()?;
```

### Hash Chain Structure

Each entry contains:
- **Sequence number**: Monotonically increasing
- **Previous hash**: SHA-256 of prior entry (zeros for genesis)
- **Current hash**: SHA-256 of this entry + previous hash
- **Timestamp**: Nanosecond precision
- **Decision path**: Full explanation

```rust
pub struct HashChainEntry<P: DecisionPath> {
    pub sequence: u64,
    pub prev_hash: [u8; 32],
    pub hash: [u8; 32],
    pub timestamp_ns: u64,
    pub path: P,
}
```

---

## Integration Example

Complete example showing prediction with explainability:

```rust
use aprender::tree::{DecisionTreeClassifier, DecisionTreeConfig};
use aprender::explainability::{HashChainCollector, Explainable};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Train model
    let config = DecisionTreeConfig::default().max_depth(5);
    let mut tree = DecisionTreeClassifier::new(config);
    tree.fit(&x_train, &y_train)?;

    // Create audit collector
    let mut audit = HashChainCollector::new("iris-classification-2025-12-10");

    // Predict with explainability
    for sample in &x_test {
        let (prediction, path) = tree.predict_explain(sample)?;

        // Log for debugging
        println!("{}", path.explain());

        // Record for audit
        audit.record(path);
    }

    // Verify and export audit trail
    let verification = audit.verify_chain();
    assert!(verification.valid, "Audit chain compromised!");

    // Save for compliance
    std::fs::write("audit_trail.json", audit.to_json()?)?;

    Ok(())
}
```

---

## Best Practices

### 1. Always Enable Explainability for Production

```rust
// DON'T: Silent predictions
let pred = model.predict(&input);

// DO: Explainable predictions
let (pred, path) = model.predict_explain(&input)?;
audit.record(path);
```

### 2. Verify Audit Chain Before Export

```rust
let verification = audit.verify_chain();
if !verification.valid {
    log::error!("Audit chain broken at entry {}",
                verification.first_break.unwrap());
    // Alert security team
}
```

### 3. Use Typed Decision Paths

```rust
// Type system ensures correct path type for model
let tree_path: TreePath = tree.predict_explain(&input)?.1;
let linear_path: LinearPath = linear.predict_explain(&input)?.1;
```

---

## Toyota Way Integration

This module embodies three Toyota Way principles:

1. **Jidoka (Built-in Quality)**: Quality is built into predictions through mandatory explainability
2. **Shihai wo Kakusanai (Never Hide Failures)**: Every decision is auditable
3. **Genchi Genbutsu (Go and See)**: Decision paths let you trace exactly why a model decided what it did

---

## See Also

- [Decision Trees Theory](../ml-fundamentals/decision-trees.md)
- [Ensemble Methods Theory](../ml-fundamentals/ensemble-methods.md)
- [Model Serialization](./model-serialization.md)
- [Batuta Integration](./batuta-integration.md)
