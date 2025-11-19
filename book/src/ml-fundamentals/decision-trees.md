# Decision Trees Theory

<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (All examples verified)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 15+ | CART algorithm verified |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: src/tree/mod.rs tests*
<!-- DOC_STATUS_END -->

---

## Overview

Decision trees learn hierarchical decision rules by recursively partitioning the feature space. They're interpretable, handle non-linear relationships, and require no feature scaling.

**Key Concepts**:
- **CART Algorithm**: Classification And Regression Trees
- **Gini Impurity**: Measures node purity (classification)
- **Recursive Splitting**: Build tree top-down, greedy
- **Max Depth**: Controls overfitting

**Why This Matters**:
Decision trees mirror human decision-making: "If feature X > threshold, then..." They're the foundation of powerful ensemble methods (Random Forests, Gradient Boosting).

---

## Mathematical Foundation

### The Decision Tree Structure

A decision tree is a binary tree where:
- **Internal nodes**: Test one feature against a threshold
- **Edges**: Represent test outcomes (≤ threshold, > threshold)
- **Leaves**: Contain class predictions

**Example Tree**:
```
        [Petal Width ≤ 0.8]
       /                    \
   Class 0           [Petal Length ≤ 4.9]
                    /                    \
               Class 1                 Class 2
```

### Gini Impurity

**Definition**:
```
Gini(S) = 1 - Σ p_i²

where:
S = set of samples in a node
p_i = proportion of class i in S
```

**Interpretation**:
- **Gini = 0.0**: Pure node (all samples same class)
- **Gini = 0.5**: Maximum impurity (binary, 50/50 split)
- **Gini < 0.5**: More pure than random

**Why squared?** Penalizes mixed distributions more than linear measure.

### Information Gain

When we split a node into left and right children:

```
InfoGain = Gini(parent) - [w_L * Gini(left) + w_R * Gini(right)]

where:
w_L = n_left / n_total  (weight of left child)
w_R = n_right / n_total (weight of right child)
```

**Goal**: Maximize information gain → find best split

### CART Algorithm (Classification)

**Recursive Tree Building**:
```
function BuildTree(X, y, depth, max_depth):
    if stopping_criterion_met:
        return Leaf(majority_class(y))

    best_split = find_best_split(X, y)  # Maximize InfoGain

    if no_valid_split or depth >= max_depth:
        return Leaf(majority_class(y))

    X_left, y_left, X_right, y_right = partition(X, y, best_split)

    return Node(
        feature = best_split.feature,
        threshold = best_split.threshold,
        left = BuildTree(X_left, y_left, depth+1, max_depth),
        right = BuildTree(X_right, y_right, depth+1, max_depth)
    )
```

**Stopping Criteria**:
1. All samples in node have same class (Gini = 0)
2. Reached max_depth
3. Node has too few samples (min_samples_split)
4. No split reduces impurity

---

## Implementation in Aprender

### Example 1: Simple Binary Classification

```rust
use aprender::tree::DecisionTreeClassifier;
use aprender::primitives::Matrix;

// XOR-like problem (not linearly separable)
let x = Matrix::from_vec(4, 2, vec![
    0.0, 0.0,  // Class 0
    0.0, 1.0,  // Class 1
    1.0, 0.0,  // Class 1
    1.0, 1.0,  // Class 0
]).unwrap();
let y = vec![0, 1, 1, 0];

// Train decision tree with max depth 3
let mut tree = DecisionTreeClassifier::new()
    .with_max_depth(3);

tree.fit(&x, &y).unwrap();

// Predict on training data (should be perfect)
let predictions = tree.predict(&x);
println!("Predictions: {:?}", predictions); // [0, 1, 1, 0]

let accuracy = tree.score(&x, &y);
println!("Accuracy: {:.3}", accuracy); // 1.000
```

**Test Reference**: `src/tree/mod.rs::tests::test_build_tree_simple_split`

### Example 2: Multi-Class Classification (Iris)

```rust
// Iris dataset (3 classes, 4 features)
// Simplified example - see case study for full implementation

let mut tree = DecisionTreeClassifier::new()
    .with_max_depth(5);

tree.fit(&x_train, &y_train).unwrap();

// Test set evaluation
let y_pred = tree.predict(&x_test);
let accuracy = tree.score(&x_test, &y_test);
println!("Test Accuracy: {:.3}", accuracy); // e.g., 0.967
```

**Case Study**: See [Decision Tree - Iris Classification](../examples/decision-tree-iris.md)

### Example 3: Model Serialization

```rust
// Train and save tree
let mut tree = DecisionTreeClassifier::new()
    .with_max_depth(4);
tree.fit(&x_train, &y_train).unwrap();

tree.save("tree_model.bin").unwrap();

// Load in production
let loaded_tree = DecisionTreeClassifier::load("tree_model.bin").unwrap();
let predictions = loaded_tree.predict(&x_test);
```

**Test Reference**: `src/tree/mod.rs::tests` (save/load tests)

---

## Understanding Gini Impurity

### Example Calculation

**Scenario**: Node with 6 samples: [A, A, A, B, B, C]

```
Class A: 3/6 = 0.5
Class B: 2/6 = 0.33
Class C: 1/6 = 0.17

Gini = 1 - (0.5² + 0.33² + 0.17²)
     = 1 - (0.25 + 0.11 + 0.03)
     = 1 - 0.39
     = 0.61
```

**Interpretation**: 0.61 impurity (moderately mixed)

### Pure vs Impure Nodes

| Node | Distribution | Gini | Interpretation |
|------|-------------|------|----------------|
| [A, A, A, A] | 100% A | 0.0 | Pure (stop splitting) |
| [A, A, B, B] | 50% A, 50% B | 0.5 | Maximum impurity (binary) |
| [A, A, A, B] | 75% A, 25% B | 0.375 | Moderately pure |

**Test Reference**: `src/tree/mod.rs::tests::test_gini_impurity_*`

---

## Choosing Max Depth

### The Depth Trade-off

**Too shallow (max_depth = 1)**:
- Underfitting
- High bias, low variance
- Poor train and test accuracy

**Too deep (max_depth = ∞)**:
- Overfitting
- Low bias, high variance
- Perfect train accuracy, poor test accuracy

**Just right (max_depth = 3-7)**:
- Balanced bias-variance
- Good generalization

### Finding Optimal Depth

Use cross-validation:

```rust
// Pseudocode
for depth in 1..=10 {
    model = DecisionTreeClassifier::new().with_max_depth(depth);
    cv_score = cross_validate(model, x, y, k=5);
    // Select depth with best cv_score
}
```

**Rule of Thumb**:
- Simple problems: max_depth = 3-5
- Complex problems: max_depth = 5-10
- If using ensemble (Random Forest): deeper trees OK (15-30)

---

## Advantages and Limitations

### Advantages ✅

1. **Interpretable**: Can visualize and explain decisions
2. **No feature scaling**: Works on raw features
3. **Handles non-linear**: Learns complex boundaries
4. **Mixed data types**: Numeric and categorical features
5. **Fast prediction**: O(log n) traversal

### Limitations ❌

1. **Overfitting**: Single trees overfit easily
2. **Instability**: Small data changes → different tree
3. **Bias toward dominant classes**: In imbalanced data
4. **Greedy algorithm**: May miss global optimum
5. **Axis-aligned splits**: Can't learn diagonal boundaries easily

**Solution to overfitting**: Use ensemble methods (Random Forests, Gradient Boosting)

---

## Decision Trees vs Other Methods

### Comparison Table

| Method | Interpretability | Feature Scaling | Non-linear | Overfitting Risk | Speed |
|--------|------------------|-----------------|------------|------------------|-------|
| **Decision Tree** | High | Not needed | Yes | High (single tree) | Fast |
| **Logistic Regression** | Medium | Required | No (unless polynomial) | Low | Fast |
| **SVM** | Low | Required | Yes (kernels) | Medium | Slow |
| **Random Forest** | Medium | Not needed | Yes | Low | Medium |

### When to Use Decision Trees

**Good for**:
- Interpretability required (medical, legal domains)
- Mixed feature types
- Quick baseline
- Building block for ensembles

**Not good for**:
- Need best single-model accuracy (use ensemble instead)
- Linear relationships (logistic regression simpler)
- Large feature space (curse of dimensionality)

---

## Practical Considerations

### Feature Importance

Decision trees naturally rank feature importance:
- **Most important**: Features near the root (used early)
- **Less important**: Features deeper in tree or unused

**Interpretation**: Features used for early splits have highest information gain.

### Handling Imbalanced Classes

**Problem**: Tree biased toward majority class

**Solutions**:
1. **Class weights**: Penalize majority class errors more
2. **Sampling**: SMOTE, undersampling majority
3. **Threshold tuning**: Adjust prediction threshold

### Pruning (Post-Processing)

**Idea**: Build full tree, then remove nodes with low information gain

**Benefit**: Reduces overfitting without limiting depth during training

**Status in Aprender**: Not yet implemented (use max_depth instead)

---

## Verification Through Tests

Decision tree tests verify mathematical properties:

**Gini Impurity Tests**:
- Pure node → Gini = 0.0
- 50/50 binary split → Gini = 0.5
- Gini always in [0, 1]

**Tree Building Tests**:
- Pure leaf stops splitting
- Max depth enforced
- Predictions match majority class

**Property Tests** (via integration tests):
- Tree depth ≤ max_depth
- All leaves are pure or at max_depth
- Information gain non-negative

**Test Reference**: `src/tree/mod.rs` (15+ tests)

---

## Real-World Application

### Medical Diagnosis Example

**Problem**: Diagnose disease from symptoms (temperature, blood pressure, age)

**Decision Tree**:
```
          [Temperature > 38°C]
         /                    \
   [BP > 140]               Healthy
   /        \
Disease A   Disease B
```

**Why Decision Tree?**
- Interpretable (doctors can verify logic)
- No feature scaling (raw measurements)
- Handles mixed units (°C, mmHg, years)

### Credit Scoring Example

**Features**: Income, debt, employment length, credit history

**Decision Tree learns**:
- If income < $30k and debt > $20k → High risk
- If income > $80k → Low risk
- Else, check employment length...

**Advantage**: Transparent lending decisions (regulatory compliance)

---

## Further Reading

### Peer-Reviewed Papers

**Breiman et al. (1984)** - *Classification and Regression Trees*
- **Relevance**: Original CART algorithm (Gini impurity, recursive splitting)
- **Link**: Chapman and Hall/CRC (book, library access)
- **Key Contribution**: Unified framework for classification and regression trees
- **Applied in**: `src/tree/mod.rs` CART implementation

**Quinlan (1986)** - *Induction of Decision Trees*
- **Relevance**: Alternative algorithm using entropy (ID3)
- **Link**: [SpringerLink](https://link.springer.com/article/10.1007/BF00116251)
- **Key Contribution**: Information gain via entropy (alternative to Gini)

### Related Chapters

- [Ensemble Methods Theory](./ensemble-methods.md) - Random Forests (next chapter)
- [Classification Metrics Theory](./classification-metrics.md) - Evaluating trees
- [Cross-Validation Theory](./cross-validation.md) - Finding optimal max_depth

---

## Summary

**What You Learned**:
- ✅ Decision trees: hierarchical if-then rules
- ✅ Gini impurity: Gini = 1 - Σ p_i² (0 = pure, 0.5 = max)
- ✅ CART algorithm: greedy, top-down, recursive
- ✅ Information gain: Maximize reduction in impurity
- ✅ Max depth: Controls overfitting (tune with CV)
- ✅ Advantages: Interpretable, no scaling, non-linear
- ✅ Limitations: Overfitting, instability (use ensembles)

**Verification Guarantee**: Decision tree implementation extensively tested (15+ tests) in `src/tree/mod.rs`. Tests verify Gini calculations, tree building, and prediction logic.

**Quick Reference**:
- **Pure node**: Gini = 0 (stop splitting)
- **Max impurity**: Gini = 0.5 (binary 50/50)
- **Best split**: Maximize information gain
- **Prevent overfit**: Set max_depth (3-7 typical)

**Key Equations**:
```
Gini(S) = 1 - Σ p_i²
InfoGain = Gini(parent) - Weighted_Avg(Gini(children))
Split: feature ≤ threshold → left, else → right
```

---

**Next Chapter**: [Ensemble Methods Theory](./ensemble-methods.md)

**Previous Chapter**: [Classification Metrics Theory](./classification-metrics.md)
