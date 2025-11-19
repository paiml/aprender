# Logistic Regression

## Prerequisites

Before reading this chapter, you should understand:

**Core Concepts:**
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md) - The testing methodology
- [The RED-GREEN-REFACTOR Cycle](../methodology/red-green-refactor.md) - The development cycle
- Basic machine learning concepts (supervised learning, training/testing)

**Rust Skills:**
- Builder pattern (for fluent APIs)
- Error handling with `Result`
- Basic vector/matrix operations

**Recommended reading order:**
1. [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
2. This chapter (Logistic Regression Case Study)
3. [Property-Based Testing](../advanced-testing/property-based-testing.md)

---

📝 **This chapter demonstrates binary classification using Logistic Regression.**

## Overview

Logistic Regression is a fundamental classification algorithm that uses the sigmoid function to model the probability of binary outcomes. This case study demonstrates the RED-GREEN-REFACTOR cycle for implementing a production-quality classifier.

## RED Phase: Writing Failing Tests

Following EXTREME TDD principles, we begin by writing comprehensive tests before implementation:

```rust
#[test]
fn test_logistic_regression_fit_simple() {
    let x = Matrix::from_vec(4, 2, vec![...]).unwrap();
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    let result = model.fit(&x, &y);
    assert!(result.is_ok());
}
```

**Test categories implemented:**
- Unit tests (12 tests)
- Property tests (4 tests)
- Doc tests (1 test)

## GREEN Phase: Minimal Implementation

The implementation includes:
- **Sigmoid activation**: σ(z) = 1 / (1 + e^(-z))
- **Binary cross-entropy loss** (implicit in gradient)
- **Gradient descent optimization**
- **Builder pattern API**

## REFACTOR Phase: Code Quality

**Optimizations applied:**
- Used `.enumerate()` instead of manual indexing
- Applied clippy suggestion for range contains
- Added comprehensive error validation

## Key Learning Points

1. **Mathematical correctness**: Sigmoid function ensures probabilities in [0, 1]
2. **API design**: Builder pattern for flexible configuration
3. **Property testing**: Invariants verified across random inputs
4. **Error handling**: Input validation prevents runtime panics

## Test Results

- **Total tests**: 514 passing
- **Coverage**: 100% for classification module
- **Mutation testing**: Builder pattern mutants caught
- **Property tests**: All 4 invariants hold

## Example Output

```
Training Accuracy: 100.0%
Test predictions:
  Feature1=2.50, Feature2=2.00 -> Class 0 (0.043 probability)
  Feature1=7.50, Feature2=8.00 -> Class 1 (0.990 probability)
```

---

## Model Persistence: SafeTensors Serialization

**Added in v0.4.0** (Issue #6)

LogisticRegression now supports SafeTensors format for model serialization, enabling deployment to production inference engines like **realizar**, **Ollama**, and integration with **HuggingFace**, **PyTorch**, and **TensorFlow** ecosystems.

### Why SafeTensors?

SafeTensors is the industry-standard format for ML model serialization because it:
- **Zero-copy loading** - Efficient memory usage
- **Cross-platform** - Compatible with Python, Rust, JavaScript
- **Language-agnostic** - Works with all major ML frameworks
- **Safe** - No arbitrary code execution (unlike pickle)
- **Deterministic** - Reproducible builds with sorted keys

### RED Phase: SafeTensors Tests

Following EXTREME TDD, we wrote 5 comprehensive tests before implementation:

```rust
#[test]
fn test_save_safetensors_unfitted_model() {
    // Test 1: Cannot save unfitted model
    let model = LogisticRegression::new();
    let result = model.save_safetensors("/tmp/model.safetensors");

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unfitted"));
}

#[test]
fn test_save_load_safetensors_roundtrip() {
    // Test 2: Save and load preserves model state
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();

    model.save_safetensors("model.safetensors").unwrap();
    let loaded = LogisticRegression::load_safetensors("model.safetensors").unwrap();

    // Verify predictions match exactly
    assert_eq!(model.predict(&x), loaded.predict(&x));
}

#[test]
fn test_safetensors_preserves_probabilities() {
    // Test 5: Probabilities are identical after save/load
    let probas_before = model.predict_proba(&x);

    model.save_safetensors("model.safetensors").unwrap();
    let loaded = LogisticRegression::load_safetensors("model.safetensors").unwrap();

    let probas_after = loaded.predict_proba(&x);

    // Verify probabilities match exactly (critical for binary classification)
    assert_eq!(probas_before, probas_after);
}
```

**All 5 tests:**
1. ✅ Unfitted model fails with clear error
2. ✅ Roundtrip preserves coefficients and intercept
3. ✅ Corrupted file fails gracefully
4. ✅ Missing file fails with clear error
5. ✅ **Probabilities preserved exactly** (critical for classification)

### GREEN Phase: Implementation

The implementation serializes two tensors: **coefficients** and **intercept**.

```rust
pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
    use crate::serialization::safetensors;
    use std::collections::BTreeMap;

    // Verify model is fitted
    let coefficients = self.coefficients.as_ref()
        .ok_or("Cannot save unfitted model. Call fit() first.")?;

    // Prepare tensors (BTreeMap ensures deterministic ordering)
    let mut tensors = BTreeMap::new();
    tensors.insert("coefficients".to_string(),
                   (coef_data, vec![coefficients.len()]));
    tensors.insert("intercept".to_string(),
                   (vec![self.intercept], vec![1]));

    safetensors::save_safetensors(path, tensors)?;
    Ok(())
}
```

**SafeTensors Binary Format:**

```text
┌─────────────────────────────────────────────────┐
│ 8-byte header (u64 little-endian)              │
│ = Length of JSON metadata in bytes             │
├─────────────────────────────────────────────────┤
│ JSON metadata:                                  │
│ {                                               │
│   "coefficients": {                             │
│     "dtype": "F32",                             │
│     "shape": [2],                               │
│     "data_offsets": [0, 8]                      │
│   },                                            │
│   "intercept": {                                │
│     "dtype": "F32",                             │
│     "shape": [1],                               │
│     "data_offsets": [8, 12]                     │
│   }                                             │
│ }                                               │
├─────────────────────────────────────────────────┤
│ Raw tensor data (IEEE 754 F32 little-endian)   │
│ coefficients: [w₁, w₂]                          │
│ intercept: [b]                                  │
└─────────────────────────────────────────────────┘
```

### Loading Models

```rust
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, String> {
    use crate::serialization::safetensors;

    // Load SafeTensors file
    let (metadata, raw_data) = safetensors::load_safetensors(path)?;

    // Extract tensors
    let coef_data = safetensors::extract_tensor(&raw_data,
        &metadata["coefficients"])?;
    let intercept_data = safetensors::extract_tensor(&raw_data,
        &metadata["intercept"])?;

    // Reconstruct model
    Ok(Self {
        coefficients: Some(Vector::from_vec(coef_data)),
        intercept: intercept_data[0],
        learning_rate: 0.01,  // Default hyperparameters
        max_iter: 1000,
        tol: 1e-4,
    })
}
```

### Production Deployment Example

Train in **aprender**, deploy to **realizar**:

```rust
// 1. Train LogisticRegression in aprender
let mut model = LogisticRegression::new()
    .with_learning_rate(0.1)
    .with_max_iter(1000);
model.fit(&x_train, &y_train).unwrap();

// 2. Save to SafeTensors
model.save_safetensors("fraud_detection.safetensors").unwrap();

// 3. Deploy to realizar inference engine
// realizar upload fraud_detection.safetensors \
//     --name "fraud-detector-v1" \
//     --version "1.0.0"

// 4. Inference via REST API
// curl -X POST http://realizar:8080/predict/fraud-detector-v1 \
//     -d '{"features": [1.5, 2.3]}'
// Response: {"prediction": 1, "probability": 0.847}
```

### Key Design Decisions

**1. Deterministic Serialization (BTreeMap)**

We use `BTreeMap` instead of `HashMap` to ensure sorted keys:

```rust
// ✅ CORRECT: Deterministic (sorted keys)
let mut tensors = BTreeMap::new();
tensors.insert("coefficients", ...);
tensors.insert("intercept", ...);
// JSON: {"coefficients": {...}, "intercept": {...}}  (alphabetical)

// ❌ WRONG: Non-deterministic (hash-based order)
let mut tensors = HashMap::new();
tensors.insert("intercept", ...);
tensors.insert("coefficients", ...);
// JSON: {"intercept": {...}, "coefficients": {...}}  (random order)
```

**Why it matters:**
- Git diffs show real changes only
- Reproducible builds for compliance
- Identical byte-for-byte outputs

**2. Probability Preservation**

Binary classification requires **exact** probability preservation:

```rust
// Before save
let prob = model.predict_proba(&x)[0];  // 0.847362

// After load
let loaded = LogisticRegression::load_safetensors("model.safetensors")?;
let prob_loaded = loaded.predict_proba(&x)[0];  // 0.847362 (EXACT)

assert_eq!(prob, prob_loaded);  // ✅ Passes (IEEE 754 F32 precision)
```

**Why it matters:**
- Medical diagnosis (life/death decisions)
- Financial fraud detection (regulatory compliance)
- Probability calibration must be exact

**3. Hyperparameters Not Serialized**

Training hyperparameters (`learning_rate`, `max_iter`, `tol`) are **not** saved:

```rust
// Hyperparameters only needed during training
let mut model = LogisticRegression::new()
    .with_learning_rate(0.1)   // Not saved
    .with_max_iter(1000);       // Not saved
model.fit(&x, &y).unwrap();

// Only weights saved (coefficients + intercept)
model.save_safetensors("model.safetensors").unwrap();

// Loaded model has default hyperparameters (doesn't matter for inference)
let loaded = LogisticRegression::load_safetensors("model.safetensors").unwrap();
// loaded.learning_rate = 0.01 (default, not 0.1)
// BUT predictions are identical!
```

**Rationale:**
- Hyperparameters affect **training**, not **inference**
- Smaller file size (only weights)
- Compatible with frameworks that don't support hyperparameters

### Integration with ML Ecosystem

**HuggingFace:**
```python
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(tensors["coefficients"])  # torch.Tensor([...])
```

**realizar (Rust):**
```rust
use realizar::SafetensorsModel;

let model = SafetensorsModel::from_file("model.safetensors")?;
let coefficients = model.get_tensor("coefficients")?;
let intercept = model.get_tensor("intercept")?;
```

### Lessons Learned

1. **Test-First Design** - Writing 5 tests before implementation revealed edge cases
2. **Roundtrip Testing** - Critical for serialization (save → load → verify identical)
3. **Determinism Matters** - BTreeMap ensures reproducible builds
4. **Probability Preservation** - Binary classification requires exact float equality
5. **Industry Standards** - SafeTensors enables cross-language model deployment

### Metrics

- **Implementation**: 131 lines (save_safetensors + load_safetensors + docs)
- **Tests**: 5 comprehensive tests (unfitted, roundtrip, corrupted, missing, probabilities)
- **Test Coverage**: 100% for serialization methods
- **Quality Gates**: ✅ fmt, ✅ clippy, ✅ doc, ✅ test
- **Mutation Testing**: All mutants caught (verified with cargo-mutants)

---

## Next Steps

Now that you've seen binary classification with Logistic Regression, explore related topics:

**More Classification Algorithms:**
1. **[Decision Tree Iris](./decision-tree-iris.md)** ← Next case study
   Multi-class classification with decision trees

2. **[Random Forest](./random-forest.md)**
   Ensemble methods for improved accuracy

**Advanced Testing:**
3. **[Property-Based Testing](../advanced-testing/property-based-testing.md)**
   Learn how to write the 4 property tests shown in this chapter

4. **[Mutation Testing](../advanced-testing/mutation-testing.md)**
   Verify tests catch bugs

**Best Practices:**
5. **[Builder Pattern](../best-practices/builder-pattern.md)**
   Master the fluent API design used in this example

6. **[Error Handling](../best-practices/error-handling.md)**
   Best practices for robust error handling
