# Work Prompt: Issue #8 - Complete SafeTensors Model Serialization

**GitHub Issue**: https://github.com/paiml/aprender/issues/8
**Status**: In Progress
**Sprint**: production-v0.3.1
**Priority**: P0

---

## 📊 PROGRESS TRACKER

**Overall Progress**: 0/7 models (0%)

### Models to Implement (7 total)

- ⬜ [1/7] Ridge (linear_model) - 0%
- ⬜ [2/7] Lasso (linear_model) - 0%
- ⬜ [3/7] ElasticNet (linear_model) - 0%
- ⬜ [4/7] DecisionTreeClassifier (tree) - 0%
- ⬜ [5/7] RandomForestClassifier (tree) - 0%
- ⬜ [6/7] KMeans (cluster) - 0%
- ⬜ [7/7] StandardScaler (preprocessing) - 0%

---

## 🎯 OBJECTIVE

Add SafeTensors serialization support to all remaining models in aprender, completing the production-ready serialization story for the entire library.

**Why This Matters**:
- **Production Deployment**: Cross-platform model sharing (Rust ↔ Python ↔ JavaScript)
- **HuggingFace Ecosystem**: Industry-standard format used by transformers, diffusers, etc.
- **Security**: No pickle vulnerabilities, safe deserialization
- **Interoperability**: Enable aprender models in any SafeTensors-compatible framework
- **Completeness**: 9/9 models fully serializable (currently 2/9)

---

## 🔬 TECHNICAL APPROACH

### Pattern to Follow (Established in Issues #5, #6)

Each model implementation follows this **proven pattern**:

```rust
impl SafeTensorsModel for ModelName {
    fn model_type() -> &'static str {
        "ModelName"
    }

    fn safetensors_metadata(&self) -> HashMap<String, String> {
        // Model-specific metadata (hyperparameters, state)
    }

    fn safetensors_tensors(&self) -> Vec<(&str, TensorView)> {
        // Model parameters as SafeTensors
    }

    fn from_safetensors(
        tensors: &HashMap<String, TensorView>,
        metadata: &HashMap<String, String>
    ) -> Result<Self, Box<dyn Error>> {
        // Reconstruct model from SafeTensors
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // Default implementation (usually no override needed)
    }

    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        // Default implementation (usually no override needed)
    }
}
```

### Test Suite Requirements (12+ tests per model)

**Basic Functionality** (4 tests):
1. `test_MODEL_save_load_roundtrip` - Save then load, verify identical
2. `test_MODEL_safetensors_metadata` - Verify metadata correctness
3. `test_MODEL_safetensors_tensors` - Verify tensor shapes/values
4. `test_MODEL_from_safetensors` - Reconstruct from tensors+metadata

**Edge Cases** (4 tests):
5. `test_MODEL_save_invalid_path` - Error handling
6. `test_MODEL_load_nonexistent_file` - Error handling
7. `test_MODEL_load_corrupted_metadata` - Robust parsing
8. `test_MODEL_load_missing_tensors` - Validation

**Property Tests** (2 tests):
9. `test_MODEL_roundtrip_preserves_predictions` - Predictions identical after save/load
10. `test_MODEL_multiple_save_loads_idempotent` - No data corruption over multiple cycles

**Cross-Platform** (2 tests):
11. `test_MODEL_python_interop` - Can load in Python (if Python bindings exist)
12. `test_MODEL_file_size_reasonable` - Verify compression/efficiency

---

## 📋 TASK BREAKDOWN

### Task 1: Ridge SafeTensors Implementation

**RED Phase**:
```rust
// tests/safetensors/ridge.rs
#[test]
fn test_ridge_save_load_roundtrip() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

    let mut model = Ridge::new(1.0);
    model.fit(&x, &y).unwrap();

    // Save to temporary file
    let path = "test_ridge_model.safetensors";
    model.save(path).unwrap();

    // Load from file
    let loaded_model = Ridge::load(path).unwrap();

    // Verify predictions match
    let pred_original = model.predict(&x).unwrap();
    let pred_loaded = loaded_model.predict(&x).unwrap();

    assert_eq!(pred_original.len(), pred_loaded.len());
    for i in 0..pred_original.len() {
        assert!((pred_original[i] - pred_loaded[i]).abs() < 1e-6);
    }

    // Cleanup
    std::fs::remove_file(path).unwrap();
}
```

**GREEN Phase**: Implement `SafeTensorsModel for Ridge`

**REFACTOR Phase**: Run quality gates, optimize

**Completion Criteria**:
- [ ] 12+ tests passing
- [ ] Zero clippy warnings
- [ ] Documentation with examples
- [ ] Book chapter updated

---

### Task 2: Lasso SafeTensors Implementation

Same pattern as Ridge. Key difference: Lasso uses coordinate descent (no closed-form), so metadata includes convergence info.

**Metadata to include**:
- `alpha` (regularization strength)
- `max_iter` (maximum iterations)
- `tol` (convergence tolerance)
- `n_iter` (actual iterations taken)

---

### Task 3: ElasticNet SafeTensors Implementation

Same pattern. Metadata includes both `alpha` and `l1_ratio`.

---

### Task 4: DecisionTreeClassifier SafeTensors Implementation

**Challenge**: Trees are recursive structures, not matrices.

**Solution**: Serialize tree as flat arrays:
- `node_features` (which feature each node splits on)
- `node_thresholds` (threshold value for each split)
- `node_left_child` (index of left child, -1 if leaf)
- `node_right_child` (index of right child, -1 if leaf)
- `node_values` (class probabilities for leaves)

**Metadata**:
- `max_depth`
- `n_nodes` (total nodes in tree)
- `n_classes` (number of classes)

---

### Task 5: RandomForestClassifier SafeTensors Implementation

**Challenge**: Forest is array of trees.

**Solution**:
- Serialize each tree separately (tree_0_node_features, tree_1_node_features, ...)
- Metadata includes `n_estimators` (number of trees)
- Each tree uses same flat array format as DecisionTreeClassifier

---

### Task 6: KMeans SafeTensors Implementation

**Simple**: Just centroids matrix + metadata.

**Tensors**:
- `centroids` (K × n_features matrix)

**Metadata**:
- `n_clusters` (K)
- `max_iter`
- `tol`
- `inertia` (within-cluster sum of squares)
- `n_iter` (actual iterations)

---

### Task 7: StandardScaler SafeTensors Implementation

**Tensors**:
- `mean` (n_features vector)
- `std` (n_features vector)

**Metadata**:
- `n_features`
- `with_mean` (boolean)
- `with_std` (boolean)

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat: Add SafeTensors support for [ModelName] (Refs #8)

Implements SafeTensorsModel trait for [ModelName]:
- save/load methods with .safetensors format
- Metadata: [list key metadata fields]
- Tensors: [list key tensors]

Cross-platform interoperability:
- Python/JavaScript can now load [ModelName] models
- Industry-standard HuggingFace format
- No pickle security vulnerabilities

Quality:
- 12+ tests (basic, edge cases, property tests)
- 100% pass rate
- Zero clippy warnings
- Updated book chapter with example

Progress: Task [X/7] complete ([X*14]%)

Refs #8
```

---

## 🚀 COMPLETION CRITERIA

Before marking Issue #8 complete:

- [ ] All 7 models implement SafeTensorsModel trait
- [ ] 84+ tests passing (12 per model)
- [ ] Zero clippy warnings (strict mode)
- [ ] Book chapter updated with all 7 models
- [ ] Cross-platform examples (Python interop if available)
- [ ] Cargo.toml version bumped to 0.3.1
- [ ] v0.3.1 released to crates.io
- [ ] CHANGELOG.md updated

---

## 🔍 QUALITY GATES (Run before each commit)

```bash
# Tier 1: Fast feedback (<1s)
cargo fmt --check && cargo clippy -- -W all && cargo check

# Tier 2: Pre-commit (<5s)
cargo test --lib && cargo clippy -- -D warnings

# Tier 3: Full validation (<5min)
cargo test --all
cargo test --test safetensors

# Tier 4: Coverage (optional, <10min)
cargo llvm-cov --all-features --workspace
```

---

## 📚 REFERENCES

- **Existing Implementations**:
  - `src/linear_model/linear_regression.rs` (Issue #5)
  - `src/linear_model/logistic_regression.rs` (Issue #6)
- **Test Patterns**:
  - `tests/safetensors/linear_regression.rs`
  - `tests/safetensors/logistic_regression.rs`
- **Book Chapter**: `book/src/examples/safetensors-delivery-verification.md`
- **SafeTensors Spec**: https://huggingface.co/docs/safetensors/

---

## 🎯 SUCCESS METRICS

**Production Readiness**:
- ✅ All 9 models fully serializable
- ✅ Cross-platform compatibility verified
- ✅ Security: No pickle dependencies

**Quality**:
- ✅ 84+ tests passing
- ✅ Zero defects
- ✅ TDG score maintained at A+ (100.0/100)

**Documentation**:
- ✅ Book chapter includes all 7 new models
- ✅ Code examples for each model
- ✅ Python interop examples (if available)

---

**Next Work**: After completing Issue #8, consider:
- Issue #9: Complete ML Fundamentals Book (3 remaining chapters)
- Issue #10: Neural Networks Phase 1 (MLP implementation)
