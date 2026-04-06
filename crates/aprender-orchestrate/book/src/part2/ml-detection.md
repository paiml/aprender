# ML Framework Detection

ML framework detection scans Python source files for import statements from NumPy, scikit-learn, and PyTorch. Each detected operation is mapped to its equivalent in the Sovereign AI Stack.

## Detection Pipeline

The `LibraryAnalyzer` in `src/pipeline_analysis.rs` walks all `.py` files and checks for library-specific import patterns:

```rust
pub struct LibraryAnalyzer {
    numpy_converter: NumPyConverter,
    sklearn_converter: SklearnConverter,
    pytorch_converter: PyTorchConverter,
}
```

Detection is import-gated: a file must contain `import numpy` or `from numpy` before individual operations are scanned. This avoids false positives from string matches in comments or documentation.

## Framework Mapping

| Python Framework | Sovereign Stack Crate | Layer |
|-----------------|----------------------|-------|
| NumPy | trueno | SIMD/GPU compute primitives |
| scikit-learn | aprender | ML algorithms |
| PyTorch / Transformers | realizar | Inference engine |

## NumPy to Trueno

The `NumPyConverter` maps 12 NumPy operations to Trueno equivalents:

| NumPy | Trueno | Complexity |
|-------|--------|-----------|
| `np.array([...])` | `Vector::from_slice(&[...])` | Low |
| `np.add(a, b)` | `a.add(&b).unwrap()` | Low |
| `np.subtract(a, b)` | `a.sub(&b).unwrap()` | Low |
| `np.multiply(a, b)` | `a.mul(&b).unwrap()` | Low |
| `np.dot(a, b)` | `a.dot(&b).unwrap()` | High |
| `np.sum(a)` | `a.sum()` | Medium |

Each operation carries a complexity level that feeds into the MoE backend selector during Phase 3 optimization.

## scikit-learn to Aprender

The `SklearnConverter` maps algorithms across six sklearn module groups:

| sklearn Module | Example Algorithm | Aprender Equivalent |
|---------------|-------------------|-------------------|
| `linear_model` | `LinearRegression` | `aprender::linear_model::LinearRegression` |
| `cluster` | `KMeans` | `aprender::cluster::KMeans` |
| `tree` | `DecisionTreeClassifier` | `aprender::tree::DecisionTreeClassifier` |
| `preprocessing` | `StandardScaler` | `aprender::preprocessing::StandardScaler` |
| `model_selection` | `train_test_split` | `aprender::model_selection::train_test_split` |
| `metrics` | `accuracy_score` | `aprender::metrics::accuracy_score` |

## PyTorch to Realizar

The `PyTorchConverter` handles inference-focused operations:

| PyTorch | Realizar | Notes |
|---------|----------|-------|
| `torch.load()` / `from_pretrained()` | `GGUFModel::from_file()` | Model loading |
| `model.forward(x)` | `model.forward(&input)` | Inference |
| `model.generate()` | `generate_text(&model, &tokens, len)` | Text generation |
| `AutoTokenizer` | `Tokenizer::from_file()` | Tokenization |
| `nn.Linear` | `LinearLayer::new(in, out)` | Layer types |
| `nn.MultiheadAttention` | `AttentionLayer::new(dim, heads)` | Attention |

## CLI Usage

```bash
$ batuta analyze --languages --dependencies --tdg ./ml-project

ML Framework Detection
----------------------
NumPy:    model.py (np.array, np.dot, np.sum) --> trueno::Vector
sklearn:  train.py (LinearRegression, KMeans) --> aprender
PyTorch:  infer.py (torch.load, .forward)     --> realizar
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
