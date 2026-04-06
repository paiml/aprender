# Example 1: Python ML Project

This walkthrough demonstrates a full transpilation of a Python ML pipeline using
scikit-learn and NumPy into pure Rust powered by the Sovereign AI Stack.

## Scenario

A data science team maintains a fraud detection service written in Python. The
pipeline reads CSV data, normalizes features with `StandardScaler`, trains a
`RandomForestClassifier`, and serves predictions over HTTP. Latency is 12 ms
per request. The team wants sub-millisecond inference in a single static binary.

## Source Project Layout

```
fraud_detector/
  requirements.txt      # numpy, scikit-learn, pandas, flask
  train.py              # Training script
  serve.py              # Flask prediction endpoint
  tests/test_model.py   # pytest suite
```

## Step 1 -- Analyze

```bash
batuta analyze --languages --tdg ./fraud_detector
```

Batuta scans every file, detects Python, identifies NumPy, scikit-learn, and
Flask imports, and computes a Technical Debt Grade. Output includes a dependency
graph and framework detection summary.

```
Languages detected: Python (100%)
ML frameworks: numpy (32 ops), scikit-learn (8 algorithms)
Web framework: Flask (1 endpoint)
TDG Score: B (72/100)
```

## Step 2 -- Detect Frameworks

```bash
batuta analyze --ml-frameworks ./fraud_detector
```

The ML framework detector maps every NumPy call to a `trueno` operation and
every scikit-learn algorithm to an `aprender` equivalent. The report shows which
conversions are fully automated and which require manual review.

## Step 3 -- Transpile

```bash
batuta transpile ./fraud_detector --tool depyler --output ./fraud_detector_rs
```

Depyler converts Python to Rust. Batuta replaces NumPy calls with `trueno`
operations and scikit-learn models with `aprender` equivalents. The Flask
endpoint becomes an `axum` handler.

## Step 4 -- Optimize

```bash
batuta optimize ./fraud_detector_rs --backend auto
```

The MoE backend selector analyzes each operation. Small element-wise operations
stay scalar. Feature normalization across thousands of rows uses SIMD via
`trueno`. The random forest ensemble uses GPU when the data exceeds the 5x PCIe
transfer cost threshold.

## Step 5 -- Validate

```bash
batuta validate ./fraud_detector_rs --reference ./fraud_detector
```

Batuta runs the original Python test suite and the generated Rust test suite
side by side, comparing outputs with configurable tolerance (default 1e-6 for
floating point). Syscall tracing via `renacer` confirms identical I/O behavior.

## Result

| Metric        | Python  | Rust     |
|---------------|---------|----------|
| Inference     | 12 ms   | 0.4 ms   |
| Binary size   | 48 MB   | 3.2 MB   |
| Dependencies  | 127     | 4 crates |
| Memory        | 180 MB  | 12 MB    |

## Key Takeaways

- The 5-phase pipeline (Analyze, Transpile, Optimize, Validate, Build) handles
  the entire conversion without manual Rust authoring for standard patterns.
- Batuta's Jidoka principle stops the pipeline at the first validation failure,
  preventing broken code from reaching later phases.
- Framework-specific converters (NumPy, sklearn, PyTorch) are detailed in the
  following sub-chapters.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
