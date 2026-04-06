# Batuta Migration Examples

Real-world end-to-end migration examples demonstrating Python → Rust transpilation with Batuta.

## Overview

This directory contains complete migration examples showing:
- **Input**: Production-quality Python code using NumPy, sklearn, PyTorch
- **Output**: Equivalent Rust code with performance and safety improvements
- **Documentation**: Detailed migration guides, performance comparisons, deployment strategies

Each example is a self-contained demonstration that can be run independently.

## Examples

### 1. NumPy Data Processing

**Directory**: [`numpy-data-processing/`](numpy-data-processing/)

Sensor data analysis pipeline showing NumPy → Trueno migration:
- Load and preprocess 100 sensor readings
- Statistical analysis (mean, std, min, max, median)
- Outlier detection and removal
- Moving average smoothing
- **Performance**: 4-5× faster with SIMD acceleration

**Key Techniques**:
- Array operations → Iterator-based functional composition
- Broadcasting → Explicit loops with SIMD hints
- Fancy indexing → Filter/map combinators

**Use Cases**: IoT data processing, time-series analysis, scientific computing

[Read full guide →](numpy-data-processing/README.md)

### 2. sklearn Classification

**Directory**: [`sklearn-classifier/`](sklearn-classifier/)

ML classification pipeline showing sklearn → Aprender migration:
- Iris dataset classification (150 samples, 4 features, 3 classes)
- Train/test split with stratification
- StandardScaler feature normalization
- Logistic regression with L-BFGS optimizer
- **Performance**: 5-20× faster inference, 50% less memory

**Key Techniques**:
- Estimator API → Builder pattern with Result types
- fit/transform → Stateful transformers with ownership
- Model persistence → Serde serialization to SafeTensors

**Use Cases**: Production ML inference, edge deployment, real-time predictions

[Read full guide →](sklearn-classifier/README.md)

### 3. PyTorch Inference

**Directory**: [`pytorch-inference/`](pytorch-inference/)

Deep learning inference pipeline showing PyTorch → Realizar migration:
- LSTM-based sentiment analysis
- Word-level tokenization
- Batch processing for throughput
- Softmax probability estimation
- **Performance**: 2-5× faster inference, 50-80% less memory

**Key Techniques**:
- nn.Module → Struct with forward() method
- torch.no_grad() → Automatic inference mode
- Model loading → SafeTensors format
- GPU acceleration → Transparent backend selection

**Use Cases**: Production inference servers, serverless, WebAssembly, embedded

[Read full guide →](pytorch-inference/README.md)

## Quick Start

### Run Python Examples

```bash
# NumPy data processing
cd numpy-data-processing
python3 input.py

# sklearn classifier
cd ../sklearn-classifier
pip install scikit-learn
python3 input.py

# PyTorch inference
cd ../pytorch-inference
pip install torch
python3 input.py
```

### Run Rust Examples

```bash
# NumPy → Trueno
cargo run --example numpy_data_processing_output

# sklearn → Aprender
cargo run --example sklearn_classifier_output

# PyTorch → Realizar
cargo run --release --example pytorch_inference_output
```

### Transpile with Batuta

```bash
# Analyze Python code
batuta analyze examples/migrations/numpy-data-processing/input.py

# View detected libraries and conversion strategy
batuta status

# Transpile to Rust
batuta transpile examples/migrations/numpy-data-processing/input.py \
  --output examples/migrations/numpy-data-processing/generated.rs \
  --optimize --backend auto

# Build and run
cd examples/migrations/numpy-data-processing
cargo build --release
./target/release/generated
```

## Performance Summary

Typical performance improvements from Python → Rust migration:

| Library | Operation | Speedup | Memory Reduction |
|---------|-----------|---------|------------------|
| **NumPy** | Array operations | 4-5× | 30-50% |
| | Statistical functions | 3-6× | 30-50% |
| | Filtering/masking | 5-10× | 40-60% |
| **sklearn** | Inference | 5-20× | 40-60% |
| | Feature scaling | 10-20× | 50-70% |
| | Train/test split | 5-10× | 30-50% |
| **PyTorch** | Inference (CPU) | 2-5× | 50-80% |
| | Inference (GPU) | 10-20× | 40-60% |
| | Tokenization | 3-5× | 60-80% |

**Note**: Actual speedups vary by workload, hardware, and optimization level. Benchmarks run on Intel i7-12700K with AVX2.

## Migration Workflow

Standard workflow for migrating Python ML/data code to Rust:

### 1. **Analysis Phase**

```bash
batuta analyze your_project/
```

- Detects Python libraries (NumPy, sklearn, PyTorch, etc.)
- Identifies conversion patterns
- Generates dependency graph
- Computes TDG (Technical Debt Grade) score

### 2. **Transpilation Phase**

```bash
batuta transpile your_project/ \
  --output rust_project/ \
  --optimize \
  --backend auto
```

- Converts Python → Rust using:
  - **NumPy** → Trueno (SIMD/GPU tensor ops)
  - **sklearn** → Aprender (ML algorithms)
  - **PyTorch** → Realizar (inference engine)
- Applies optimization passes
- Selects optimal backend (CPU/SIMD/GPU)

### 3. **Optimization Phase**

```bash
batuta optimize rust_project/ \
  --simd --gpu \
  --target-throughput 1000
```

- SIMD auto-vectorization
- GPU kernel generation (if beneficial)
- Loop unrolling, fusion
- Memory layout optimization

### 4. **Validation Phase**

```bash
batuta validate rust_project/ \
  --compare-with your_project/ \
  --trace-syscalls
```

- Semantic equivalence verification
- Numerical accuracy testing
- Syscall tracing with Renacer
- Performance profiling

### 5. **Build Phase**

```bash
batuta build rust_project/ \
  --release \
  --target x86_64-unknown-linux-gnu
```

- Cargo compilation with LTO
- Link-time optimization
- Strip symbols for smaller binaries
- Cross-compilation support

## Benefits of Migration

### Performance
- ✅ **2-20× faster execution**: No interpreter overhead, optimized codegen
- ✅ **30-80% less memory**: No Python object overhead, better cache locality
- ✅ **Predictable latency**: No GC pauses, no GIL contention
- ✅ **Hardware acceleration**: SIMD, GPU with automatic backend selection

### Safety
- ✅ **Memory safety**: No buffer overflows, use-after-free, null pointers
- ✅ **Type safety**: Compile-time guarantees, no silent type coercion
- ✅ **Thread safety**: Data races prevented by ownership model
- ✅ **Numerical stability**: Explicit NaN/Inf handling

### Deployment
- ✅ **Single binary**: 2-20 MB vs 100-500 MB for Python + libraries
- ✅ **No dependencies**: Self-contained, no runtime installation
- ✅ **Cross-platform**: Easy cross-compilation for ARM, RISC-V, WebAssembly
- ✅ **Container-friendly**: Scratch/distroless images, 5-10 MB total

## Common Migration Patterns

### Arrays and Tensors

```python
# Python (NumPy)
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
result = np.mean(arr)
```

```rust
// Rust (Trueno or native)
use trueno::Tensor;
let arr = Tensor::from_slice(&[1.0, 2.0, 3.0]);
let result = arr.mean();
```

### Machine Learning

```python
# Python (sklearn)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

```rust
// Rust (Aprender)
use aprender::linear_model::LogisticRegression;
let mut model = LogisticRegression::new();
model.fit(&X_train, &y_train)?;
let predictions = model.predict(&X_test)?;
```

### Deep Learning Inference

```python
# Python (PyTorch)
import torch
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

```rust
// Rust (Realizar)
use realizar::Model;
model.set_eval_mode(true);
let output = model.forward(&input_tensor)?;
```

## Target Platforms

Batuta migrations support deployment to:

- ✅ **Linux** (x86_64, aarch64, riscv64)
- ✅ **Windows** (x86_64)
- ✅ **macOS** (x86_64, aarch64/M1)
- ✅ **WebAssembly** (wasm32-unknown-unknown, wasm32-wasi)
- ✅ **Embedded** (ARM Cortex-M, ESP32, Raspberry Pi)
- ✅ **Cloud** (AWS Lambda, Google Cloud Run, Azure Functions)

## Production Checklist

Before deploying migrated code to production:

- [ ] **Testing**: Unit tests, integration tests, property-based tests
- [ ] **Validation**: Numerical accuracy verification vs Python
- [ ] **Benchmarking**: Performance profiling, latency/throughput measurement
- [ ] **Error handling**: Replace unwrap() with proper error propagation
- [ ] **Logging**: Add tracing/metrics for observability
- [ ] **Documentation**: API docs, deployment guide
- [ ] **CI/CD**: Automated testing, benchmarking, deployment pipeline
- [ ] **Monitoring**: Metrics, alerts, dashboards

## Contributing

Found a bug or want to add more examples? See [CONTRIBUTING.md](../../CONTRIBUTING.md).

## References

- [Batuta Documentation](../../docs/)
- [Batuta Specification](../../docs/specifications/sovereign-ai-spec.md)
- [Trueno Tensor Library](https://github.com/paiml/trueno)
- [Aprender ML Library](https://github.com/paiml/aprender)
- [Realizar Inference Engine](https://github.com/paiml/realizar)

## License

MIT - See [LICENSE](../../LICENSE) for details.

---

*Generated by Batuta - Sovereign AI Stack*
*https://github.com/paiml/Batuta*
