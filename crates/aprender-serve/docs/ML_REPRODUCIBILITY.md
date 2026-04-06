# ML Reproducibility Guide

This document describes Realizar's approach to ML reproducibility, including deterministic inference, model versioning, and seed management.

## Deterministic Inference

### CPU Inference Guarantee

**Claim**: CPU inference in Realizar is fully deterministic. Given the same model and input, output is identical across runs.

**Implementation**:
- No floating-point reassociation optimizations that change order of operations
- No non-deterministic thread scheduling in inference path
- Fixed iteration order in all loops

**Verification**:
```bash
# Run inference twice with same input
./target/release/realizar infer --model model.gguf --input "Hello" --seed 42
./target/release/realizar infer --model model.gguf --input "Hello" --seed 42
# Outputs must be byte-for-byte identical
```

### Seed Management

**Random number generator**: We use a seeded PRNG for all stochastic operations.

**Seed sources**:
1. **Explicit seed**: User-provided via `--seed` flag or API parameter
2. **Default seed**: 42 (when not specified, for reproducibility)
3. **Random seed**: Only when `--seed random` is explicitly requested

**API usage**:
```rust
use realizar::{Model, SamplingConfig};

let config = SamplingConfig {
    seed: Some(42),  // Explicit seed for reproducibility
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    ..Default::default()
};

let output = model.generate("Hello", &config)?;
```

**REST API**:
```json
{
  "prompt": "Hello",
  "max_tokens": 50,
  "seed": 42,
  "temperature": 0.7
}
```

### Temperature and Sampling

**Sampling strategies** and their determinism:

| Strategy | Deterministic? | Seed Required? |
|----------|---------------|----------------|
| Greedy | Yes | No |
| Top-K | With seed | Yes |
| Top-P | With seed | Yes |
| Temperature | With seed | Yes |

**Greedy sampling** is always deterministic (no randomness involved).

## Model Versioning

### Model File Checksums

All model files should be verified by SHA256 checksum:

```bash
# Generate checksum
sha256sum models/mnist_784x2.apr > models/mnist_784x2.apr.sha256

# Verify before inference
sha256sum -c models/mnist_784x2.apr.sha256
```

### Model Metadata

Model files should include:

```json
{
  "model_name": "mnist_784x2",
  "model_version": "1.0.0",
  "training_date": "2025-01-15",
  "framework": "aprender",
  "framework_version": "0.1.0",
  "architecture": "mlp",
  "parameters": 1576,
  "quantization": null,
  "training_config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "sgd",
    "seed": 42
  }
}
```

### Artifact Tracking

**Checked-in models** (for reproducibility):
```
models/
  mnist_784x2.apr          # 3,248 bytes - checked into git
  mnist_784x2.apr.sha256   # Checksum file
  mnist_784x2.meta.json    # Model metadata
```

**Large models** (not checked in):
```
# Download script with checksum verification
./scripts/download-model.sh llama-3.2-1b
# Verifies: llama-3.2-1b.gguf SHA256 = <expected>
```

## Dataset Reproducibility

### Canonical Datasets

Benchmarks use canonical ML datasets with fixed versions:

| Dataset | Version | Source | Checksum |
|---------|---------|--------|----------|
| MNIST | 2024-01 | Alimentar | `sha256:...` |
| CIFAR-10 | 2024-01 | Alimentar | `sha256:...` |
| WikiText-2 | 2024-01 | HuggingFace | `sha256:...` |

### Data Loading Reproducibility

```rust
use alimentar::Dataset;

// Fixed seed for train/test split
let (train, test) = Dataset::mnist()
    .split(0.8)
    .seed(42)
    .load()?;
```

### Preprocessing Determinism

All preprocessing operations are deterministic:
- Normalization uses pre-computed mean/std (not computed per-run)
- Augmentation seeds are fixed and documented
- Tokenization is stateless and deterministic

## Environment Reproducibility

### Rust Toolchain

Pinned in `rust-toolchain.toml`:
```toml
[toolchain]
channel = "1.83.0"
components = ["rustfmt", "clippy", "llvm-tools"]
```

### Dependencies

Locked in `Cargo.lock`:
```bash
# Verify dependencies haven't changed
cargo verify-project
cargo metadata --format-version 1 | sha256sum
```

### Docker Environment

Full environment isolation:
```bash
# Build reproducible image
docker build -t realizar:v0.2.3 .

# Run with fixed configuration
docker run --cpus=4 --memory=8g realizar:v0.2.3
```

### Hardware Configuration

Document and control:
```bash
# CPU configuration
cat /proc/cpuinfo | grep "model name"
lscpu | grep "CPU MHz"

# Set performance governor
sudo cpupower frequency-set --governor performance

# Disable hyperthreading (optional)
echo off | sudo tee /sys/devices/system/cpu/smt/control
```

## Reproducibility Checklist

Before publishing results:

### Code
- [ ] All code committed and pushed
- [ ] Cargo.lock checked in
- [ ] rust-toolchain.toml present
- [ ] No uncommitted changes: `git status --porcelain`

### Models
- [ ] Model files have checksums
- [ ] Model metadata documented
- [ ] Training configuration recorded

### Data
- [ ] Dataset versions documented
- [ ] Download scripts include checksums
- [ ] Preprocessing steps reproducible

### Environment
- [ ] Hardware specs documented
- [ ] OS/kernel version recorded
- [ ] CPU governor set to performance
- [ ] Docker image tagged with version

### Execution
- [ ] Random seeds documented
- [ ] Sampling configuration saved
- [ ] Output logs preserved

## Reporting Template

Use this template for reproducible results:

```markdown
## Experiment: [Name]

### Environment
- **Hardware**: Intel Core i7-12700K, 32GB RAM
- **OS**: Ubuntu 22.04.3 LTS, kernel 6.2.0
- **Rust**: 1.83.0 (from rust-toolchain.toml)
- **Realizar**: 0.2.3 (commit abc1234)

### Configuration
- **Model**: llama-3.2-1b.gguf (SHA256: abc...)
- **Seed**: 42
- **Temperature**: 0.7
- **Max tokens**: 100

### Results
- **Latency (p50)**: 45.2 ms ± 2.1 ms (n=1000)
- **Throughput**: 22.1 tokens/sec

### Reproduction
```bash
git checkout abc1234
cargo build --release
./target/release/realizar bench --config experiment.toml
```
```

## Known Non-Determinism Sources

### Acceptable Non-Determinism
- Benchmark timing (varies by system load)
- Log timestamps
- Process IDs in output

### Unacceptable Non-Determinism (bugs if present)
- Different output tokens for same seed
- Different logits for same input
- Different model loading behavior

Report any unacceptable non-determinism as a bug.

## References

1. Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *JMLR*, 22(164), 1-20.
2. Henderson, P., et al. (2018). Deep reinforcement learning that matters. *AAAI*.
3. Bouthillier, X., et al. (2019). Unreproducible research is reproducible. *ICML*.
4. MLCommons. (2023). MLPerf Inference Rules. https://github.com/mlcommons/inference_policies

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-10
**Authors**: Pragmatic AI Labs
