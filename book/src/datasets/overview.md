# Canonical Datasets

Alimentar provides built-in access to well-known ML datasets for tutorials, benchmarking, and quick experimentation. All datasets follow a **sovereign-first** design: embedded samples work offline without any network dependency.

## Design Philosophy

- **Offline by default**: Small embedded samples work without downloads
- **Optional full data**: Enable `hf-hub` feature for complete datasets
- **Uniform API**: All datasets implement `CanonicalDataset` trait
- **Zero configuration**: One-liner loading with sensible defaults

## Available Datasets

| Dataset | Function | Embedded | Full (hf-hub) | Use Case |
|---------|----------|----------|---------------|----------|
| Iris | `iris()` | 150 | N/A | Classification intro |
| MNIST | `mnist()` | 100 | 70,000 | Digit recognition |
| Fashion-MNIST | `fashion_mnist()` | 100 | 70,000 | Clothing classification |
| CIFAR-10 | `cifar10()` | 100 | 60,000 | Image classification |
| CIFAR-100 | `cifar100()` | 100 | 60,000 | Fine-grained classification |

## Quick Start

```rust
use alimentar::datasets::{iris, mnist, cifar10, CanonicalDataset};

// Load datasets (no network required)
let iris = iris()?;
let mnist = mnist()?;
let cifar = cifar10()?;

// Common trait methods
println!("Iris: {} samples, {} features", iris.len(), iris.num_features());
println!("MNIST: {} classes", mnist.num_classes());
println!("CIFAR-10: {}", cifar.description());
```

## The CanonicalDataset Trait

All canonical datasets implement this trait:

```rust
pub trait CanonicalDataset {
    fn data(&self) -> &ArrowDataset;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn num_features(&self) -> usize;
    fn num_classes(&self) -> usize;
    fn feature_names(&self) -> &'static [&'static str];
    fn target_name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}
```

## Train/Test Splits

MNIST and CIFAR-10 provide built-in 80/20 splits:

```rust
let mnist = mnist()?;
let split = mnist.split()?;

println!("Train: {} samples", split.train.len());
println!("Test: {} samples", split.test.len());
```

## Full Datasets (Optional)

For production use, enable the `hf-hub` feature to download complete datasets:

```toml
[dependencies]
alimentar = { version = "0.1", features = ["hf-hub"] }
```

```rust
// Downloads from HuggingFace Hub on first use
let full_mnist = MnistDataset::load_full()?;
let full_cifar = Cifar10Dataset::load_full()?;
```
