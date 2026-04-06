# Fashion-MNIST Dataset

Zalando's Fashion-MNIST clothing classification dataset (Xiao et al., 2017).

## Overview

- **Embedded**: 100 samples (10 per class)
- **Full** (hf-hub): 70,000 samples
- **Features**: 784 pixels (28x28 grayscale)
- **Classes**: 10 clothing categories
- **Task**: Multi-class classification

## Loading

```rust
use alimentar::datasets::{fashion_mnist, CanonicalDataset};

let dataset = fashion_mnist()?;
assert_eq!(dataset.len(), 100);
assert_eq!(dataset.num_features(), 784);
assert_eq!(dataset.num_classes(), 10);
```

## Class Names

```rust
use alimentar::datasets::{FashionMnistDataset, FASHION_MNIST_CLASSES};

println!("{:?}", FASHION_MNIST_CLASSES);
// ["t-shirt/top", "trouser", "pullover", "dress", "coat",
//  "sandal", "shirt", "sneaker", "bag", "ankle boot"]

let name = FashionMnistDataset::class_name(0); // Some("t-shirt/top")
let name = FashionMnistDataset::class_name(9); // Some("ankle boot")
```

## Full Dataset

```toml
[dependencies]
alimentar = { version = "0.1", features = ["hf-hub"] }
```

```rust
let full = FashionMnistDataset::load_full()?;
```

## Train/Test Split

```rust
let dataset = fashion_mnist()?;
let split = dataset.split()?;
assert_eq!(split.train.len(), 80);
assert_eq!(split.test.len(), 20);
```

## Reference

Xiao, H., Rasul, K., & Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." arXiv:1708.07747.
