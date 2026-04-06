# MNIST Dataset

Handwritten digit recognition dataset (LeCun et al., 1998).

## Overview

- **Embedded**: 100 samples (10 per digit)
- **Full** (hf-hub): 70,000 samples
- **Features**: 784 pixels (28x28 grayscale)
- **Classes**: 10 digits (0-9)
- **Task**: Multi-class classification

## Loading

```rust
use alimentar::datasets::{mnist, CanonicalDataset};

// Embedded sample (offline)
let dataset = mnist()?;
assert_eq!(dataset.len(), 100);
assert_eq!(dataset.num_features(), 784);
assert_eq!(dataset.num_classes(), 10);
```

## Full Dataset

Enable `hf-hub` feature for complete MNIST:

```toml
[dependencies]
alimentar = { version = "0.1", features = ["hf-hub"] }
```

```rust
use alimentar::datasets::MnistDataset;

let full = MnistDataset::load_full()?;
assert_eq!(full.len(), 70_000);
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| pixel_0..pixel_783 | f32 | Pixel intensities (0.0-1.0) |
| label | i32 | Digit class (0-9) |

## Train/Test Split

```rust
let dataset = mnist()?;
let split = dataset.split()?;

// 80/20 split
assert_eq!(split.train.len(), 80);
assert_eq!(split.test.len(), 20);
```

## Pixel Layout

Pixels are stored in row-major order:

```text
pixel_0   pixel_1   ... pixel_27     (row 0)
pixel_28  pixel_29  ... pixel_55     (row 1)
...
pixel_756 pixel_757 ... pixel_783    (row 27)
```

To reconstruct a 28x28 image:

```rust
fn pixel_index(row: usize, col: usize) -> usize {
    row * 28 + col
}
```

## Embedded Sample

The embedded dataset contains procedurally generated digit patterns:

- 10 samples per digit class
- Simple geometric representations
- Useful for testing pipelines without downloads

## Example: Digit Classification Pipeline

```rust
use alimentar::datasets::{mnist, CanonicalDataset};
use alimentar::{DataLoader, Normalize, NormMethod, Transform};

let dataset = mnist()?;
let split = dataset.split()?;

// Normalize pixel values
let normalizer = Normalize::new(NormMethod::MinMax);

let train_loader = DataLoader::new(split.train)
    .batch_size(32)
    .shuffle(true);

for batch in train_loader {
    let normalized = normalizer.apply(batch)?;
    // Feed to model...
}
```

## Reference

LeCun, Y., Cortes, C., & Burges, C.J. (1998). "The MNIST database of handwritten digits." http://yann.lecun.com/exdb/mnist/
