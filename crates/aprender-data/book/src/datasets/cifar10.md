# CIFAR-10 Dataset

Color image classification dataset (Krizhevsky, 2009).

## Overview

- **Embedded**: 100 samples (10 per class)
- **Full** (hf-hub): 60,000 samples
- **Features**: 3,072 pixels (32x32x3 RGB)
- **Classes**: 10 object categories
- **Task**: Multi-class image classification

## Loading

```rust
use alimentar::datasets::{cifar10, CanonicalDataset};

let dataset = cifar10()?;
assert_eq!(dataset.len(), 100);
assert_eq!(dataset.num_features(), 3072);
assert_eq!(dataset.num_classes(), 10);
```

## Full Dataset

Enable `hf-hub` feature for complete CIFAR-10:

```toml
[dependencies]
alimentar = { version = "0.1", features = ["hf-hub"] }
```

```rust
use alimentar::datasets::Cifar10Dataset;

let full = Cifar10Dataset::load_full()?;
assert_eq!(full.len(), 60_000);
```

## Class Names

```rust
use alimentar::datasets::{Cifar10Dataset, CIFAR10_CLASSES};

// All class names
println!("{:?}", CIFAR10_CLASSES);
// ["airplane", "automobile", "bird", "cat", "deer",
//  "dog", "frog", "horse", "ship", "truck"]

// Lookup by label
let name = Cifar10Dataset::class_name(0); // Some("airplane")
let name = Cifar10Dataset::class_name(9); // Some("truck")
let name = Cifar10Dataset::class_name(10); // None
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| pixel_0..pixel_3071 | f32 | Pixel intensities (0.0-1.0) |
| label | i32 | Class index (0-9) |

## Pixel Layout

Pixels are stored channel-first (planar):

```text
R channel: pixel_0    .. pixel_1023   (32x32 = 1024)
G channel: pixel_1024 .. pixel_2047
B channel: pixel_2048 .. pixel_3071
```

To extract RGB for pixel (row, col):

```rust
fn rgb_indices(row: usize, col: usize) -> (usize, usize, usize) {
    let idx = row * 32 + col;
    (idx, idx + 1024, idx + 2048)  // R, G, B
}
```

## Train/Test Split

```rust
let dataset = cifar10()?;
let split = dataset.split()?;

// 80/20 split
assert_eq!(split.train.len(), 80);
assert_eq!(split.test.len(), 20);
```

## Embedded Sample

The embedded dataset uses class-specific color patterns:

| Class | Color Pattern |
|-------|---------------|
| airplane | Sky blue |
| automobile | Gray |
| bird | Brown |
| cat | Orange |
| deer | Dark brown |
| dog | Tan |
| frog | Green |
| horse | Brown |
| ship | Navy |
| truck | Red |

## Example: Image Classification Pipeline

```rust
use alimentar::datasets::{cifar10, Cifar10Dataset, CanonicalDataset};
use alimentar::DataLoader;

let dataset = cifar10()?;
let split = dataset.split()?;

let train_loader = DataLoader::new(split.train)
    .batch_size(64)
    .shuffle(true);

for batch in train_loader {
    println!("Batch: {} images", batch.num_rows());
    // Extract features and labels for training...
}
```

## Reference

Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images." Technical Report, University of Toronto.
