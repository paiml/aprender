# CIFAR-100 Dataset

Fine-grained image classification dataset (Krizhevsky, 2009).

## Overview

- **Embedded**: 100 samples (1 per fine class)
- **Full** (hf-hub): 60,000 samples
- **Features**: 3,072 pixels (32x32x3 RGB)
- **Fine classes**: 100 object categories
- **Coarse classes**: 20 superclasses
- **Task**: Hierarchical multi-class classification

## Loading

```rust
use alimentar::datasets::{cifar100, CanonicalDataset};

let dataset = cifar100()?;
assert_eq!(dataset.len(), 100);
assert_eq!(dataset.num_features(), 3072);
assert_eq!(dataset.num_classes(), 100);
```

## Hierarchical Labels

CIFAR-100 provides two label levels:

```rust
// Schema includes both label types
// - fine_label: 0-99 (100 specific classes)
// - coarse_label: 0-19 (20 superclasses)
```

## Class Names

```rust
use alimentar::datasets::{Cifar100Dataset, CIFAR100_FINE_CLASSES, CIFAR100_COARSE_CLASSES};

// Fine classes (100)
let fine = Cifar100Dataset::fine_class_name(0);   // Some("apple")
let fine = Cifar100Dataset::fine_class_name(99);  // Some("worm")

// Coarse classes (20)
let coarse = Cifar100Dataset::coarse_class_name(0);  // Some("aquatic_mammals")
let coarse = Cifar100Dataset::coarse_class_name(19); // Some("vehicles_2")
```

## Superclass Mapping

| Coarse Class | Fine Classes (examples) |
|--------------|------------------------|
| aquatic_mammals | beaver, dolphin, otter, seal, whale |
| fish | aquarium_fish, flatfish, ray, shark, trout |
| flowers | orchid, poppy, rose, sunflower, tulip |
| fruit_and_vegetables | apple, mushroom, orange, pear, sweet_pepper |
| vehicles_1 | bicycle, bus, motorcycle, pickup_truck, train |
| vehicles_2 | lawn_mower, rocket, streetcar, tank, tractor |

## Full Dataset

```toml
[dependencies]
alimentar = { version = "0.1", features = ["hf-hub"] }
```

```rust
let full = Cifar100Dataset::load_full()?;
```

## Train/Test Split

```rust
let dataset = cifar100()?;
let split = dataset.split()?;
assert_eq!(split.train.len(), 80);
assert_eq!(split.test.len(), 20);
```

## Reference

Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images." Technical Report, University of Toronto.
