# Curriculum Learning

Curriculum learning progressively increases training difficulty,
starting with easy examples and advancing to harder ones as the
model improves. This is particularly effective for CITL
(Compiler-in-the-Loop) training where error complexity varies.

## Overview

```rust
use entrenar::train::{TieredCurriculum, AdaptiveCurriculum, CurriculumCallback};

// Tiered: Fixed accuracy thresholds
let curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8]);

// Adaptive: Error-based tier selection with Feldman weighting
let curriculum = AdaptiveCurriculum::new()
    .with_error_weights(error_frequencies)
    .with_advancement_threshold(0.85);

trainer.add_callback(curriculum);
```

## TieredCurriculum

Advances through difficulty tiers when accuracy thresholds are met:

```rust
pub struct TieredCurriculum {
    thresholds: Vec<f32>,   // [0.6, 0.7, 0.8] = 60%, 70%, 80%
    current_tier: usize,    // 0 = Basic, 1 = Intermediate, etc.
    tier_epochs: usize,     // Epochs at current tier
}

impl TieredCurriculum {
    pub fn new(thresholds: Vec<f32>) -> Self;
    pub fn current_tier(&self) -> usize;
    pub fn tier_name(&self) -> &str;
    pub fn should_advance(&self, accuracy: f32) -> bool;
}
```

### Tier Progression

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tiered Curriculum                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tier 0 (Basic)         ───► 60% accuracy ───►                  │
│  Tier 1 (Intermediate)  ───► 70% accuracy ───►                  │
│  Tier 2 (Advanced)      ───► 80% accuracy ───►                  │
│  Tier 3 (Expert)        ───► Complete                           │
│                                                                 │
│  Example data mapping:                                          │
│  • Basic: Simple type errors, missing imports                   │
│  • Intermediate: Borrow checker basics                          │
│  • Advanced: Lifetime annotations, trait bounds                 │
│  • Expert: Complex generics, async/await patterns               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```rust
use entrenar::train::{Trainer, TieredCurriculum, TrainConfig};

// Define tier thresholds
let curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8]);

trainer.add_callback(curriculum);

// During training, curriculum automatically:
// 1. Tracks accuracy each epoch
// 2. Advances tier when threshold met
// 3. Adjusts data sampling based on current tier
```

### Callback Implementation

```rust
impl TrainerCallback for TieredCurriculum {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        let accuracy = 1.0 - ctx.loss; // Simplified; real impl uses val_accuracy

        if self.should_advance(accuracy) {
            self.current_tier += 1;
            self.tier_epochs = 0;
            println!("Tier {} → {} ↑", self.current_tier - 1, self.current_tier);
        } else {
            self.tier_epochs += 1;
        }

        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "TieredCurriculum"
    }
}
```

## AdaptiveCurriculum

Dynamically selects tier based on error category distribution (Feldman 2020):

```rust
pub struct AdaptiveCurriculum {
    error_weights: HashMap<String, f32>,  // Error category → weight
    advancement_threshold: f32,            // Min accuracy to advance
    current_difficulty: f32,               // 0.0 (easy) to 1.0 (hard)
}

impl AdaptiveCurriculum {
    pub fn new() -> Self;
    pub fn with_error_weights(self, weights: HashMap<String, f32>) -> Self;
    pub fn with_advancement_threshold(self, threshold: f32) -> Self;

    /// Compute sample weight based on error rarity
    pub fn sample_weight(&self, error_category: &str) -> f32;

    /// Update difficulty based on recent performance
    pub fn update_difficulty(&mut self, recent_accuracy: f32);
}
```

### Feldman Reweighting

Rare error categories receive higher weights to prevent model bias:

```rust
// Error frequency in corpus
let frequencies = hashmap! {
    "E0308" => 434,  // Mismatched types (common)
    "E0599" => 373,  // Method not found
    "E0106" => 45,   // Missing lifetime (rare)
    "E0621" => 23,   // Lifetime mismatch (very rare)
};

// Compute inverse frequency weights
let weights: HashMap<String, f32> = frequencies.iter()
    .map(|(code, count)| {
        let weight = 1.0 / (*count as f32).sqrt();
        (code.to_string(), weight)
    })
    .collect();

let curriculum = AdaptiveCurriculum::new()
    .with_error_weights(weights)
    .with_advancement_threshold(0.85);
```

**Weight Formula**: `weight = 1.0 / sqrt(frequency)`

| Error Code | Frequency | Weight |
|------------|-----------|--------|
| E0308 | 434 | 0.048 |
| E0106 | 45 | 0.149 |
| E0621 | 23 | 0.208 |

### Usage with alimentar

```rust
use alimentar::{ArrowDataset, WeightedDataLoader};
use entrenar::train::{Trainer, AdaptiveCurriculum};

// Load corpus with weights
let dataset = ArrowDataset::from_parquet("training_data.parquet")?;
let weights: Vec<f32> = dataset.column_as_vec("weight")?;

let loader = WeightedDataLoader::new(dataset, weights)?
    .batch_size(32)
    .seed(42);

// Curriculum adjusts sampling as training progresses
let curriculum = AdaptiveCurriculum::new()
    .with_advancement_threshold(0.85);

trainer.add_callback(curriculum);
let result = trainer.train(100, || loader.iter(), |batch| model.forward(batch));
```

## Efficiency Score

Track curriculum effectiveness with the efficiency metric:

```rust
/// E(T) = Accuracy / log(CorpusSize)
/// Higher is better - achieving high accuracy with less data
pub fn efficiency_score(accuracy: f32, corpus_size: usize) -> f32 {
    accuracy / (corpus_size as f32).ln()
}

// Example
let accuracy = 0.89;
let corpus_size = 10_000;
let efficiency = efficiency_score(accuracy, corpus_size);
// E(T) = 0.89 / ln(10000) = 0.89 / 9.21 = 0.097
```

**Interpretation:**
- Higher efficiency = better generalization
- Useful for comparing models trained on different corpus sizes
- Target: efficiency > 0.08 for production models

## CITL Integration

Complete curriculum learning setup for CITL training:

```rust
use alimentar::{ArrowDataset, WeightedDataLoader, AsyncPrefetchDataset};
use entrenar::train::{
    Trainer, TrainConfig, TieredCurriculum, ExplainabilityCallback,
    EarlyStopping, CheckpointCallback, MonitorCallback,
};
use entrenar::optim::AdamW;

fn train_citl_model(corpus_path: &str) -> Result<TrainResult> {
    // Load corpus with weighted sampling
    let dataset = ArrowDataset::from_parquet(corpus_path)?;
    let weights = dataset.column_as_vec::<f32>("weight")?;

    let loader = WeightedDataLoader::new(dataset, weights)?
        .batch_size(32)
        .num_samples(10_000)
        .seed(42);

    // Setup trainer with CITL callbacks
    let mut trainer = Trainer::new(
        params,
        Box::new(AdamW::new(0.0001, 0.9, 0.999, 1e-8, 0.01)),
        TrainConfig::default(),
    );

    // Monitoring first (catches NaN/Inf)
    trainer.add_callback(MonitorCallback::new());

    // Curriculum learning
    trainer.add_callback(TieredCurriculum::new(vec![0.6, 0.7, 0.8]));

    // Feature attribution
    trainer.add_callback(
        ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
            .with_top_k(10),
    );

    // Early stopping and checkpoints
    trainer.add_callback(EarlyStopping::new(5, 0.001));
    trainer.add_callback(CheckpointCallback::new("./checkpoints"));

    // Train
    let result = trainer.train(100, || loader.iter(), |batch| model.forward(batch));

    // Report efficiency
    let efficiency = efficiency_score(result.accuracy, loader.num_samples());
    println!("Efficiency: {:.4}", efficiency);

    Ok(result)
}
```

### Training Output

```
Epoch  1/100: loss=2.3456, acc=45.2%, tier=0 (Basic)
Epoch  5/100: loss=1.8234, acc=58.1%, tier=0 (Basic)
Epoch 10/100: loss=1.2345, acc=62.1%, tier=0 → tier=1 ↑ (Intermediate)
Epoch 15/100: loss=0.9876, acc=68.5%, tier=1 (Intermediate)
Epoch 25/100: loss=0.5678, acc=71.5%, tier=1 → tier=2 ↑ (Advanced)
Epoch 40/100: loss=0.3456, acc=82.3%, tier=2 → tier=3 ↑ (Expert)
Epoch 47/100: loss=0.2345, acc=89.3%, tier=3 (Expert)
Early stopping: patience exhausted

Final: acc=89.3%, efficiency=0.097
Top features: error_code (0.342), message_length (0.187), has_suggestion (0.156)
```

## Best Practices

### Threshold Selection

```rust
// Conservative: Ensure mastery before advancing
let conservative = TieredCurriculum::new(vec![0.7, 0.8, 0.9]);

// Aggressive: Faster advancement for quick iteration
let aggressive = TieredCurriculum::new(vec![0.5, 0.6, 0.7]);

// Balanced (recommended for CITL)
let balanced = TieredCurriculum::new(vec![0.6, 0.7, 0.8]);
```

### Combining with Other Callbacks

Order matters:

```rust
// 1. Monitoring (critical safety)
trainer.add_callback(MonitorCallback::new());

// 2. Curriculum (affects data sampling)
trainer.add_callback(TieredCurriculum::new(vec![0.6, 0.7, 0.8]));

// 3. Explainability (analysis)
trainer.add_callback(ExplainabilityCallback::new(ExplainMethod::PermutationImportance));

// 4. Early stopping (termination)
trainer.add_callback(EarlyStopping::new(5, 0.001));

// 5. Checkpointing (persistence)
trainer.add_callback(CheckpointCallback::new("./ckpt"));
```

### Monitoring Tier Progression

```rust
impl TrainerCallback for TierMonitor {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        if let Some(curriculum) = self.curriculum.as_ref() {
            let tier = curriculum.current_tier();
            let tier_name = curriculum.tier_name();
            let epochs_at_tier = curriculum.tier_epochs();

            println!(
                "Tier {}: {} ({} epochs at this tier)",
                tier, tier_name, epochs_at_tier
            );
        }
        CallbackAction::Continue
    }
}
```

## See Also

- [Callback System](./callback-system.md) - Full callback documentation
- [Explainability](./explainability.md) - Feature attribution
- [Real-Time Monitoring](../monitor/overview.md) - Training observability
- [alimentar WeightedDataLoader](https://docs.rs/alimentar) - Weighted sampling
