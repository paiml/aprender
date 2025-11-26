# Case Study: Synthetic Data Generation for ML

Synthetic data generation augments training datasets when labeled data is scarce. This example demonstrates aprender's synthetic data module for text augmentation, template-based generation, and weak supervision.

## Running the Example

```bash
cargo run --example synthetic_data_generation
```

## Techniques Demonstrated

### 1. EDA (Easy Data Augmentation)

EDA applies simple text transformations to generate variations:

```rust
use aprender::synthetic::eda::{EdaConfig, EdaGenerator};
use aprender::synthetic::{SyntheticConfig, SyntheticGenerator};

let generator = EdaGenerator::new(EdaConfig::default());

let seeds = vec![
    "git commit -m 'fix bug'".to_string(),
    "cargo build --release".to_string(),
];

let config = SyntheticConfig::default()
    .with_augmentation_ratio(2.0)  // 2x original data
    .with_quality_threshold(0.3)
    .with_seed(42);

let augmented = generator.generate(&seeds, &config)?;
```

**Output:**
```
Original commands (3):
  git commit -m 'fix bug'
  cargo build --release
  docker run nginx

Augmented commands (6):
  git commit -m 'fix bug' (quality: 1.00)
  git -m commit 'fix bug' (quality: 0.67)
  cargo build --release (quality: 1.00)
  cargo --release build (quality: 0.67)
```

### 2. Template-Based Generation

Generate structured commands from templates with variable slots:

```rust
use aprender::synthetic::template::{Template, TemplateGenerator};

let git_template = Template::new("git {action} {args}")
    .with_slot("action", &["commit", "push", "pull", "checkout"])
    .with_slot("args", &["-m 'update'", "--all", "main"]);

let cargo_template = Template::new("cargo {cmd} {flags}")
    .with_slot("cmd", &["build", "test", "run", "check"])
    .with_slot("flags", &["--release", "--all-features", ""]);

let generator = TemplateGenerator::new()
    .with_template(git_template)
    .with_template(cargo_template);

// Total combinations = 4*3 + 4*3 = 24
println!("Possible combinations: {}", generator.total_combinations());
```

### 3. Weak Supervision

Label unlabeled data using heuristic labeling functions:

```rust
use aprender::synthetic::weak_supervision::{
    WeakSupervisionGenerator, WeakSupervisionConfig,
    AggregationStrategy, KeywordLF, LabelVote,
};

let mut generator = WeakSupervisionGenerator::<String>::new()
    .with_config(
        WeakSupervisionConfig::new()
            .with_aggregation(AggregationStrategy::MajorityVote)
            .with_min_votes(1)
            .with_min_confidence(0.5),
    );

// Add domain-specific labeling functions
generator.add_lf(Box::new(KeywordLF::new(
    "version_control",
    &["git", "svn", "commit", "push"],
    LabelVote::Positive,
)));

generator.add_lf(Box::new(KeywordLF::new(
    "dangerous",
    &["rm -rf", "sudo rm", "format"],
    LabelVote::Negative,
)));

let samples = vec![
    "git push origin main".to_string(),
    "rm -rf /tmp/cache".to_string(),
];

let labeled = generator.generate(&samples, &config)?;
```

**Output:**
```
Labeled samples:
  [SAFE] (conf: 0.75) git push origin main
  [UNSAFE] (conf: 0.80) rm -rf /tmp/cache
  [SAFE] (conf: 0.65) cargo test --all
  [UNKNOWN] (conf: 0.20) echo hello world
```

### 4. Caching for Efficiency

Cache generated data to avoid redundant computation:

```rust
use aprender::synthetic::cache::SyntheticCache;

let mut cache = SyntheticCache::<String>::new(100_000); // 100KB cache
let generator = EdaGenerator::new(EdaConfig::default());

// First call - cache miss, runs generation
let result1 = cache.get_or_generate(&seeds, &config, &generator)?;

// Second call - cache hit, returns cached result
let result2 = cache.get_or_generate(&seeds, &config, &generator)?;

println!("Hit rate: {:.1}%", cache.stats().hit_rate() * 100.0);
```

## Quality Metrics

### Diversity Score

Measures how diverse the generated samples are:

```rust
let diversity = generator.diversity_score(&augmented);
// Returns value between 0.0 (identical) and 1.0 (completely diverse)
```

### Quality Score

Measures how well generated samples preserve semantic meaning:

```rust
let quality = generator.quality_score(&generated_sample, &original_seed);
// Returns value between 0.0 (unrelated) and 1.0 (identical)
```

## Use Cases

| Technique | Best For | Example |
|-----------|----------|---------|
| EDA | Text classification | Sentiment analysis training |
| Templates | Structured data | Command generation |
| Weak Supervision | Unlabeled data | Auto-labeling datasets |
| Caching | Repeated generation | Batch augmentation pipelines |

## Configuration Reference

### `SyntheticConfig`

```rust
SyntheticConfig::default()
    .with_augmentation_ratio(2.0)   // Generate 2x original
    .with_quality_threshold(0.3)    // Minimum quality score
    .with_seed(42)                  // Reproducible randomness
```

### `EdaConfig`

```rust
EdaConfig::default()
    .with_swap_probability(0.1)     // Word swap chance
    .with_delete_probability(0.1)   // Word deletion chance
    .with_insert_probability(0.1)   // Word insertion chance
```

### `WeakSupervisionConfig`

```rust
WeakSupervisionConfig::new()
    .with_aggregation(AggregationStrategy::MajorityVote)
    .with_min_votes(2)              // Need 2+ LFs to agree
    .with_min_confidence(0.5)       // 50% confidence threshold
```

## See Also

- [AutoML Chapter](../ml-fundamentals/automl.md) - Automated model tuning
- [Text Preprocessing](./text-preprocessing.md) - NLP preprocessing
