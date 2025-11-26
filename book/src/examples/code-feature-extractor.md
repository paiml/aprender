# Case Study: Code Feature Extraction for Defect Prediction

Extract 8-dimensional feature vectors from code commits for defect prediction, based on D'Ambros et al. (2012) benchmark methodology.

## Quick Start

```rust
use aprender::synthetic::code_features::{
    CodeFeatureExtractor, CommitFeatures, CommitDiff
};

let extractor = CodeFeatureExtractor::new();

let diff = CommitDiff::new()
    .with_files_changed(3)
    .with_lines_added(150)
    .with_lines_deleted(50)
    .with_timestamp(1700000000)
    .with_message("fix: resolve memory leak");

let features = extractor.extract(&diff);

// 8-dimensional feature vector
let vector = features.to_vec();
assert_eq!(vector.len(), 8);
```

## The 8-Dimensional Feature Vector

`CommitFeatures` contains standardized metrics for ML pipelines:

| Index | Field | Type | Description |
|-------|-------|------|-------------|
| 0 | `defect_category` | u8 | Predicted defect type (0-4) |
| 1 | `files_changed` | f32 | Number of modified files |
| 2 | `lines_added` | f32 | Lines of code added |
| 3 | `lines_deleted` | f32 | Lines of code removed |
| 4 | `complexity_delta` | f32 | Estimated complexity change |
| 5 | `timestamp` | f64 | Unix timestamp |
| 6 | `hour_of_day` | u8 | Hour (0-23 UTC) |
| 7 | `day_of_week` | u8 | Day (0=Sunday, 6=Saturday) |

## Defect Classification

The extractor automatically classifies commits based on message keywords:

### Categories

| Category | Value | Keywords |
|----------|-------|----------|
| Clean/Unknown | 0 | (no matches) |
| Bug Fix | 1 | fix, bug, error, crash, fault, defect, problem, wrong, broken, fail |
| Security | 2 | security, vulnerability, cve, exploit, injection, xss, csrf, auth |
| Performance | 3 | performance, perf, optimize, speed, fast, slow, memory, cache |
| Refactoring | 4 | refactor, clean, rename, move, reorganize, restructure, simplify |

### Priority Order

Security > Bug > Performance > Refactor > Clean

```rust
// Message contains both "security" and "bug"
let diff = CommitDiff::new()
    .with_message("fix security vulnerability bug");

let features = extractor.extract(&diff);
assert_eq!(features.defect_category, 2);  // Security takes priority
```

## Complexity Estimation

Complexity delta is estimated from line changes:

```
complexity_delta = (lines_added - lines_deleted) / complexity_factor
```

Default `complexity_factor = 10.0` (approximately 10 lines per complexity point).

```rust
let extractor = CodeFeatureExtractor::new()
    .with_complexity_factor(10.0);

let diff = CommitDiff::new()
    .with_lines_added(100)
    .with_lines_deleted(20);

let features = extractor.extract(&diff);
// (100 - 20) / 10 = 8.0
assert!((features.complexity_delta - 8.0).abs() < f32::EPSILON);
```

## Time-Based Features

Extracts temporal patterns from Unix timestamps:

```rust
// 1700000000 = Tuesday, November 14, 2023 22:13:20 UTC
let diff = CommitDiff::new()
    .with_timestamp(1700000000);

let features = extractor.extract(&diff);
assert_eq!(features.hour_of_day, 22);   // 10 PM UTC
assert_eq!(features.day_of_week, 2);    // Tuesday
```

**Why time matters for defect prediction:**
- Late-night commits (hour 22-4) correlate with higher defect rates
- Friday commits show higher bug introduction rates
- These patterns help ML models learn temporal risk factors

## Batch Processing

Extract features from multiple commits efficiently:

```rust
let diffs = vec![
    CommitDiff::new()
        .with_files_changed(1)
        .with_message("feat: add login"),
    CommitDiff::new()
        .with_files_changed(5)
        .with_message("fix: null pointer crash"),
    CommitDiff::new()
        .with_files_changed(2)
        .with_message("refactor: clean utils"),
];

let features = extractor.extract_batch(&diffs);
assert_eq!(features.len(), 3);
assert_eq!(features[1].defect_category, 1);  // Bug fix
```

## Feature Normalization

Normalize features for ML pipelines using dataset statistics:

```rust
use aprender::synthetic::code_features::FeatureStats;

// Collect statistics from training data
let all_features = extractor.extract_batch(&training_diffs);
let stats = FeatureStats::from_features(&all_features);

// Normalize new features to [0, 1]
let normalized = extractor.normalize(&features, &stats);
```

### FeatureStats

```rust
pub struct FeatureStats {
    pub files_changed_max: f32,
    pub lines_added_max: f32,
    pub lines_deleted_max: f32,
    pub complexity_max: f32,
}
```

## Derived Metrics

### Churn

Total lines modified (useful for change-proneness analysis):

```rust
let features = CommitFeatures {
    lines_added: 100.0,
    lines_deleted: 50.0,
    ..Default::default()
};

let churn = features.churn();        // 150.0
let net = features.net_change();     // 50.0
```

### Fix Detection

Check if commit is a bug fix:

```rust
if features.is_fix() {
    println!("This commit fixes a bug");
}
```

## Custom Keywords

Extend keyword sets for domain-specific classification:

```rust
let mut extractor = CodeFeatureExtractor::new();

// Add custom bug keywords
extractor.add_bug_keywords(&["glitch", "oops", "typo"]);

// Add custom security keywords
extractor.add_security_keywords(&["hack", "breach", "leak"]);
```

## Integration with aprender-shell

The `aprender-shell` CLI includes an `analyze` command:

```bash
# Analyze recent commits
aprender-shell analyze

# Output:
# Commit Analysis (last 10 commits):
#   abc123: [BUG] fix: resolve null pointer (churn: 45)
#   def456: [CLEAN] feat: add dashboard (churn: 230)
#   ghi789: [PERF] optimize: cache queries (churn: 12)
```

## ML Pipeline Example

Train a defect predictor using extracted features:

```rust
use aprender::classification::LogisticRegression;

// Extract features from historical commits
let features: Vec<Vec<f32>> = commits
    .iter()
    .map(|c| extractor.extract(c).to_vec())
    .collect();

// Labels: 1 = introduced defect, 0 = clean
let labels: Vec<f32> = commits
    .iter()
    .map(|c| if c.had_defect { 1.0 } else { 0.0 })
    .collect();

// Train classifier
let mut model = LogisticRegression::default();
model.fit(&features, &labels)?;

// Predict defect probability for new commit
let new_features = extractor.extract(&new_commit).to_vec();
let defect_prob = model.predict_proba(&[new_features])?;
```

## Use Cases

### 1. CI/CD Risk Scoring

Flag high-risk commits before merge:

```rust
fn risk_score(features: &CommitFeatures) -> f32 {
    let mut score = 0.0;

    // Large changes are riskier
    if features.files_changed > 10.0 { score += 0.2; }
    if features.churn() > 500.0 { score += 0.3; }

    // Late-night commits
    if features.hour_of_day >= 22 || features.hour_of_day <= 4 {
        score += 0.15;
    }

    // Friday commits
    if features.day_of_week == 5 { score += 0.1; }

    // Bug fixes might introduce new bugs
    if features.is_fix() { score += 0.1; }

    score.min(1.0)
}
```

### 2. Developer Analytics

Track individual developer patterns:

```rust
let dev_commits: Vec<CommitFeatures> = /* ... */;

let avg_churn = dev_commits.iter()
    .map(|f| f.churn())
    .sum::<f32>() / dev_commits.len() as f32;

let fix_rate = dev_commits.iter()
    .filter(|f| f.is_fix())
    .count() as f32 / dev_commits.len() as f32;

println!("Avg churn: {:.0} lines, Fix rate: {:.1}%",
    avg_churn, fix_rate * 100.0);
```

### 3. Technical Debt Tracking

Monitor complexity growth over time:

```rust
let weekly_delta: f32 = week_commits
    .iter()
    .map(|f| f.complexity_delta)
    .sum();

if weekly_delta > 50.0 {
    println!("Warning: Significant complexity increase this week");
}
```

## Performance

| Operation | Complexity | Throughput |
|-----------|------------|------------|
| Single extraction | O(m) | ~1M commits/sec |
| Batch extraction | O(n*m) | ~500K commits/sec |
| Normalization | O(1) | ~10M/sec |

Where m = message length, n = batch size.

## References

- D'Ambros et al. (2012). "Evaluating Defect Prediction Approaches: A Benchmark and an Extensive Comparison"
- Mockus & Votta (2000). "Identifying Reasons for Software Changes Using Historic Databases"
- Hassan (2009). "Predicting Faults Using the Complexity of Code Changes"

## See Also

- [CodeEDA](./code-eda.md) - Code-aware data augmentation
- [Synthetic Data Generation](./synthetic-data-generation.md) - General synthetic data techniques
- [Shell Completion](./shell-completion.md) - AI-powered shell autocomplete
