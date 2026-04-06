# Quality & Validation (Examples 46-55)

This section covers data quality checking and validation.

## Examples 46-47: Quality Report and Missing Values

```rust
use alimentar::{ArrowDataset, QualityChecker};

let dataset = ArrowDataset::from_parquet("messy.parquet")?;
let checker = QualityChecker::new();
let report = checker.check(&dataset)?;

println!("Row count: {}", report.row_count);
println!("Issues: {:?}", report.issues);
```

## Examples 48-49: Duplicate and Type Validation

```rust
use alimentar::QualityChecker;

let checker = QualityChecker::new()
    .check_duplicates(true)
    .check_types(true);

let report = checker.check(&dataset)?;
for issue in &report.issues {
    println!("Column {}: {:?}", issue.column, issue.issue_type);
}
```

## Examples 50-51: Range and Cardinality Checks

```rust
use alimentar::{QualityChecker, RangeCheck, CardinalityCheck};

let checker = QualityChecker::new()
    .add_check(RangeCheck::new("age", 0.0, 150.0))
    .add_check(CardinalityCheck::new("category", 100)); // max 100 unique

let report = checker.check(&dataset)?;
```

## Examples 52-53: Constant Detection and Scoring

```rust
use alimentar::QualityChecker;

let checker = QualityChecker::new()
    .detect_constants(true);

let report = checker.check(&dataset)?;
let score = report.quality_score(); // 0.0 to 1.0

println!("Quality score: {:.2}", score);
```

## Examples 54-55: Quality Profiles and Export

```rust
use alimentar::{QualityChecker, QualityProfile};

// Use strict profile
let checker = QualityChecker::with_profile(QualityProfile::Strict);
let report = checker.check(&dataset)?;

// Export to JSON
let json = serde_json::to_string_pretty(&report)?;
std::fs::write("quality_report.json", json)?;
```

## CLI Usage

```bash
# Basic quality report
alimentar quality data.parquet

# With JSON output
alimentar quality data.parquet --format json

# Score only
alimentar quality score data.parquet

# Strict profile
alimentar quality data.parquet --profile strict
```

## Quality Issues Detected

| Issue Type | Description |
|------------|-------------|
| MissingValues | Null/NA values in column |
| Duplicates | Duplicate rows detected |
| TypeMismatch | Value doesn't match schema type |
| OutOfRange | Value outside expected range |
| HighCardinality | Too many unique values |
| ConstantColumn | Column has single value |
| Outliers | Statistical outliers detected |

## Key Concepts

- **Profiles**: Predefined severity thresholds
- **Scoring**: Single metric for quality
- **Issue details**: Column-level diagnostics
- **Export**: JSON/CSV for reporting
