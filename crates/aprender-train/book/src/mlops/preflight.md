# Preflight Validation

Catch data and environment issues before training starts. Research shows preflight validation prevents 30-50% of ML
pipeline failures.

## Toyota Principle: Jidoka (自働化)

Built-in quality through automatic defect detection at source. Stop the line immediately when problems are detected.

## Quick Start

```rust
use entrenar::storage::{Preflight, PreflightCheck, PreflightContext};

// Standard data integrity checks
let preflight = Preflight::standard();

// Run checks
let results = preflight.run(&data);

if results.all_passed() {
    println!("All preflight checks passed!");
} else {
    eprintln!("{}", results.report());
    std::process::exit(1);
}
```

## Built-in Checks

### Data Integrity Checks

```rust
// No NaN values
PreflightCheck::no_nan_values()

// No infinite values
PreflightCheck::no_inf_values()

// Minimum sample count
PreflightCheck::min_samples(1000)

// Minimum feature count
PreflightCheck::min_features(10)

// Consistent dimensions (all rows same length)
PreflightCheck::consistent_dimensions()

// No constant features (zero variance)
PreflightCheck::no_constant_features()

// Class imbalance check
PreflightCheck::label_balance(5.0)  // max 5:1 ratio
```

### Environment Checks

```rust
// Disk space check
PreflightCheck::disk_space_mb(10240)  // 10GB minimum

// Memory check
PreflightCheck::memory_mb(8192)  // 8GB minimum

// GPU availability
PreflightCheck::gpu_available()
```

## Preflight Presets

```rust
// Standard data checks (NaN, Inf, dimensions, constant features)
let preflight = Preflight::standard();

// Comprehensive (data + environment)
let preflight = Preflight::comprehensive();

// Custom combination
let preflight = Preflight::new()
    .add_check(PreflightCheck::no_nan_values())
    .add_check(PreflightCheck::min_samples(500))
    .add_check(PreflightCheck::disk_space_mb(5120));
```

## Check Results

```rust
let results = preflight.run(&data);

// Overall status
println!("Passed: {}", results.all_passed());

// Counts
println!("Passed: {}", results.passed_count());
println!("Failed: {}", results.failed_count());
println!("Warnings: {}", results.warning_count());
println!("Skipped: {}", results.skipped_count());

// Get failed checks
for (check, result) in results.failed_checks() {
    eprintln!("FAILED: {} - {:?}", check.name, result);
}

// Get warnings
for (check, result) in results.warnings() {
    println!("WARNING: {} - {:?}", check.name, result);
}

// Formatted report
println!("{}", results.report());
```

## Result Types

```rust
use entrenar::storage::CheckResult;

// Passed
CheckResult::passed("All values valid")

// Failed with details
CheckResult::failed_with_details(
    "Found NaN values",
    "First locations: (0, 3), (5, 7)"
)

// Warning (non-fatal)
CheckResult::warning("High class imbalance detected")

// Skipped
CheckResult::skipped("No data to check")
```

## Optional vs Required Checks

```rust
// Required check (blocks training if failed)
PreflightCheck::no_nan_values()

// Optional check (warning only)
PreflightCheck::no_constant_features().optional()
PreflightCheck::label_balance(5.0).optional()
```

## Context-Based Thresholds

```rust
use entrenar::storage::PreflightContext;

let ctx = PreflightContext::new()
    .with_min_samples(10000)
    .with_min_features(100)
    .with_min_disk_space_mb(51200)
    .with_min_memory_mb(32768);

let preflight = Preflight::new()
    .add_check(PreflightCheck::min_samples(1))  // Default overridden by context
    .with_context(ctx);
```

## Validate or Fail

```rust
// Validate and return error if failed
let results = preflight.validate(&data)?;

// This will return Err(PreflightError::ValidationFailed) if checks fail
```

## Integration with Training

```rust
use entrenar::storage::Preflight;
use entrenar::train::Trainer;

fn train_with_preflight(data: &[Vec<f64>]) -> Result<(), Box<dyn std::error::Error>> {
    // Run preflight checks
    let preflight = Preflight::comprehensive();
    preflight.validate(data)?;

    // Proceed with training
    let trainer = Trainer::new(config);
    trainer.fit(data)?;

    Ok(())
}
```

## Cargo Run Example

```bash
# Run preflight validation
cargo run --example preflight_check -- --data train.parquet

# With verbose output
cargo run --example preflight_check -- --data train.parquet --verbose
```

## Sample Report Output

```
=== Preflight Check Results ===
Status: FAILED
Passed: 4, Failed: 1, Warnings: 1, Skipped: 0

✓ no_nan_values: No NaN values found
✓ no_inf_values: No infinite values found
✓ consistent_dimensions: All 10000 rows have 128 features
✗ min_samples: Only 500 samples found (minimum: 1000)
⚠ no_constant_features: Found 2 constant feature(s): [45, 89]
✓ disk_space: 52480 MB available (minimum: 10240 MB)
```

## Best Practices

1. **Run preflight before every training job** - Catches issues early
2. **Use comprehensive preset for production** - Includes environment checks
3. **Make class balance optional** - Not always applicable
4. **Set appropriate thresholds** - Too strict causes false positives
5. **Log preflight results** - Useful for debugging failed runs

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Quality Gates (Jidoka)](../monitor/quality-gates.md)
