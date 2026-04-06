# Drift Detection Simulation

This example demonstrates drift detection with KS test and PSI, showing how to set up baseline distributions, detect
drift, and use Andon callbacks.

## Running the Example

```bash
cargo run --example drift_simulation
```

## Code

```rust
{{#include ../../../examples/drift_simulation.rs}}
```

## Expected Output

```
=== Drift Detection Simulation ===

Generating baseline data (1000 samples, 2 features)...
Baseline set.

--- Test 1: Same Distribution (No Drift Expected) ---
Result: No drift detected (as expected)

--- Test 2: Shifted Distribution (Drift Expected) ---
Shifting feature 1 mean from 50 to 80 (+3 std devs)
ANDON ALERT: Drift detected!
  - feature_0 (Kolmogorov-Smirnov): statistic=0.7290, severity=Critical
  - feature_0 (PSI): statistic=7.3945, severity=Critical
Result: Drift detected in 2 tests

--- Test 3: Completely Different Distribution ---
Both features shifted significantly
ANDON ALERT: Drift detected!
Summary:
  Total features checked: 4
  Features with drift: 4
  Critical alerts: 4
  Drift percentage: 100.0%

=== Summary ===
Andon callback was triggered 2 time(s)
```

## Key Concepts

### Setting Up Drift Detection

```rust
// Create detector with multiple tests
let mut detector = DriftDetector::new(vec![
    DriftTest::KS { threshold: 0.05 },   // Kolmogorov-Smirnov
    DriftTest::PSI { threshold: 0.1 },   // Population Stability Index
]);

// Set baseline from training data
detector.set_baseline(&training_data);
```

### Andon Callbacks (Jidoka)

```rust
// Register callback for drift events
detector.on_drift(|results| {
    println!("ANDON ALERT: Drift detected!");
    for r in results.iter().filter(|r| r.drifted) {
        println!("  - {} ({}): severity={:?}",
            r.feature, r.test.name(), r.severity);
    }
});

// Check and trigger callbacks
let results = detector.check_and_trigger(&new_data);
```

### Interpreting Results

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No significant drift |
| 0.1 - 0.2 | Moderate drift (warning) |
| > 0.2 | Significant drift (critical) |

| KS p-value | Interpretation |
|------------|----------------|
| > 0.05 | Same distribution (no drift) |
| < 0.05 | Different distribution (drift) |
