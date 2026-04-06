# Test-Driven Visualization

This chapter explains the TDD methodology used throughout trueno-viz
development and how to apply it to your own visualization code.

## The TDD Cycle

```text
┌─────────────────────────────────────────────────────────────┐
│                     TDD CYCLE                               │
│                                                             │
│    ┌───────┐         ┌───────┐         ┌──────────┐        │
│    │  RED  │ ──────▶ │ GREEN │ ──────▶ │ REFACTOR │        │
│    │       │         │       │         │          │        │
│    │ Write │         │ Make  │         │ Improve  │        │
│    │ Test  │         │ Pass  │         │ Code     │        │
│    └───────┘         └───────┘         └──────────┘        │
│         ▲                                    │             │
│         └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## RED Phase: Writing Failing Tests

Before implementing features, write tests that define expected behavior:

```rust
#[test]
fn test_scatter_plot_basic() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .build();

    // Verify data stored correctly
    assert_eq!(plot.x_data(), &x);
    assert_eq!(plot.y_data(), &y);
    assert_eq!(plot.len(), 3);
}
```

Run: `cargo test test_scatter_plot_basic` → FAILS (RED)

## GREEN Phase: Minimal Implementation

Implement just enough to pass:

```rust
impl ScatterPlot {
    pub fn new() -> Self {
        Self {
            x_data: Vec::new(),
            y_data: Vec::new(),
        }
    }

    pub fn x(mut self, data: &[f32]) -> Self {
        self.x_data = data.to_vec();
        self
    }

    pub fn y(mut self, data: &[f32]) -> Self {
        self.y_data = data.to_vec();
        self
    }

    pub fn build(self) -> ScatterPlotResult {
        ScatterPlotResult {
            x_data: self.x_data,
            y_data: self.y_data,
        }
    }
}
```

Run: `cargo test test_scatter_plot_basic` → PASSES (GREEN)

## REFACTOR Phase: Improve Code Quality

Improve without changing behavior:

```rust
impl ScatterPlot {
    pub fn x(mut self, data: &[f32]) -> Self {
        self.x_data = data.to_vec();
        self
    }

    // Refactored: validate data length matches
    pub fn build(self) -> Result<ScatterPlotResult> {
        if self.x_data.len() != self.y_data.len() {
            return Err(Error::DataLengthMismatch);
        }
        Ok(ScatterPlotResult {
            x_data: self.x_data,
            y_data: self.y_data,
        })
    }
}
```

Add test for the new validation:

```rust
#[test]
fn test_scatter_mismatched_lengths() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0];  // Different length!

    let result = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .build();

    assert!(result.is_err());
}
```

## Test Categories

### Unit Tests

Test individual functions in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scale_transform() {
        let scale = LinearScale::new()
            .domain(0.0, 100.0)
            .range(0.0, 800.0);

        assert!((scale.transform(50.0) - 400.0).abs() < f32::EPSILON);
    }
}
```

### Integration Tests

Test components working together:

```rust
// tests/integration_test.rs
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

#[test]
fn test_scatter_to_png() {
    let plot = ScatterPlot::new()
        .x(&[1.0, 2.0, 3.0])
        .y(&[1.0, 4.0, 9.0])
        .build();

    let result = plot.render_to_bytes(800, 600);
    assert!(result.is_ok());

    let bytes = result.unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n"); // PNG magic bytes
}
```

### Edge Case Tests

Test boundary conditions:

```rust
#[test]
fn test_histogram_empty_data() {
    let empty: Vec<f32> = vec![];
    let hist = Histogram::new(&empty).build();
    assert_eq!(hist.bin_count(), 0);
}

#[test]
fn test_histogram_single_value() {
    let single = vec![5.0];
    let hist = Histogram::new(&single).build();
    assert_eq!(hist.bin_count(), 1);
}

#[test]
fn test_histogram_all_same_value() {
    let same = vec![3.0, 3.0, 3.0, 3.0];
    let hist = Histogram::new(&same).build();
    // Should handle gracefully
    assert!(hist.bin_count() >= 1);
}
```

## Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_scatter_plot_basic

# Tests with output
cargo test -- --nocapture

# Tests in specific module
cargo test plots::scatter

# With coverage
cargo llvm-cov --html
```

## Test Organization in trueno-viz

```text
src/
├── plots/
│   ├── scatter.rs     # Contains #[cfg(test)] mod tests
│   ├── histogram.rs   # Contains #[cfg(test)] mod tests
│   └── ...
└── ...

tests/                  # Integration tests
├── plots_test.rs
├── output_test.rs
└── integration_test.rs
```

## Complete TDD Example

Feature: Add regression line to scatter plot

### Step 1: Write Test (RED)

```rust
#[test]
fn test_scatter_with_regression() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .regression_line(true)
        .build();

    // Regression line should be calculated
    let (slope, intercept) = plot.regression_coefficients();
    assert!((slope - 2.0).abs() < 0.1);     // Should be ~2.0
    assert!((intercept - 0.0).abs() < 0.5); // Should be ~0
}
```

### Step 2: Implement (GREEN)

```rust
impl ScatterPlotBuilder {
    pub fn regression_line(mut self, show: bool) -> Self {
        self.show_regression = show;
        self
    }
}

impl ScatterPlotResult {
    pub fn regression_coefficients(&self) -> (f32, f32) {
        let n = self.x_data.len() as f32;
        let sum_x: f32 = self.x_data.iter().sum();
        let sum_y: f32 = self.y_data.iter().sum();
        let sum_xy: f32 = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_x2: f32 = self.x_data.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }
}
```

### Step 3: Refactor

- Extract statistics to separate module
- Add SIMD optimization
- Improve numerical stability

## Next Chapter

Continue to [Property-Based Testing](./property-testing.md) for advanced testing techniques.
