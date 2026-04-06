# Property-Based Testing

Property-based testing generates random inputs to verify that code
maintains invariants across many cases.

## Using Proptest

```toml
[dev-dependencies]
proptest = "1.0"
```

## Basic Property Test

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_linear_scale_roundtrip(
        domain_min in -1000.0f32..1000.0,
        domain_max in -1000.0f32..1000.0,
        value in -1000.0f32..1000.0,
    ) {
        prop_assume!(domain_min < domain_max);
        prop_assume!(value >= domain_min && value <= domain_max);

        let scale = LinearScale::new()
            .domain(domain_min, domain_max)
            .range(0.0, 100.0);

        let transformed = scale.transform(value);
        let recovered = scale.inverse(transformed);

        prop_assert!((value - recovered).abs() < 0.001);
    }
}
```

## Scale Invariants

```rust
proptest! {
    #[test]
    fn test_scale_preserves_order(
        values in prop::collection::vec(-1000.0f32..1000.0, 2..100),
    ) {
        let scale = LinearScale::new().domain(-1000.0, 1000.0).range(0.0, 800.0);

        let transformed: Vec<f32> = values.iter()
            .map(|v| scale.transform(*v))
            .collect();

        // Order should be preserved
        for i in 0..values.len()-1 {
            if values[i] < values[i+1] {
                prop_assert!(transformed[i] < transformed[i+1]);
            } else if values[i] > values[i+1] {
                prop_assert!(transformed[i] > transformed[i+1]);
            }
        }
    }
}
```

## Histogram Properties

```rust
proptest! {
    #[test]
    fn test_histogram_sum_equals_input_count(
        data in prop::collection::vec(-100.0f32..100.0, 1..1000),
    ) {
        let hist = Histogram::new(&data).bins(10).build();

        let bin_sum: usize = hist.bins().iter().map(|b| b.count).sum();

        prop_assert_eq!(bin_sum, data.len());
    }

    #[test]
    fn test_histogram_all_values_in_bins(
        data in prop::collection::vec(0.0f32..100.0, 1..100),
    ) {
        let hist = Histogram::new(&data).bins(10).build();

        for value in &data {
            let in_some_bin = hist.bins().iter()
                .any(|b| *value >= b.min && *value < b.max);
            prop_assert!(in_some_bin, "Value {} not in any bin", value);
        }
    }
}
```

## Color Invariants

```rust
proptest! {
    #[test]
    fn test_rgba_to_hsla_roundtrip(
        r in 0u8..=255,
        g in 0u8..=255,
        b in 0u8..=255,
        a in 0u8..=255,
    ) {
        let rgba = Rgba::new(r, g, b, a);
        let hsla = Hsla::from(rgba);
        let back = Rgba::from(hsla);

        // Allow small rounding errors
        prop_assert!((rgba.r as i16 - back.r as i16).abs() <= 1);
        prop_assert!((rgba.g as i16 - back.g as i16).abs() <= 1);
        prop_assert!((rgba.b as i16 - back.b as i16).abs() <= 1);
        prop_assert_eq!(rgba.a, back.a);
    }
}
```

## Geometry Properties

```rust
proptest! {
    #[test]
    fn test_point_distance_symmetric(
        x1 in -1000.0f32..1000.0,
        y1 in -1000.0f32..1000.0,
        x2 in -1000.0f32..1000.0,
        y2 in -1000.0f32..1000.0,
    ) {
        let p1 = Point::new(x1, y1);
        let p2 = Point::new(x2, y2);

        prop_assert!((p1.distance(p2) - p2.distance(p1)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_contains_corners(
        x in -1000.0f32..1000.0,
        y in -1000.0f32..1000.0,
        w in 0.1f32..1000.0,
        h in 0.1f32..1000.0,
    ) {
        let rect = Rect::new(x, y, w, h);

        prop_assert!(rect.contains(Point::new(x, y)));
        prop_assert!(rect.contains(Point::new(x + w, y)));
        prop_assert!(rect.contains(Point::new(x, y + h)));
        prop_assert!(rect.contains(Point::new(x + w, y + h)));
    }
}
```

## Box Plot Statistics Properties

```rust
proptest! {
    #[test]
    fn test_boxplot_quartile_ordering(
        data in prop::collection::vec(-1000.0f32..1000.0, 5..100),
    ) {
        let boxplot = BoxPlot::new(&data).build();
        let stats = boxplot.statistics();

        prop_assert!(stats.min <= stats.q1);
        prop_assert!(stats.q1 <= stats.median);
        prop_assert!(stats.median <= stats.q3);
        prop_assert!(stats.q3 <= stats.max);
    }
}
```

## Running Property Tests

```bash
# Run all property tests
PROPTEST_CASES=1000 cargo test

# Increase case count for more thorough testing
PROPTEST_CASES=10000 cargo test property

# Show regression file location on failure
PROPTEST_VERBOSE=1 cargo test
```

## Shrinking

When a test fails, proptest automatically "shrinks" the input to find
the minimal failing case:

```text
test test_histogram_invariant ... FAILED

thread panicked: assertion failed
Minimal failing input: data = [0.0, 0.0]
```

## Complete Example

```rust
use proptest::prelude::*;
use trueno_viz::plots::ScatterPlot;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_scatter_data_integrity(
        x_data in prop::collection::vec(-1e6f32..1e6, 1..1000),
        y_data in prop::collection::vec(-1e6f32..1e6, 1..1000),
    ) {
        let len = x_data.len().min(y_data.len());
        let x = &x_data[..len];
        let y = &y_data[..len];

        let plot = ScatterPlot::new()
            .x(x)
            .y(y)
            .build();

        // Data integrity
        prop_assert_eq!(plot.len(), len);
        prop_assert_eq!(plot.x_data(), x);
        prop_assert_eq!(plot.y_data(), y);

        // Range calculation
        let (x_min, x_max) = plot.x_range();
        let (y_min, y_max) = plot.y_range();

        for val in x {
            prop_assert!(*val >= x_min && *val <= x_max);
        }
        for val in y {
            prop_assert!(*val >= y_min && *val <= y_max);
        }
    }
}
```

## Next Chapter

Continue to [Coverage Requirements](./coverage.md) for quality metrics.
