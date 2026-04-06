# Chart

Data visualization widget supporting multiple chart types.

## Chart Types

| Type | Use Case |
|------|----------|
| `line` | Trends over time |
| `bar` | Category comparison |
| `pie` | Part of whole |
| `scatter` | Correlation |
| `area` | Cumulative values |

## Basic Line Chart

```yaml
widgets:
  - type: Chart
    chart_type: line
    data: "{{ timeseries }}"
    x: date
    y: value
```

## Bar Chart

```yaml
widgets:
  - type: Chart
    chart_type: bar
    data:
      - { label: "Q1", value: 100 }
      - { label: "Q2", value: 150 }
      - { label: "Q3", value: 120 }
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `chart_type` | string | line/bar/pie/scatter |
| `data` | array/expr | Data points |
| `x` | string | X-axis field |
| `y` | string | Y-axis field |
| `colors` | array | Series colors |
| `legend` | boolean | Show legend |

## Styling

```yaml
widgets:
  - type: Chart
    chart_type: line
    data: "{{ data }}"
    colors:
      - "#4285f4"
      - "#ea4335"
    grid: true
    legend: bottom
```

## Responsive

```yaml
widgets:
  - type: Chart
    responsive: true
    aspect_ratio: "16:9"
```

## Multiple Series

```yaml
widgets:
  - type: Chart
    chart_type: line
    series:
      - { data: "{{ sales }}", label: "Sales" }
      - { data: "{{ costs }}", label: "Costs" }
```

## Verified Test

```rust
#[test]
fn test_chart_data_scaling() {
    // Chart axis scaling
    fn calculate_axis_range(values: &[f32]) -> (f32, f32) {
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Add 10% padding
        let padding = (max - min) * 0.1;
        (min - padding, max + padding)
    }

    let data = vec![10.0, 25.0, 15.0, 30.0, 20.0];
    let (min, max) = calculate_axis_range(&data);

    // Range should include all values with padding
    assert!(min < 10.0);
    assert!(max > 30.0);

    // Scale value to pixel coordinates
    fn scale(value: f32, min: f32, max: f32, height: f32) -> f32 {
        let normalized = (value - min) / (max - min);
        height * (1.0 - normalized)  // Invert for screen coords
    }

    let y = scale(20.0, min, max, 100.0);
    assert!(y > 0.0 && y < 100.0);
}
```
