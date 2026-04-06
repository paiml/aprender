# Charts

Comprehensive data visualization with Chart widgets. Presentar provides a rich set of chart types with full test coverage.

## Chart Types

| Type | Use Case | Example |
|------|----------|---------|
| Line | Trends over time | `cht_sparkline` |
| Bar | Category comparison | `cht_boxplot` |
| Pie/Donut | Part of whole | `cht_donut` |
| Scatter/Bubble | Correlation | `cht_scatter_bubble` |
| Area | Cumulative values | `cht_area_stacked` |
| Heatmap | 2D density | `cht_heatmap_basic` |
| Multi-Axis | Dual metrics | `cht_multi_axis` |

## Scatter Plot with Size (CHT-004)

Bubble charts map a third dimension to point radius:

```rust
// From cht_scatter_bubble.rs
pub struct BubbleChart {
    points: Vec<BubblePoint>,
    min_radius: f32,
    max_radius: f32,
}

impl BubbleChart {
    pub fn size_to_radius(&self, size: f32) -> f32 {
        let (min_size, max_size) = self.size_range();
        if (max_size - min_size).abs() < 0.0001 {
            return (self.min_radius + self.max_radius) / 2.0;
        }
        let normalized = (size - min_size) / (max_size - min_size);
        self.min_radius + normalized * (self.max_radius - self.min_radius)
    }
}
```

Run: `cargo run --example cht_scatter_bubble`

## Heatmap (CHT-005)

2D heatmaps with colormap support:

```rust
// From cht_heatmap_basic.rs
pub enum Colormap {
    Viridis, Plasma, Inferno, Blues, Reds, Greens, Grayscale
}

impl Colormap {
    pub fn map(&self, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => {
                let r = 0.267 + t * (0.993 - 0.267);
                let g = 0.004 + t * (0.906 - 0.004);
                let b = 0.329 + t * (0.143 - 0.329);
                Color::new(r, g, b, 1.0)
            }
            // ... other colormaps
        }
    }
}
```

Run: `cargo run --example cht_heatmap_basic`

## Box Plot (CHT-006)

Statistical box plots with quartile calculation:

```rust
// From cht_boxplot.rs
pub struct BoxPlotStats {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
    pub mean: f32,
    pub outliers: Vec<f32>,
}

impl BoxPlotStats {
    pub fn from_data(data: &[f32]) -> Option<Self> {
        // Calculates quartiles, IQR, and detects outliers
        // using 1.5 * IQR fence rule
    }

    pub fn iqr(&self) -> f32 {
        self.q3 - self.q1
    }
}
```

Run: `cargo run --example cht_boxplot`

## Stacked Area Chart (CHT-007)

Area charts with proper stacking order:

```rust
// From cht_area_stacked.rs
impl StackedAreaChart {
    pub fn stacked_values(&self) -> Vec<Vec<f32>> {
        let n = self.data_points();
        let mut result = Vec::with_capacity(self.series.len());
        let mut cumulative = vec![0.0f32; n];

        for series in &self.series {
            let mut stacked = Vec::with_capacity(n);
            for (i, &val) in series.values.iter().enumerate() {
                cumulative[i] += val;
                stacked.push(cumulative[i]);
            }
            result.push(stacked);
        }
        result
    }
}
```

Run: `cargo run --example cht_area_stacked`

## Donut Chart (CHT-008)

Pie charts with configurable inner radius and center metric:

```rust
// From cht_donut.rs
pub struct DonutChart {
    segments: Vec<DonutSegment>,
    inner_radius_ratio: f32,  // 0.0 = pie, 0.6 = donut
    center_label: Option<String>,
    center_value: Option<String>,
}

impl DonutChart {
    pub fn segment_angles(&self, index: usize) -> Option<(f32, f32)> {
        // Returns (start_angle, end_angle) in radians
        // Starting at 12 o'clock (-π/2)
    }
}
```

Run: `cargo run --example cht_donut`

## Sparkline (CHT-009)

Compact inline charts for dashboards:

```rust
// From cht_sparkline.rs
impl Sparkline {
    pub fn render_inline(&self) -> String {
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        self.values
            .iter()
            .map(|&v| {
                let normalized = self.normalize(v);
                let idx = ((normalized * 7.0).round() as usize).min(7);
                blocks[idx]
            })
            .collect()
    }

    pub fn trend_percentage(&self) -> f32 {
        // Calculate percentage change from first to last value
    }
}
```

Run: `cargo run --example cht_sparkline`

## Multi-Axis Chart (CHT-010)

Dual y-axis for correlation visualization:

```rust
// From cht_multi_axis.rs
impl MultiAxisChart {
    pub fn correlation(&self) -> Option<f32> {
        // Calculates Pearson correlation coefficient
        // between left and right axis data
    }

    pub fn normalize(&self, value: f32, axis: AxisSide) -> f32 {
        // Normalizes value to 0-1 range for specific axis
    }
}
```

Run: `cargo run --example cht_multi_axis`

## YAML Configuration

```yaml
widgets:
  - type: Chart
    chart_type: line
    data: "{{ data.timeseries }}"
    x_label: "Time"
    y_label: "Value"
```

## Data Binding

```yaml
data:
  sales:
    source: "sales.ald"

widgets:
  - type: Chart
    chart_type: line
    data: "{{ sales | select('date', 'revenue') }}"
```

## Styling Options

| Property | Description |
|----------|-------------|
| `colors` | Series colors |
| `grid` | Show grid lines |
| `legend` | Legend position |
| `axis_*` | Axis configuration |
| `colormap` | Heatmap colormap |

## Test Coverage

All chart examples include comprehensive tests:

| Example | Tests | Coverage |
|---------|-------|----------|
| cht_scatter_bubble | 6 | Bounds, sizing, transform |
| cht_heatmap_basic | 7 | Colormap, normalization |
| cht_boxplot | 7 | Quartiles, outliers |
| cht_area_stacked | 8 | Stacking, percentages |
| cht_donut | 9 | Angles, segments |
| cht_sparkline | 11 | Trends, rendering |
| cht_multi_axis | 8 | Correlation, normalization |

## Verified Test

```rust
#[test]
fn test_bubble_chart_radius() {
    let mut chart = BubbleChart::new(5.0, 25.0);
    chart.add_point(0.0, 0.0, 10.0, None);
    chart.add_point(100.0, 100.0, 50.0, None);

    // Size 10 is minimum -> min radius
    assert_eq!(chart.size_to_radius(10.0), 5.0);
    // Size 50 is maximum -> max radius
    assert_eq!(chart.size_to_radius(50.0), 25.0);
    // Size 30 is middle -> middle radius
    assert_eq!(chart.size_to_radius(30.0), 15.0);
}
```
