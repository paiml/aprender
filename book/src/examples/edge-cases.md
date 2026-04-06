# Edge Cases

Robust handling of edge cases ensures applications work correctly with international text, extreme values, slow networks, and accessibility requirements.

## Edge Case Categories

| Category | Focus | Example |
|----------|-------|---------|
| Unicode | International text | `edg_unicode` |
| RTL | Right-to-left layouts | `edg_rtl` |
| Numeric | NaN, Infinity handling | `edg_numeric` |
| Slow Data | Loading states | `edg_slow_data` |
| High Cardinality | Large datasets | `edg_high_cardinality` |
| Theming | Dynamic theme switching | `edg_theme_switching` |

## Unicode Handling (EDG-003)

Proper handling of international text, CJK characters, and emoji:

```rust
// From edg_unicode.rs
impl TextMetrics {
    pub fn visual_width(s: &str) -> usize {
        s.chars()
            .map(|c| {
                if c.is_ascii() { 1 }
                else if is_emoji(c) { 2 }
                else if is_wide_char(c) { 2 }
                else { 1 }
            })
            .sum()
    }

    pub fn truncate_to_width(s: &str, max_width: usize) -> String {
        // Truncates string respecting character widths
    }
}
```

### Visual Width Examples

| Text | Chars | Visual Width |
|------|-------|--------------|
| `Hello` | 5 | 5 |
| `你好` | 2 | 4 |
| `🌍` | 1 | 2 |
| `Hello世界` | 7 | 9 |

Run: `cargo run --example edg_unicode`

## Right-to-Left Layout (EDG-004)

Bidirectional text handling for Arabic, Hebrew, and mixed content:

```rust
// From edg_rtl.rs
pub fn detect_direction(text: &str) -> TextDirection {
    let mut rtl_count = 0;
    let mut ltr_count = 0;

    for c in text.chars() {
        if is_rtl_char(c) { rtl_count += 1; }
        else if is_ltr_char(c) { ltr_count += 1; }
    }

    if rtl_count > ltr_count { TextDirection::RightToLeft }
    else if ltr_count > 0 { TextDirection::LeftToRight }
    else { TextDirection::Auto }
}

pub struct RtlTextBox {
    pub text: String,
    pub direction: TextDirection,
    pub alignment: TextAlignment,
}
```

### RTL Alignment

| Alignment | LTR Result | RTL Result |
|-----------|------------|------------|
| Start | Left | Right |
| End | Right | Left |
| Center | Center | Center |

Run: `cargo run --example edg_rtl`

## Numeric Edge Cases (EDG-005)

Safe handling of NaN, Infinity, and division by zero:

```rust
// From edg_numeric.rs
pub enum NumericValue {
    Normal(f64),
    Infinity,
    NegInfinity,
    NaN,
    Zero,
    NegZero,
}

pub fn safe_divide(a: f64, b: f64) -> NumericValue {
    if b == 0.0 {
        if a == 0.0 { NumericValue::NaN }
        else if a.is_sign_positive() { NumericValue::Infinity }
        else { NumericValue::NegInfinity }
    } else {
        NumericValue::from_f64(a / b)
    }
}

impl NumericFormatter {
    pub fn format(&self, value: f64) -> String {
        match NumericValue::from_f64(value) {
            NumericValue::NaN => self.nan_display.clone(),
            NumericValue::Infinity => "∞".to_string(),
            NumericValue::NegInfinity => "-∞".to_string(),
            // ...
        }
    }

    pub fn format_si(&self, value: f64) -> String {
        // Formats with SI prefixes: K, M, B, T
    }
}
```

Run: `cargo run --example edg_numeric`

## Slow/Missing Data (EDG-006)

Graceful handling of network delays and timeouts:

```rust
// From edg_slow_data.rs
pub enum LoadingState<T> {
    Initial,
    Loading { started: Instant },
    Loaded(T),
    Error(String),
    Timeout,
    Stale { data: T, age_secs: u64 },
}

pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub backoff_factor: f64,
}

impl RetryConfig {
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms = self.base_delay_ms as f64
            * self.backoff_factor.powi(attempt as i32);
        Duration::from_millis(delay_ms.min(self.max_delay_ms as f64) as u64)
    }
}
```

### Data Freshness Indicators

| Age | Status | Display |
|-----|--------|---------|
| < 1 min | Fresh | ● Live |
| 1-5 min | Recent | ◐ Updated recently |
| 5-15 min | Stale | ○ May be outdated |
| > 15 min | Very Stale | ✗ Data is stale |

Run: `cargo run --example edg_slow_data`

## High Cardinality Data (EDG-007)

Handling datasets with many unique values:

```rust
// From edg_high_cardinality.rs
pub enum AggregationStrategy {
    TopN(usize),           // Keep top N by count
    Threshold(f64),        // Keep above percentage threshold
    GroupSmall(usize, &'static str), // Group small into "Other"
}

impl CardinalityHandler {
    pub fn aggregate(&self, strategy: AggregationStrategy) -> AggregatedData {
        match strategy {
            AggregationStrategy::TopN(n) => {
                // Keep top N categories, group rest into "Other"
            }
            AggregationStrategy::Threshold(pct) => {
                // Keep categories above percentage threshold
            }
            // ...
        }
    }
}

// Virtualized list for large datasets
pub struct VirtualizedList<T> {
    items: Vec<T>,
    visible_start: usize,
    visible_count: usize,
}
```

Run: `cargo run --example edg_high_cardinality`

## Theme Switching (EDG-010)

Dynamic theme changes without layout shifts:

```rust
// From edg_theme_switching.rs
pub enum ColorRole {
    Background, Surface, Primary, Secondary, Accent,
    Text, TextSecondary, Border,
    Error, Warning, Success, Info,
}

pub struct Theme {
    colors: HashMap<ColorRole, Color>,
    pub border_radius: f32,
    pub spacing_unit: f32,
}

impl Theme {
    pub fn light() -> Self { /* ... */ }
    pub fn dark() -> Self { /* ... */ }
    pub fn high_contrast() -> Self { /* ... */ }
}

impl ThemeManager {
    pub fn interpolate_color(from: Color, to: Color, t: f32) -> Color {
        Color::new(
            from.r + (to.r - from.r) * t,
            from.g + (to.g - from.g) * t,
            from.b + (to.b - from.b) * t,
            from.a + (to.a - from.a) * t,
        )
    }
}
```

### Available Themes

| Theme | Background | Text | Purpose |
|-------|------------|------|---------|
| Light | White | Dark | Default |
| Dark | Dark gray | Light | Low light |
| High Contrast | Black | White | Accessibility |

Run: `cargo run --example edg_theme_switching`

## Test Coverage

| Example | Tests | Coverage |
|---------|-------|----------|
| edg_unicode | 12 | Width, truncation, padding |
| edg_rtl | 12 | Direction, alignment, BiDi |
| edg_numeric | 13 | NaN, infinity, formatting |
| edg_slow_data | 10 | Loading states, retry, freshness |
| edg_high_cardinality | 9 | Aggregation, virtualization |
| edg_theme_switching | 9 | Themes, interpolation |

## Verified Test

```rust
#[test]
fn test_unicode_visual_width() {
    assert_eq!(TextMetrics::visual_width("Hello"), 5);
    assert_eq!(TextMetrics::visual_width("你好"), 4);     // CJK: 2 each
    assert_eq!(TextMetrics::visual_width("🌍"), 2);       // Emoji: 2
    assert_eq!(TextMetrics::visual_width("Hello世界"), 9); // 5 + 4
}

#[test]
fn test_safe_divide() {
    assert!(matches!(safe_divide(10.0, 2.0), NumericValue::Normal(v) if (v - 5.0).abs() < 0.01));
    assert!(matches!(safe_divide(10.0, 0.0), NumericValue::Infinity));
    assert!(matches!(safe_divide(0.0, 0.0), NumericValue::NaN));
}
```
