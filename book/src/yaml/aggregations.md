# Aggregations

Aggregate functions for data summarization.

## Functions

| Function | Description | Example |
|----------|-------------|---------|
| `count` | Number of rows | `{{ users \| count }}` |
| `sum(f)` | Sum of field | `{{ orders \| sum(amount) }}` |
| `mean(f)` | Average | `{{ scores \| mean(value) }}` |
| `min(f)` | Minimum | `{{ prices \| min(cost) }}` |
| `max(f)` | Maximum | `{{ prices \| max(cost) }}` |
| `median(f)` | Median value | `{{ ages \| median(age) }}` |
| `stddev(f)` | Standard deviation | `{{ data \| stddev(value) }}` |

## Count

```yaml
# Total rows
total: "{{ users | count }}"

# Filtered count
active_users: "{{ users | filter(active=true) | count }}"
```

## Sum

```yaml
# Total revenue
revenue: "{{ orders | sum(amount) }}"

# With filter
today_sales: "{{ orders | filter(date=today) | sum(amount) }}"
```

## Mean (Average)

```yaml
# Average order value
aov: "{{ orders | mean(amount) }}"

# Display formatted
aov_display: "{{ orders | mean(amount) | currency }}"
```

## Min/Max

```yaml
# Price range
lowest: "{{ products | min(price) }}"
highest: "{{ products | max(price) }}"
```

## Rate Calculations

```yaml
# Events per second
rate: "{{ events | rate(1s) }}"

# Requests per minute
rpm: "{{ requests | rate(1m) }}"
```

## Percentage

```yaml
# Convert ratio to percentage
growth: "{{ (current - previous) / previous | percentage }}"

# Error rate
error_rate: "{{ errors | count }} / {{ total | count }} | percentage }}"
```

## Verified Test

```rust
#[test]
fn test_aggregation_functions() {
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    // count
    let count = values.len();
    assert_eq!(count, 5);

    // sum
    let sum: f32 = values.iter().sum();
    assert_eq!(sum, 150.0);

    // mean
    let mean = sum / count as f32;
    assert_eq!(mean, 30.0);

    // min
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    assert_eq!(min, 10.0);

    // max
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(max, 50.0);

    // median (sorted middle value)
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    assert_eq!(median, 30.0);
}
```
