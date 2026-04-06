# Expression Language

Presentar's expression language enables data binding in YAML manifests.

## Syntax

```
{{ source | transform | transform | ... }}
```

Expressions are enclosed in `{{ }}` and consist of:
1. A **source** - the data to operate on
2. Zero or more **transforms** - operations applied in sequence

## Examples

```yaml
# Simple data reference
value: "{{ data.users }}"

# With transforms
value: "{{ data.transactions | filter(status=completed) | count }}"

# Chained operations
value: "{{ data.orders | sort(date, desc=true) | limit(10) }}"
```

## Available Transforms

### Filtering

```yaml
# Filter rows where field equals value
{{ data | filter(status=active) }}

# Multiple filters (AND)
{{ data | filter(status=active) | filter(amount>100) }}
```

### Selection

```yaml
# Select specific columns
{{ data | select(id, name, email) }}

# Single column
{{ data | select(amount) }}
```

### Sorting

```yaml
# Sort ascending
{{ data | sort(created_at) }}

# Sort descending
{{ data | sort(created_at, desc=true) }}
```

### Limiting

```yaml
# Take first N rows
{{ data | limit(10) }}

# Sample N random rows
{{ data | sample(100) }}
```

### Aggregations

```yaml
# Count rows
{{ data.users | count }}

# Sum a column
{{ data.orders | sum(amount) }}

# Average a column
{{ data.transactions | mean(value) }}

# Rate over time window
{{ data.events | rate(1m) }}

# Convert to percentage
{{ data.errors | percentage }}
```

### Joins

```yaml
# Join with another dataset
{{ data.orders | join(data.customers, on=customer_id) }}
```

## Implementation

All transforms execute **client-side in WASM**—no server round-trips:

```rust
pub enum Transform {
    Filter { field: String, value: String },
    Select { fields: Vec<String> },
    Sort { field: String, desc: bool },
    Limit { n: usize },
    Count,
    Sum { field: String },
    Mean { field: String },
    Rate { window: String },
    Percentage,
    Join { other: String, on: String },
    Sample { n: usize },
}
```

## Parsing

```rust
use presentar_yaml::ExpressionParser;

let parser = ExpressionParser::new();
let expr = parser.parse("{{ data.transactions | filter(status=completed) | count }}")?;

assert_eq!(expr.source, "data.transactions");
assert_eq!(expr.transforms.len(), 2);
```

## Performance

Transforms are optimized using Trueno SIMD operations:

| Operation | 10K rows | 100K rows | 1M rows |
|-----------|----------|-----------|---------|
| filter | <1ms | ~5ms | ~50ms |
| count | <0.1ms | <1ms | ~5ms |
| sum | <0.5ms | ~2ms | ~20ms |
| sort | ~2ms | ~20ms | ~200ms |

## Next Steps

- [Transforms](./transforms.md) - Detailed transform reference
- [Filters](./filters.md) - Filter syntax and operators
- [Aggregations](./aggregations.md) - Aggregation functions
