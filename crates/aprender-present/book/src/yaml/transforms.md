# Transforms

Data transformation functions in expressions.

## Overview

```yaml
{{ source | transform1 | transform2 | ... }}
```

Transforms chain left to right, each receiving output from previous.

## Filter Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| `filter(field=value)` | Equality filter | `filter(status=active)` |
| `filter(field>value)` | Comparison | `filter(amount>100)` |
| `filter(field~pattern)` | Regex match | `filter(name~^A)` |

```yaml
# Multiple filters (AND)
{{ data | filter(active=true) | filter(age>18) }}
```

## Selection Transforms

| Transform | Description |
|-----------|-------------|
| `select(f1, f2)` | Keep specific fields |
| `exclude(f1, f2)` | Remove fields |
| `rename(old=new)` | Rename field |

```yaml
{{ users | select(id, name, email) }}
```

## Sorting Transforms

| Transform | Description |
|-----------|-------------|
| `sort(field)` | Ascending sort |
| `sort(field, desc)` | Descending sort |

```yaml
{{ data | sort(created_at, desc) }}
```

## Limiting Transforms

| Transform | Description |
|-----------|-------------|
| `limit(n)` | Take first n |
| `offset(n)` | Skip first n |
| `sample(n)` | Random sample |

```yaml
{{ data | sort(score, desc) | limit(10) }}
```

## Aggregation Transforms

| Transform | Description |
|-----------|-------------|
| `count` | Row count |
| `sum(field)` | Sum values |
| `mean(field)` | Average |
| `min(field)` | Minimum |
| `max(field)` | Maximum |

```yaml
{{ orders | sum(amount) }}
```

## Formatting Transforms

| Transform | Output |
|-----------|--------|
| `percentage` | 85% |
| `currency` | $1,234.56 |
| `date` | Jan 1, 2024 |
| `number(2)` | 1,234.57 |

```yaml
value: "{{ growth | percentage }}"
```

## Verified Test

```rust
#[test]
fn test_transform_chain() {
    // Simulate transform chain
    #[derive(Clone)]
    struct Row { active: bool, amount: f32 }

    let data = vec![
        Row { active: true, amount: 100.0 },
        Row { active: false, amount: 200.0 },
        Row { active: true, amount: 50.0 },
        Row { active: true, amount: 150.0 },
    ];

    // filter(active=true) | sort(amount, desc) | limit(2)
    let mut result: Vec<_> = data.iter()
        .filter(|r| r.active)
        .cloned()
        .collect();
    result.sort_by(|a, b| b.amount.partial_cmp(&a.amount).unwrap());
    let result: Vec<_> = result.into_iter().take(2).collect();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].amount, 150.0);
    assert_eq!(result[1].amount, 100.0);
}
```
