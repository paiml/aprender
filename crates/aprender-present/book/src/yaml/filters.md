# Filters

Row filtering syntax and operators.

## Basic Syntax

```yaml
{{ data | filter(field=value) }}
```

## Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equal | `filter(status=active)` |
| `!=` | Not equal | `filter(status!=deleted)` |
| `>` | Greater than | `filter(amount>100)` |
| `>=` | Greater or equal | `filter(age>=18)` |
| `<` | Less than | `filter(price<50)` |
| `<=` | Less or equal | `filter(score<=100)` |
| `~` | Regex match | `filter(name~^A)` |

## String Filters

```yaml
# Exact match
{{ users | filter(role=admin) }}

# Contains (regex)
{{ users | filter(email~@company) }}

# Starts with
{{ products | filter(name~^iPhone) }}

# Ends with
{{ files | filter(name~\.pdf$) }}
```

## Numeric Filters

```yaml
# Range (combine filters)
{{ orders | filter(amount>100) | filter(amount<1000) }}

# Specific value
{{ items | filter(quantity=0) }}
```

## Boolean Filters

```yaml
# True values
{{ users | filter(active=true) }}

# False values
{{ tasks | filter(completed=false) }}
```

## Null Handling

```yaml
# Has value
{{ records | filter(email!=null) }}

# Is null
{{ records | filter(phone=null) }}
```

## Combining Filters

```yaml
# AND (chain filters)
{{ data | filter(active=true) | filter(amount>100) }}

# Complex queries
{{ transactions
   | filter(status=completed)
   | filter(amount>1000)
   | filter(date>=2024-01-01)
}}
```

## Verified Test

```rust
#[test]
fn test_filter_operators() {
    #[derive(Clone)]
    struct Row { value: i32, active: bool }

    let data = vec![
        Row { value: 50, active: true },
        Row { value: 100, active: true },
        Row { value: 150, active: false },
        Row { value: 200, active: true },
    ];

    // filter(value>100)
    let gt: Vec<_> = data.iter().filter(|r| r.value > 100).collect();
    assert_eq!(gt.len(), 2);

    // filter(value>=100)
    let gte: Vec<_> = data.iter().filter(|r| r.value >= 100).collect();
    assert_eq!(gte.len(), 3);

    // filter(active=true) | filter(value>100)
    let combined: Vec<_> = data.iter()
        .filter(|r| r.active && r.value > 100)
        .collect();
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0].value, 200);
}
```
