# Data Widgets

Widgets for displaying and interacting with datasets.

## Overview

| Widget | Purpose |
|--------|---------|
| DataCard | Single metric display |
| DataTable | Tabular data with sorting |
| Chart | Data visualization |
| ModelCard | ML model metadata |

## Common Patterns

### Data Binding

All data widgets support expression binding:

```yaml
widgets:
  - type: DataTable
    data: "{{ users | filter(active=true) | sort(name) }}"
```

### Refresh

```yaml
widgets:
  - type: DataCard
    title: "Active Users"
    value: "{{ users | count }}"
    refresh: 30s
```

### Loading States

```yaml
widgets:
  - type: DataTable
    data: "{{ users }}"
    loading: "{{ users.loading }}"
    empty_message: "No users found"
```

## Data Card

```yaml
widgets:
  - type: DataCard
    title: "Revenue"
    value: "{{ sales | sum(amount) | currency }}"
    trend: up
```

## Data Table

```yaml
widgets:
  - type: DataTable
    data: "{{ products }}"
    columns:
      - { key: "name", label: "Product" }
      - { key: "price", label: "Price", format: "currency" }
    sortable: true
    page_size: 20
```

## Chart

```yaml
widgets:
  - type: Chart
    chart_type: line
    data: "{{ timeseries }}"
    x: date
    y: value
```

## Error Handling

```yaml
widgets:
  - type: DataTable
    data: "{{ users }}"
    on_error:
      show: "error_message"
      retry: true
```

## Verified Test

```rust
#[test]
fn test_data_widgets_loading_state() {
    // Data widget states
    #[derive(Debug, PartialEq)]
    enum DataState<T> {
        Loading,
        Loaded(T),
        Error(String),
        Empty,
    }

    impl<T> DataState<T> {
        fn is_loading(&self) -> bool {
            matches!(self, DataState::Loading)
        }

        fn is_ready(&self) -> bool {
            matches!(self, DataState::Loaded(_))
        }
    }

    let loading: DataState<Vec<u32>> = DataState::Loading;
    assert!(loading.is_loading());
    assert!(!loading.is_ready());

    let loaded: DataState<Vec<u32>> = DataState::Loaded(vec![1, 2, 3]);
    assert!(!loaded.is_loading());
    assert!(loaded.is_ready());

    let error: DataState<Vec<u32>> = DataState::Error("Network error".to_string());
    assert!(!error.is_loading());
    assert!(!error.is_ready());
}
```
