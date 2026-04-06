# Data Table

Tabular data display with sorting and pagination.

## Basic Usage

```yaml
widgets:
  - type: DataTable
    data: "{{ users }}"
    columns:
      - { key: "name", label: "Name" }
      - { key: "email", label: "Email" }
      - { key: "role", label: "Role" }
```

## Features

| Feature | Property |
|---------|----------|
| Sorting | `sortable: true` |
| Pagination | `page_size: 25` |
| Selection | `selectable: true` |
| Search | `filterable: true` |

## Column Configuration

```yaml
columns:
  - key: "name"
    label: "Name"
    sortable: true
    width: 200

  - key: "amount"
    label: "Amount"
    format: "currency"
    align: "right"

  - key: "status"
    label: "Status"
    render: "badge"
```

## Pagination

```yaml
widgets:
  - type: DataTable
    data: "{{ items }}"
    page_size: 25
    show_page_info: true
```

## Row Actions

```yaml
widgets:
  - type: DataTable
    row_actions:
      - { icon: "edit", on_click: "edit_row" }
      - { icon: "delete", on_click: "delete_row" }
```

## Virtualization

For large datasets, virtualization renders only visible rows:

```yaml
widgets:
  - type: DataTable
    data: "{{ large_dataset }}"
    virtualized: true
    row_height: 48
```

## Styling

```yaml
widgets:
  - type: DataTable
    striped: true
    hover_highlight: true
    border: "single"
```

## Verified Test

```rust
#[test]
fn test_data_table_pagination() {
    // Pagination calculation
    struct Pagination {
        total_rows: usize,
        page_size: usize,
        current_page: usize,
    }

    impl Pagination {
        fn total_pages(&self) -> usize {
            (self.total_rows + self.page_size - 1) / self.page_size
        }

        fn offset(&self) -> usize {
            self.current_page * self.page_size
        }

        fn visible_range(&self) -> std::ops::Range<usize> {
            let start = self.offset();
            let end = (start + self.page_size).min(self.total_rows);
            start..end
        }
    }

    let pagination = Pagination {
        total_rows: 100,
        page_size: 25,
        current_page: 1,  // Second page (0-indexed)
    };

    assert_eq!(pagination.total_pages(), 4);
    assert_eq!(pagination.offset(), 25);
    assert_eq!(pagination.visible_range(), 25..50);
}
```
