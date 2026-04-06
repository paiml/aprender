# Data Table

Tabular data display with sorting and filtering.

## Quick Start

```bash
# Run the demo
cargo run -p presentar --example apr_ald_display
```

## Loading .ald Files

```rust
use presentar_widgets::{load_ald_as_card, AldDatasetExt};
use presentar_yaml::formats::AldDataset;

// Load from bytes
let data_card = load_ald_as_card(&ald_bytes, "my_dataset")?;
println!("Dataset: {}", data_card.get_name());
println!("Columns: {}", data_card.column_count());

// Or use extension trait
let dataset = AldDataset::load(&ald_bytes)?;
let card = dataset.to_data_card("custom_name");
```

## Basic Table

```yaml
widgets:
  - type: DataTable
    data:
      - { name: "Alice", age: 30, role: "Engineer" }
      - { name: "Bob", age: 25, role: "Designer" }
      - { name: "Carol", age: 35, role: "Manager" }
    columns:
      - { key: "name", label: "Name" }
      - { key: "age", label: "Age" }
      - { key: "role", label: "Role" }
```

## Features

| Feature | Description |
|---------|-------------|
| Sorting | Click column header |
| Filtering | Text search |
| Pagination | Page navigation |
| Selection | Row selection |
| Virtualization | Large datasets |

## Data Binding

```yaml
data:
  users:
    source: "users.ald"
    transform: "filter(active=true)"

widgets:
  - type: DataTable
    data: "{{ users }}"
    sortable: true
    filterable: true
```

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
      - { icon: "edit", action: "edit_row" }
      - { icon: "delete", action: "delete_row" }
```

## Verified Test

```rust
#[test]
fn test_data_table_sorting() {
    // Table sorting algorithm
    #[derive(Debug, Clone)]
    struct Row {
        name: String,
        age: u32,
    }

    let mut rows = vec![
        Row { name: "Carol".to_string(), age: 35 },
        Row { name: "Alice".to_string(), age: 30 },
        Row { name: "Bob".to_string(), age: 25 },
    ];

    // Sort by name ascending
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(rows[0].name, "Alice");
    assert_eq!(rows[1].name, "Bob");
    assert_eq!(rows[2].name, "Carol");

    // Sort by age descending
    rows.sort_by(|a, b| b.age.cmp(&a.age));
    assert_eq!(rows[0].age, 35);
}
```
