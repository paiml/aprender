# Select

Dropdown selection widget.

## Basic Usage

```rust
use presentar_widgets::Select;

let select = Select::new()
    .option("red", "Red")
    .option("green", "Green")
    .option("blue", "Blue");
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `option(value, label)` | Add option |
| `selected(value)` | Pre-select option |
| `placeholder(str)` | Placeholder when empty |
| `disabled(bool)` | Disable interaction |

## Example

```rust
let country = Select::new()
    .placeholder("Select country...")
    .option("us", "United States")
    .option("uk", "United Kingdom")
    .option("ca", "Canada")
    .selected("us")
    .with_test_id("country-select");
```

## Event Handling

```rust
use presentar_widgets::SelectionChanged;

if let Some(msg) = select.event(&event) {
    if let Some(changed) = msg.downcast_ref::<SelectionChanged>() {
        println!("Selected: {}", changed.value);
    }
}
```

## Getters

```rust
let selected = select.get_selected();      // Option<&str>
let options = select.get_options();        // &[(String, String)]
let is_open = select.is_open();
```

## Programmatic Control

```rust
select.set_selected("blue");
select.open();
select.close();
```

## Accessibility

- Role: `ListBox`
- Keyboard: Arrow keys navigate, Enter selects
- Escape closes dropdown

## Verified Test

```rust
#[test]
fn test_select_options() {
    use presentar_widgets::Select;

    let select = Select::new()
        .option("a", "Option A")
        .option("b", "Option B")
        .selected("a");

    assert_eq!(select.get_selected(), Some("a"));
}
```
