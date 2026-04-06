# YAML Configuration

Declarative app configuration with YAML manifests.

## Basic Structure

```yaml
presentar: "0.1"
name: "my-app"
version: "1.0.0"

layout:
  type: "column"
  gap: 16
  children:
    - type: "text"
      content: "Hello"
    - type: "button"
      label: "Click"
```

## Widget Types

| Type | Description |
|------|-------------|
| `text` | Static text display |
| `button` | Clickable button |
| `column` | Vertical layout |
| `row` | Horizontal layout |
| `container` | Single-child wrapper |
| `text_input` | Text entry field |
| `checkbox` | Boolean toggle |
| `slider` | Range input |
| `select` | Dropdown |

## Properties

### Text

```yaml
- type: "text"
  content: "Hello World"
  font_size: 24
  color: "#1f2937"
  weight: "bold"
```

### Button

```yaml
- type: "button"
  label: "Submit"
  background: "#4f46e5"
  padding: 12
  on_click: "submit_form"
```

### Container

```yaml
- type: "container"
  padding: 24
  background: "#ffffff"
  corner_radius: 8
  child:
    type: "text"
    content: "Nested"
```

## Interactions

```yaml
interactions:
  - trigger: "submit_form"
    action: "update_state"
    script: |
      set_state("submitted", true)
```

## Data Binding

```yaml
- type: "text"
  content: "{{ state.counter }}"
```

## Verified Test

```rust
#[test]
fn test_yaml_parse() {
    let yaml = r#"
        presentar: "0.1"
        name: "test"
        version: "1.0.0"
    "#;

    // YAML parsing is handled by serde_yaml
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(value["name"], "test");
}
```
