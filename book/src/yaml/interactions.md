# Interactions

Event handling and user interaction in YAML.

## Event Types

| Event | Trigger | Example |
|-------|---------|---------|
| `on_click` | Mouse click/tap | Button press |
| `on_change` | Value change | Input field |
| `on_submit` | Form submit | Form completion |
| `on_hover` | Mouse enter | Tooltip |
| `on_load` | Component load | Initial fetch |

## Click Handling

```yaml
widgets:
  - type: Button
    label: "Submit"
    on_click:
      action: submit
      target: form_data
```

## Change Handling

```yaml
widgets:
  - type: TextInput
    id: search
    on_change:
      action: filter
      target: results
      value: "{{ search.value }}"
```

## Action Types

| Action | Description |
|--------|-------------|
| `navigate` | Go to route |
| `update` | Update state |
| `filter` | Filter data |
| `submit` | Submit form |
| `refresh` | Refresh data |
| `toggle` | Toggle boolean |

## Navigation

```yaml
widgets:
  - type: Button
    label: "Details"
    on_click:
      action: navigate
      target: "/details/{{ item.id }}"
```

## State Updates

```yaml
widgets:
  - type: Button
    label: "Increment"
    on_click:
      action: update
      target: counter
      value: "{{ counter + 1 }}"
```

## Conditional Actions

```yaml
widgets:
  - type: Button
    on_click:
      - condition: "{{ form.valid }}"
        action: submit
      - condition: "{{ !form.valid }}"
        action: show_errors
```

## Debouncing

```yaml
widgets:
  - type: TextInput
    on_change:
      action: search
      debounce: 300ms
```

## Verified Test

```rust
#[test]
fn test_interactions_action_dispatch() {
    // Action enum for type-safe dispatch
    #[derive(Debug, PartialEq)]
    enum Action {
        Navigate(String),
        Update { target: String, value: i32 },
        Refresh(String),
        Toggle(String),
    }

    fn dispatch(action: &Action) -> &'static str {
        match action {
            Action::Navigate(_) => "navigating",
            Action::Update { .. } => "updating",
            Action::Refresh(_) => "refreshing",
            Action::Toggle(_) => "toggling",
        }
    }

    let nav = Action::Navigate("/home".to_string());
    assert_eq!(dispatch(&nav), "navigating");

    let update = Action::Update {
        target: "counter".to_string(),
        value: 5,
    };
    assert_eq!(dispatch(&update), "updating");

    // Action values are preserved
    if let Action::Update { target, value } = update {
        assert_eq!(target, "counter");
        assert_eq!(value, 5);
    }
}
```
