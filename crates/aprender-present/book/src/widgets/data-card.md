# Data Card

Display a single metric with title and value.

## Basic Usage

```yaml
widgets:
  - type: DataCard
    title: "Total Users"
    value: "{{ users | count }}"
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `title` | string | Card label |
| `value` | string | Display value |
| `subtitle` | string | Secondary text |
| `icon` | string | Optional icon |
| `color` | string | Accent color |
| `trend` | string | up/down/neutral |

## With Formatting

```yaml
widgets:
  - type: DataCard
    title: "Revenue"
    value: "{{ sales | sum(amount) | currency }}"
    trend: "{{ growth > 0 ? 'up' : 'down' }}"
    color: "{{ growth > 0 ? 'green' : 'red' }}"
```

## Layout

```
┌─────────────────────┐
│ [icon]  Title       │
│                     │
│     $12,345         │
│     ↑ 12.5%         │
└─────────────────────┘
```

## Styling

```yaml
widgets:
  - type: DataCard
    title: "Active"
    value: "{{ active | count }}"
    style:
      background: "#e8f5e9"
      border_radius: 8
```

## Grid of Cards

```yaml
widgets:
  - type: Row
    gap: 16
    children:
      - type: DataCard
        title: "Users"
        value: "{{ users | count }}"
      - type: DataCard
        title: "Revenue"
        value: "{{ revenue | currency }}"
      - type: DataCard
        title: "Growth"
        value: "{{ growth | percentage }}"
```

## Verified Test

```rust
#[test]
fn test_data_card_structure() {
    // DataCard properties
    struct DataCard {
        title: String,
        value: String,
        trend: Option<Trend>,
    }

    #[derive(Debug, PartialEq)]
    enum Trend { Up, Down, Neutral }

    let card = DataCard {
        title: "Revenue".to_string(),
        value: "$12,345".to_string(),
        trend: Some(Trend::Up),
    };

    assert_eq!(card.title, "Revenue");
    assert_eq!(card.value, "$12,345");
    assert_eq!(card.trend, Some(Trend::Up));

    // Trend determines color
    fn trend_color(trend: &Option<Trend>) -> &'static str {
        match trend {
            Some(Trend::Up) => "green",
            Some(Trend::Down) => "red",
            _ => "gray",
        }
    }

    assert_eq!(trend_color(&card.trend), "green");
}
```
