# Manifest Schema

Structure of Presentar YAML application manifests.

## Root Structure

```yaml
app:
  name: "App Name"
  version: "1.0.0"

data:
  # Data source definitions

widgets:
  root:
    # Widget tree

theme:
  # Styling overrides
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `app.name` | string | Application name |
| `widgets.root` | object | Root widget |

## App Section

```yaml
app:
  name: "My Dashboard"
  version: "1.0.0"
  description: "Analytics dashboard"
  author: "Team"
```

## Data Section

```yaml
data:
  users:
    source: "users.ald"

  metrics:
    source: "api/metrics"
    refresh: 30s
```

## Widgets Section

```yaml
widgets:
  root:
    type: Column
    children:
      - type: Text
        value: "Hello"
      - type: Button
        label: "Click"
```

## Theme Section

```yaml
theme:
  colors:
    primary: "#4285f4"
    background: "#ffffff"
  fonts:
    body: "Inter"
    heading: "Inter"
```

## Validation

```rust
fn validate_manifest(yaml: &str) -> Result<Manifest, Error> {
    let doc: Value = serde_yaml::from_str(yaml)?;

    // Required: app.name
    if doc.get("app").and_then(|a| a.get("name")).is_none() {
        return Err(Error::MissingField("app.name"));
    }

    // Required: widgets.root
    if doc.get("widgets").and_then(|w| w.get("root")).is_none() {
        return Err(Error::MissingField("widgets.root"));
    }

    Ok(Manifest::parse(doc)?)
}
```

## Verified Test

```rust
#[test]
fn test_manifest_schema_validation() {
    // Minimal valid manifest
    let valid = r#"
        app:
          name: "Test App"
        widgets:
          root:
            type: Text
            value: "Hello"
    "#;

    // Parse and validate
    let doc: serde_yaml::Value = serde_yaml::from_str(valid).unwrap();

    // Check required fields
    let app_name = doc.get("app")
        .and_then(|a| a.get("name"))
        .and_then(|n| n.as_str());
    assert_eq!(app_name, Some("Test App"));

    let root = doc.get("widgets")
        .and_then(|w| w.get("root"));
    assert!(root.is_some());
}
```
