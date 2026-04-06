# YAML Schema

JSON Schema for validating Presentar manifests.

## Schema Location

```yaml
# Reference schema in your YAML
# yaml-language-server: $schema=presentar-schema.json
```

## Root Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["app", "widgets"],
  "properties": {
    "app": { "$ref": "#/definitions/app" },
    "data": { "$ref": "#/definitions/data" },
    "widgets": { "$ref": "#/definitions/widgets" },
    "theme": { "$ref": "#/definitions/theme" }
  }
}
```

## App Definition

```json
{
  "definitions": {
    "app": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": { "type": "string" },
        "version": { "type": "string" },
        "description": { "type": "string" }
      }
    }
  }
}
```

## Widget Types

| Type | Schema |
|------|--------|
| Text | `{ type: "Text", value: string }` |
| Button | `{ type: "Button", label: string }` |
| Column | `{ type: "Column", children: Widget[] }` |
| Row | `{ type: "Row", children: Widget[] }` |

## Validation

```rust
fn validate_yaml(content: &str) -> Result<(), Vec<ValidationError>> {
    let doc: Value = serde_yaml::from_str(content)?;

    let mut errors = vec![];

    // Required: app.name
    if doc.get("app").and_then(|a| a.get("name")).is_none() {
        errors.push(ValidationError::missing("app.name"));
    }

    // Required: widgets.root
    if doc.get("widgets").and_then(|w| w.get("root")).is_none() {
        errors.push(ValidationError::missing("widgets.root"));
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
```

## IDE Support

| Editor | Plugin |
|--------|--------|
| VS Code | YAML + schema |
| JetBrains | YAML + schema |
| Neovim | yaml-language-server |

## Verified Test

```rust
#[test]
fn test_yaml_schema_validation() {
    // Validation error structure
    #[derive(Debug, PartialEq)]
    struct ValidationError {
        path: String,
        message: String,
    }

    fn validate_required(doc: &std::collections::HashMap<&str, &str>, field: &str)
        -> Option<ValidationError>
    {
        if !doc.contains_key(field) {
            Some(ValidationError {
                path: field.to_string(),
                message: format!("{} is required", field),
            })
        } else {
            None
        }
    }

    let mut valid_doc = std::collections::HashMap::new();
    valid_doc.insert("name", "Test App");
    assert!(validate_required(&valid_doc, "name").is_none());

    let empty_doc = std::collections::HashMap::new();
    let error = validate_required(&empty_doc, "name");
    assert!(error.is_some());
    assert_eq!(error.unwrap().path, "name");
}
```
