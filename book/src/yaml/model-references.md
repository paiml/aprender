# Model References

Loading ML models in YAML manifests.

## Basic Reference

```yaml
data:
  classifier:
    model: "models/classifier.apr"
```

## Model Card

Every model reference requires a model card:

```yaml
data:
  classifier:
    model: "models/classifier.apr"
    model_card:
      name: "Digit Classifier"
      version: "1.0.0"
      task: "classification"
      accuracy: 0.98
      limitations:
        - "28x28 grayscale only"
```

## Inference

```yaml
widgets:
  - type: Text
    value: "{{ classifier.predict(input) }}"
```

## Input Binding

```yaml
widgets:
  - type: Canvas
    id: draw_area
    on_draw:
      action: predict
      model: classifier
      input: "{{ draw_area.pixels }}"
      output: prediction
```

## Model Loading

| Stage | Action |
|-------|--------|
| Parse | Read `.apr` header |
| Load | Load weights to memory |
| Warm | Run dummy inference |
| Ready | Available for use |

## Error Handling

```yaml
data:
  model:
    source: "classifier.apr"
    on_error:
      action: fallback
      value: "Model unavailable"
```

## Multiple Models

```yaml
data:
  digit_model:
    model: "digit_classifier.apr"

  text_model:
    model: "text_classifier.apr"

widgets:
  - type: Text
    value: "Digit: {{ digit_model.predict(input) }}"
  - type: Text
    value: "Text: {{ text_model.predict(input) }}"
```

## Verified Test

```rust
#[test]
fn test_model_reference_card_validation() {
    // Model card must have required fields
    struct ModelCard {
        name: String,
        version: String,
        task: String,
        accuracy: f32,
        limitations: Vec<String>,
    }

    impl ModelCard {
        fn validate(&self) -> Result<(), &'static str> {
            if self.name.is_empty() { return Err("name required"); }
            if self.version.is_empty() { return Err("version required"); }
            if self.task.is_empty() { return Err("task required"); }
            if self.accuracy < 0.0 || self.accuracy > 1.0 {
                return Err("accuracy must be 0.0-1.0");
            }
            if self.limitations.is_empty() {
                return Err("limitations required");
            }
            Ok(())
        }
    }

    let valid = ModelCard {
        name: "Classifier".to_string(),
        version: "1.0.0".to_string(),
        task: "classification".to_string(),
        accuracy: 0.98,
        limitations: vec!["Grayscale only".to_string()],
    };
    assert!(valid.validate().is_ok());

    let invalid = ModelCard {
        name: "".to_string(),
        ..valid
    };
    assert!(invalid.validate().is_err());
}
```
