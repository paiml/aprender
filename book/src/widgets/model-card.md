# Model Card

Display ML model metadata and metrics.

## Purpose

Model cards provide transparency about:
- Model capabilities and limitations
- Performance metrics
- Intended use cases
- Training data characteristics

## Basic Usage

```yaml
widgets:
  - type: ModelCard
    model: "{{ classifier }}"
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model identifier |
| `version` | string | Semantic version |
| `task` | string | Classification, regression, etc. |
| `metrics` | object | Performance numbers |
| `limitations` | array | Known constraints |

## Full Example

```yaml
widgets:
  - type: ModelCard
    name: "Digit Classifier"
    version: "1.0.0"
    task: "Image Classification"
    description: "Classifies handwritten digits 0-9"
    metrics:
      accuracy: 0.98
      precision: 0.97
      recall: 0.98
      f1: 0.975
    training:
      dataset: "MNIST"
      samples: 60000
      epochs: 10
    limitations:
      - "28x28 grayscale images only"
      - "Centered digits perform best"
      - "May struggle with unusual handwriting"
    intended_use:
      - "Educational demos"
      - "Digit recognition applications"
```

## Sections

| Section | Content |
|---------|---------|
| Overview | Name, version, description |
| Metrics | Accuracy, precision, recall |
| Training | Dataset, parameters |
| Limitations | Known constraints |
| Usage | Intended applications |

## Styling

```yaml
widgets:
  - type: ModelCard
    model: "{{ model }}"
    sections:
      - overview
      - metrics
      - limitations
    collapsible: true
```

## Verified Test

```rust
#[test]
fn test_model_card_metrics() {
    // Model card metrics validation
    struct ModelMetrics {
        accuracy: f32,
        precision: f32,
        recall: f32,
    }

    impl ModelMetrics {
        fn f1_score(&self) -> f32 {
            if self.precision + self.recall == 0.0 {
                return 0.0;
            }
            2.0 * (self.precision * self.recall) / (self.precision + self.recall)
        }

        fn is_valid(&self) -> bool {
            self.accuracy >= 0.0 && self.accuracy <= 1.0
                && self.precision >= 0.0 && self.precision <= 1.0
                && self.recall >= 0.0 && self.recall <= 1.0
        }
    }

    let metrics = ModelMetrics {
        accuracy: 0.98,
        precision: 0.97,
        recall: 0.98,
    };

    assert!(metrics.is_valid());

    // F1 = 2 * (0.97 * 0.98) / (0.97 + 0.98) = 0.9749...
    let f1 = metrics.f1_score();
    assert!((f1 - 0.975).abs() < 0.001);

    // Invalid metrics
    let invalid = ModelMetrics {
        accuracy: 1.5,  // > 1.0
        precision: 0.9,
        recall: 0.9,
    };
    assert!(!invalid.is_valid());
}
```
