# MNIST Explorer

Interactive digit recognition demo.

## Overview

| Feature | Description |
|---------|-------------|
| Drawing | Canvas for digit input |
| Inference | Real-time prediction |
| Visualization | Confidence bars |
| Dataset | Browse MNIST images |

## YAML Configuration

```yaml
app:
  name: "MNIST Explorer"

data:
  model:
    source: "mnist_classifier.apr"

  dataset:
    source: "mnist.ald"
    limit: 1000

widgets:
  root:
    type: Row
    children:
      - type: Column
        children:
          - type: Text
            value: "Draw a digit:"
          - type: Canvas
            id: "draw_canvas"
            width: 280
            height: 280
            on_draw: "predict"
          - type: Button
            label: "Clear"
            on_click: "clear_canvas"

      - type: Column
        children:
          - type: Text
            value: "Prediction:"
          - type: Text
            id: "prediction"
            value: "{{ prediction.digit }}"
            font_size: 48
          - type: ProgressBar
            label: "Confidence"
            value: "{{ prediction.confidence }}"
```

## Model Card

```yaml
model_card:
  name: "MNIST Classifier"
  version: "1.0.0"
  task: "Image Classification"
  input: "28x28 grayscale image"
  output: "10-class probability"
  accuracy: 0.98
  limitations:
    - "Grayscale only"
    - "Centered digits perform best"
```

## Inference Pipeline

```rust
// 1. Capture canvas pixels
let pixels = canvas.get_pixels();

// 2. Resize to 28x28
let resized = resize(pixels, 28, 28);

// 3. Normalize to 0-1
let normalized: Vec<f32> = resized.iter()
    .map(|&p| p as f32 / 255.0)
    .collect();

// 4. Run inference
let prediction = model.predict(&normalized);
```

## Verified Test

```rust
#[test]
fn test_mnist_normalization() {
    // Pixel normalization for MNIST
    let raw_pixels: Vec<u8> = vec![0, 128, 255];
    let normalized: Vec<f32> = raw_pixels.iter()
        .map(|&p| p as f32 / 255.0)
        .collect();

    assert_eq!(normalized[0], 0.0);
    assert!((normalized[1] - 0.502).abs() < 0.01);
    assert_eq!(normalized[2], 1.0);

    // All values in valid range
    for &v in &normalized {
        assert!(v >= 0.0 && v <= 1.0);
    }
}
```
