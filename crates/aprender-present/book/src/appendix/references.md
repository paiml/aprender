# References

Academic and industry sources.

## Layout Algorithms

| Source | Topic |
|--------|-------|
| CSS Flexbox Spec | Flex layout algorithm |
| Yoga Layout | Cross-platform flexbox |
| Flutter Layout | Constraint-based layout |

## Accessibility

| Standard | Description |
|----------|-------------|
| WCAG 2.1 | Web Content Accessibility Guidelines |
| WAI-ARIA | Accessible Rich Internet Apps |
| Section 508 | US federal accessibility |

## Testing

| Paper/Tool | Contribution |
|------------|--------------|
| Mutation Testing | Fault injection for test quality |
| Property-Based Testing | QuickCheck-style generation |
| Visual Regression | Pixel-diff comparison |

## GPU Rendering

| Technology | Use |
|------------|-----|
| WebGPU | Cross-platform GPU API |
| WGSL | WebGPU Shading Language |
| wgpu-rs | Rust WebGPU implementation |

## Rust Ecosystem

| Crate | Purpose |
|-------|---------|
| trueno | SIMD tensor operations |
| winit | Window management |
| fontdue | Font rasterization |

## Key Algorithms

```rust
// Flexbox main axis distribution
fn distribute_space(items: &[f32], available: f32) -> Vec<f32> {
    let total: f32 = items.iter().sum();
    let scale = if total > 0.0 { available / total } else { 0.0 };
    items.iter().map(|&flex| flex * scale).collect()
}
```

## Verified Test

```rust
#[test]
fn test_references_flex_distribution() {
    // Flexbox space distribution algorithm
    let items = vec![1.0, 2.0, 1.0];
    let available = 400.0;

    let total: f32 = items.iter().sum();
    let scale = available / total;
    let result: Vec<f32> = items.iter().map(|&f| f * scale).collect();

    assert_eq!(result[0], 100.0);  // 1/4
    assert_eq!(result[1], 200.0);  // 2/4
    assert_eq!(result[2], 100.0);  // 1/4
}
```
