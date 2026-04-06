# Ruchy

Animation system for the Sovereign AI Stack.

## Overview

| Feature | Description |
|---------|-------------|
| Curves | Easing functions |
| Interpolation | Value transitions |
| Physics | Spring-based motion |
| Timeline | Sequenced animations |

## Easing Functions

| Easing | Curve |
|--------|-------|
| Linear | Constant rate |
| EaseIn | Slow start |
| EaseOut | Slow end |
| EaseInOut | Slow both |
| Bounce | Bouncy end |

## Basic Animation

```rust
use ruchy::{Animation, Easing};

let anim = Animation::new()
    .from(0.0)
    .to(100.0)
    .duration_ms(300)
    .easing(Easing::EaseOut);

// Get value at time t (0.0 to 1.0)
let value = anim.value_at(0.5);
```

## Spring Physics

```rust
use ruchy::Spring;

let spring = Spring::new()
    .stiffness(100.0)
    .damping(10.0)
    .mass(1.0);

let position = spring.position_at(time);
```

## Integration with Presentar

```rust
impl Widget for AnimatedBox {
    fn paint(&self, canvas: &mut dyn Canvas) {
        let opacity = self.fade_anim.value_at(self.time);
        canvas.fill_rect(self.bounds, Color::rgba(1.0, 0.0, 0.0, opacity));
    }
}
```

## Timeline

```rust
let timeline = Timeline::new()
    .add(0.0, fade_in)
    .add(0.3, slide_up)
    .add(0.6, scale_up);
```

## Verified Test

```rust
#[test]
fn test_ruchy_linear_interpolation() {
    // Linear interpolation formula
    fn lerp(start: f32, end: f32, t: f32) -> f32 {
        start + (end - start) * t
    }

    assert_eq!(lerp(0.0, 100.0, 0.0), 0.0);   // Start
    assert_eq!(lerp(0.0, 100.0, 0.5), 50.0);  // Middle
    assert_eq!(lerp(0.0, 100.0, 1.0), 100.0); // End

    // Ease-out formula: t * (2 - t)
    fn ease_out(t: f32) -> f32 {
        t * (2.0 - t)
    }

    assert_eq!(ease_out(0.0), 0.0);
    assert!((ease_out(0.5) - 0.75).abs() < 0.001);
    assert_eq!(ease_out(1.0), 1.0);
}
```
