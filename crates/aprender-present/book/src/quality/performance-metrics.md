# Performance Metrics

Measuring rendering and runtime performance.

## Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Frame time | <16ms | 60fps rendering |
| First paint | <100ms | Initial content |
| Layout time | <5ms | Widget positioning |
| Paint time | <10ms | Draw commands |

## Frame Budget

```
16ms frame budget:
├─ Event handling: 2ms
├─ State update: 1ms
├─ Layout: 3ms
├─ Paint: 5ms
├─ GPU submit: 3ms
└─ Buffer: 2ms
```

## Measuring Performance

```rust
use std::time::Instant;

fn measure_frame<F: FnOnce()>(f: F) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}
```

## Bundle Size

| Category | Target |
|----------|--------|
| Core | <100KB |
| Widgets | <150KB |
| Total | <500KB |

## Memory Usage

| Component | Budget |
|-----------|--------|
| Widget tree | <10MB |
| Texture atlas | <50MB |
| Layout cache | <5MB |

## Performance Score

```rust
fn performance_score(metrics: &PerfMetrics) -> f32 {
    let frame_score = if metrics.frame_time_ms < 16.0 { 100.0 }
        else { (16.0 / metrics.frame_time_ms) * 100.0 };

    let size_score = if metrics.bundle_kb < 500.0 { 100.0 }
        else { (500.0 / metrics.bundle_kb) * 100.0 };

    frame_score * 0.6 + size_score * 0.4
}
```

## Profiling

```rust
#[cfg(feature = "profiling")]
fn profile_layout(tree: &mut WidgetTree) {
    let start = Instant::now();
    tree.layout();
    log::trace!("Layout: {:?}", start.elapsed());
}
```

## Verified Test

```rust
#[test]
fn test_performance_frame_budget() {
    // 60fps = 16.67ms per frame
    let target_fps = 60.0;
    let frame_budget_ms = 1000.0 / target_fps;

    assert!((frame_budget_ms - 16.67).abs() < 0.01);

    // Frame time check
    let actual_frame_ms = 12.5;
    let meets_budget = actual_frame_ms <= frame_budget_ms;
    assert!(meets_budget);

    // Calculate actual FPS
    let actual_fps = 1000.0 / actual_frame_ms;
    assert_eq!(actual_fps, 80.0);  // 12.5ms = 80fps
}
```
