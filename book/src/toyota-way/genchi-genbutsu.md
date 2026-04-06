# Genchi Genbutsu

"Go and see" - verify with real data.

## Principle

| Japanese | English | Application |
|----------|---------|-------------|
| Genchi | Actual place | Test in real browser |
| Genbutsu | Actual thing | Use real data |

## Applied to Testing

```rust
// Don't just mock - verify with real rendering
#[test]
fn test_with_real_canvas() {
    let canvas = RecordingCanvas::new();
    widget.paint(&mut canvas);

    // Verify actual draw commands
    assert_eq!(canvas.command_count(), 3);
}
```

## Real Browser Testing

| Level | Tool | Purpose |
|-------|------|---------|
| Unit | Rust tests | Logic verification |
| Integration | WASM tests | Browser compatibility |
| E2E | Real browser | User experience |

## Data Verification

```rust
// Test with actual dataset structure
#[test]
fn test_chart_with_real_data() {
    let data = load_test_fixture("sales.ald");
    let chart = Chart::new(data);

    // Verify real data renders correctly
    assert!(chart.points().len() > 0);
}
```

## Visual Verification

```rust
// Capture and compare actual pixels
#[test]
fn test_visual_output() {
    let canvas = RecordingCanvas::new();
    widget.paint(&mut canvas);

    let snapshot = Snapshot::from_canvas(&canvas);
    snapshot.assert_match("expected_output.png");
}
```

## Debugging Flow

```
Bug reported → Reproduce locally → Inspect actual output → Fix → Verify
```

## Verified Test

```rust
#[test]
fn test_genchi_genbutsu_real_canvas() {
    use presentar_core::RecordingCanvas;

    // "Go and see" - use real canvas, not mocks
    let mut canvas = RecordingCanvas::new();

    // Perform actual operations
    canvas.fill_rect(
        presentar_core::Rect::new(0.0, 0.0, 100.0, 50.0),
        presentar_core::Color::RED,
    );

    // Verify actual state
    assert_eq!(canvas.command_count(), 1);

    // Can inspect the actual command
    let commands = canvas.commands();
    assert!(!commands.is_empty());
}
```
