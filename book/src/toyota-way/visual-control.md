# Visual Control

Make the work visible to spot problems immediately.

## Principle

| Concept | Application |
|---------|-------------|
| Andon | Status indicators |
| Kanban | Progress boards |
| Dashboard | Metrics display |

## Test Status

```
✓ 1188 tests passing
✗ 0 tests failing
⚠ 0 tests skipped
```

## Quality Dashboard

```
Quality Score: 90/100 (A)
├─ Test Coverage: 97%    ✓
├─ Mutation Score: 85%   ✓
├─ Frame Time: 12ms      ✓
└─ Bundle Size: 380KB    ✓
```

## Build Feedback

```rust
// Visual indicators in output
fn print_status(result: TestResult) {
    match result {
        TestResult::Pass => println!("✓ PASS"),
        TestResult::Fail => println!("✗ FAIL"),
        TestResult::Skip => println!("○ SKIP"),
    }
}
```

## Progress Visualization

```
Building... [████████░░] 80%
Testing...  [██████████] 100%
Linting...  [██████░░░░] 60%
```

## Error Highlighting

```rust
// Clear visual distinction for errors
error[E0308]: mismatched types
  --> src/widget.rs:42:5
   |
42 |     size.width
   |     ^^^^^^^^^^ expected Size, found f32
```

## Metrics Over Time

| Week | Tests | Coverage | Frame ms |
|------|-------|----------|----------|
| 1 | 800 | 85% | 18ms |
| 2 | 950 | 90% | 15ms |
| 3 | 1100 | 95% | 13ms |
| 4 | 1188 | 97% | 12ms |

## Verified Test

```rust
#[test]
fn test_visual_control_status() {
    // Status enum for visual indicators
    #[derive(Debug, PartialEq)]
    enum Status {
        Pass,
        Fail,
        Warning,
    }

    impl Status {
        fn symbol(&self) -> &'static str {
            match self {
                Status::Pass => "✓",
                Status::Fail => "✗",
                Status::Warning => "⚠",
            }
        }
    }

    assert_eq!(Status::Pass.symbol(), "✓");
    assert_eq!(Status::Fail.symbol(), "✗");
    assert_eq!(Status::Warning.symbol(), "⚠");

    // Quick visual scan possible
    let results = vec![Status::Pass, Status::Pass, Status::Fail];
    let has_failure = results.iter().any(|s| *s == Status::Fail);
    assert!(has_failure);
}
```
