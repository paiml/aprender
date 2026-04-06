# Red-Green-Refactor

The fundamental TDD cycle.

## The Cycle

```
RED → GREEN → REFACTOR → repeat
```

## Phase 1: RED

Write a failing test FIRST:

```rust
#[test]
fn test_new_feature() {
    let widget = NewWidget::new();
    assert!(widget.works());  // FAILS: NewWidget doesn't exist
}
```

Run test - it MUST fail:
```bash
cargo test
# error: cannot find type `NewWidget`
```

## Phase 2: GREEN

Write MINIMAL code to pass:

```rust
pub struct NewWidget;

impl NewWidget {
    pub fn new() -> Self { Self }
    pub fn works(&self) -> bool { true }
}
```

Run test - it MUST pass:
```bash
cargo test
# test test_new_feature ... ok
```

## Phase 3: REFACTOR

Improve code quality:

```rust
pub struct NewWidget {
    enabled: bool,
}

impl NewWidget {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    pub fn works(&self) -> bool {
        self.enabled
    }
}
```

Run test - STILL passes:
```bash
cargo test
# test test_new_feature ... ok
```

## Rules

| Rule | Description |
|------|-------------|
| Test first | Never write production code without a failing test |
| Minimal green | Only enough code to pass |
| Refactor often | Clean up while tests are green |
| Small steps | One behavior at a time |

## Anti-Patterns

```rust
// BAD: Writing implementation before test
impl Widget for Foo {
    fn measure(&self, c: Constraints) -> Size {
        // Complex logic without test coverage
    }
}

// GOOD: Test drives implementation
#[test]
fn test_foo_measure_respects_constraints() {
    let foo = Foo::new();
    let size = foo.measure(Constraints::tight(Size::new(100.0, 50.0)));
    assert_eq!(size, Size::new(100.0, 50.0));
}
```

## Verified Test

```rust
#[test]
fn test_red_green_refactor_example() {
    // This test demonstrates the pattern
    struct Counter { value: i32 }
    impl Counter {
        fn new() -> Self { Self { value: 0 } }
        fn increment(&mut self) { self.value += 1; }
        fn get(&self) -> i32 { self.value }
    }

    let mut c = Counter::new();
    assert_eq!(c.get(), 0);
    c.increment();
    assert_eq!(c.get(), 1);
}
```
