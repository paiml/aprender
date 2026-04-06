# Type Inference Problems

Dynamic typing in Python and implicit typing in C create challenges when transpiling to Rust's strict static type system.

## Common Inference Failures

### 1. Ambiguous Numeric Types

Python has one `int` (arbitrary precision) and one `float` (f64). Rust has twelve numeric types.

| Python Type | Default Rust Mapping | When It Breaks |
|-------------|---------------------|----------------|
| `int` | `i64` | Values > i64::MAX, or used as index (`usize`) |
| `float` | `f64` | ML code expecting `f32` for performance |
| `bool` | `bool` | Used in arithmetic (`True + 1`) |

**Fix**: Add type hints to the Python source before transpiling:

```python
def compute(data: list[float], scale: float) -> list[float]:
    return [x * scale for x in data]
```

### 2. Collection Type Mismatch

Python lists are heterogeneous. Rust collections are homogeneous:

```python
# Cannot transpile: mixed types
items = [1, "two", 3.0]

# Transpiles cleanly: uniform type
items: list[int] = [1, 2, 3]
```

### 3. Optional/None Handling

Python uses `None` freely. Rust requires explicit `Option<T>`:

```rust
// Transpiler infers Option<T> from None returns
fn find(items: &[Item], key: &str) -> Option<&Item> {
    items.iter().find(|item| item.key == key)
}
```

### 4. Dict Key/Value Types

Ambiguous dict types need TypedDict or explicit annotations:

```python
from typing import TypedDict

class Config(TypedDict):
    name: str
    layers: int
    dropout: float
```

## Annotation Strategies

When transpilation fails due to type ambiguity, use these strategies in order:

1. **Add Python type hints** to the source (preferred)
2. **Use `batuta.toml` type overrides** for code you cannot modify
3. **Post-process the Rust output** to fix remaining errors

```toml
# batuta.toml type overrides
[type_overrides]
"module.function.param_x" = "f32"
"module.function.return" = "Vec<f32>"
```

## Diagnostic Output

When type inference fails, batuta reports the location and ambiguity:

```
Warning: Ambiguous type at src/model.py:42
  Variable 'weights' used as both list[float] and ndarray
  Inferred: Vec<f64> (may need manual review)
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
