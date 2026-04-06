# Test Migration

Migrating tests from Python pytest to Rust `#[test]` is as important as migrating the code itself. This chapter maps common pytest patterns to their Rust equivalents.

## Pytest to Rust Mapping

| pytest Pattern | Rust Equivalent |
|---------------|-----------------|
| `def test_foo():` | `#[test] fn test_foo()` |
| `assert x == y` | `assert_eq!(x, y)` |
| `with pytest.raises(ValueError):` | `#[should_panic]` or `assert!(result.is_err())` |
| `@pytest.fixture` | Helper function or `LazyLock` |
| `@pytest.mark.parametrize` | `test-case` crate or `proptest!` |
| `conftest.py` | `mod test_helpers` |
| `tmpdir` fixture | `tempfile::TempDir` |

## Fixture Patterns

```python
# Python
@pytest.fixture
def sample_model():
    return Model(layers=4, hidden=256)
```

```rust
// Rust: helper function
fn sample_model() -> Model {
    Model::new(4, 256)
}

// Rust: lazy static for expensive setup
use std::sync::LazyLock;
static SAMPLE_MODEL: LazyLock<Model> = LazyLock::new(|| Model::new(4, 256));
```

## Parameterized Tests

```rust
use test_case::test_case;

#[test_case(1, 2 ; "one")]
#[test_case(3, 6 ; "three")]
#[test_case(5, 10 ; "five")]
fn test_double(input: i32, expected: i32) {
    assert_eq!(double(input), expected);
}
```

## Error Testing

```rust
#[test]
fn test_invalid_input() {
    let result = compute(-1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("negative"));
}
```

## Temporary Files

```rust
#[test]
fn test_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.bin");
    save(&model, &path).unwrap();
    let loaded = load(&path).unwrap();
    // dir cleaned up on drop
}
```

## Migration Checklist

1. Inventory all pytest files and count test functions
2. Map fixtures to Rust helpers (create `test_helpers.rs`)
3. Convert assertions one file at a time
4. Run both test suites during migration to catch gaps
5. Remove Python tests only after Rust coverage meets 95%

---

**Navigate:** [Table of Contents](../SUMMARY.md)
