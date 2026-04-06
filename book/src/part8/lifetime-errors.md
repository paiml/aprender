# Lifetime Errors

Lifetime errors are the most common Rust-specific challenge when migrating from C. They arise because Rust enforces at compile time what C leaves to programmer discipline: every reference must be valid for its entire usage.

## Ownership Patterns

| Pattern | Rust Syntax | C Equivalent | Use When |
|---------|-------------|--------------|----------|
| Owned | `String`, `Vec<T>` | `malloc` + `free` | Data has a single clear owner |
| Borrowed | `&T`, `&mut T` | `const T*`, `T*` | Temporary read/write access |
| Shared | `Rc<T>`, `Arc<T>` | Reference counting | Multiple owners |

## Common C Patterns and Rust Solutions

### Returning a Pointer to Stack Data

```c
// C: undefined behavior
char* get_name() {
    char buf[64];
    sprintf(buf, "model_%d", id);
    return buf;  // BUG: pointer to expired stack frame
}
```

```rust
// Rust: return an owned String
fn get_name(id: u32) -> String {
    format!("model_{}", id)
}
```

### Mutable Aliasing

```c
// C: two pointers to the same data
void swap_first_last(int* arr, int len) {
    int tmp = arr[0]; arr[0] = arr[len-1]; arr[len-1] = tmp;
}
```

```rust
// Rust: use slice methods that handle aliasing safely
fn swap_first_last(arr: &mut [i32]) {
    let len = arr.len();
    arr.swap(0, len - 1);
}
```

## Common Lifetime Fixes

### Function That Borrows and Returns

```rust
// Error: missing lifetime specifier
fn longest(a: &str, b: &str) -> &str { ... }

// Fix: output lifetime tied to inputs
fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() { a } else { b }
}
```

## When to Use Owned Types Instead

If lifetime annotations become deeply nested, consider owning the data:

| Complexity | Approach |
|-----------|----------|
| Simple (1 lifetime) | Use `&'a T` |
| Moderate (2-3 lifetimes) | Use `&'a T` with clear naming |
| Complex (nested lifetimes) | Use `String`, `Vec<T>`, or `Arc<T>` |

## Diagnostic Tips

The Rust compiler's borrow checker errors include helpful suggestions. Look for:

- "consider borrowing here" -- add `&`
- "consider using a `let` binding" -- extend the lifetime
- "lifetime may not live long enough" -- add or adjust annotations

---

**Navigate:** [Table of Contents](../SUMMARY.md)
