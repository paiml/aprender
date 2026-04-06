# Ownership Inference

Decy analyzes C code to infer Rust ownership semantics from pointer usage
patterns. This is the core challenge of C-to-Rust transpilation: C has one
pointer type (`T*`), while Rust distinguishes between owned values, shared
references, mutable references, and raw pointers.

## Inference Rules

Decy applies the following heuristics to classify each pointer parameter:

| C Pattern                          | Inferred Rust Type  | Rationale                    |
|------------------------------------|---------------------|------------------------------|
| `const T*` read-only param         | `&T` or `&[T]`     | No mutation, no ownership    |
| `T*` modified but not freed        | `&mut T`            | Mutation without ownership   |
| `T*` returned from malloc          | `Box<T>` or `Vec<T>`| Caller owns the allocation  |
| `T*` passed to free                | Owned (consumed)    | Transfer of ownership        |
| `T**` output parameter             | `&mut Option<T>`    | Caller receives ownership    |

## Shared References

**C**

```c
double vector_sum(const double* data, size_t len) {
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}
```

**Rust**

```rust
fn vector_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}
```

The `const` qualifier on `data` combined with no `free` call tells decy that
this is a borrowed, read-only reference. The separate `len` parameter merges
into the slice type.

## Mutable References

**C**

```c
void normalize(double* data, size_t len) {
    double max = 0.0;
    for (size_t i = 0; i < len; i++) {
        if (data[i] > max) max = data[i];
    }
    for (size_t i = 0; i < len; i++) {
        data[i] /= max;
    }
}
```

**Rust**

```rust
fn normalize(data: &mut [f64]) {
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for x in data.iter_mut() {
        *x /= max;
    }
}
```

The pointer is modified in place but not freed, so decy infers `&mut [f64]`.

## Owned Values

**C**

```c
double* linspace(double start, double end, size_t n) {
    double* result = malloc(n * sizeof(double));
    double step = (end - start) / (double)(n - 1);
    for (size_t i = 0; i < n; i++) {
        result[i] = start + step * (double)i;
    }
    return result;  // Caller must free
}
```

**Rust**

```rust
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}
```

The `malloc` followed by return tells decy the caller takes ownership. The
natural Rust equivalent is `Vec<f64>`.

## Lifetime Annotations

When decy detects that a returned pointer aliases an input, it generates
lifetime annotations:

**C**

```c
// Returns pointer into data -- NOT a new allocation
const double* find_max(const double* data, size_t len) {
    const double* max = &data[0];
    for (size_t i = 1; i < len; i++) {
        if (data[i] > *max) max = &data[i];
    }
    return max;
}
```

**Rust**

```rust
fn find_max(data: &[f64]) -> &f64 {
    data.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}
```

Decy recognizes that the returned pointer points into `data` rather than a new
allocation. The Rust borrow checker enforces that the returned reference cannot
outlive `data`.

## Ambiguous Cases

When decy cannot determine ownership from usage patterns alone, it falls back
to conservative choices and emits a warning:

```
WARN: Cannot infer ownership for `ctx` in process_data(Context* ctx).
      Defaulting to &mut Context. Review and adjust if needed.
```

These warnings are surfaced in the Batuta validation report, allowing developers
to review and correct the small number of cases that require manual judgment.

## Key Takeaways

- Decy classifies C pointers into owned, shared, and mutable categories based
  on usage patterns (const, malloc, free, modification).
- Separate length parameters merge into Rust slices automatically.
- Returned pointers that alias inputs receive lifetime annotations.
- Ambiguous cases produce warnings rather than silent incorrect translations.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
