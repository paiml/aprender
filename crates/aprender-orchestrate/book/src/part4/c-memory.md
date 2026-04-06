# Memory Management: C to Rust

The most impactful transformation in C-to-Rust transpilation is replacing manual
memory management with Rust's ownership system. Decy performs this conversion
automatically for common allocation patterns.

## malloc/free to Ownership

**C**

```c
double* create_vector(size_t n) {
    double* v = (double*)malloc(n * sizeof(double));
    if (!v) return NULL;
    memset(v, 0, n * sizeof(double));
    return v;
}

void destroy_vector(double* v) {
    free(v);
}
```

**Rust**

```rust
fn create_vector(n: usize) -> Vec<f64> {
    vec![0.0; n]
}
// No destroy_vector needed -- Vec drops automatically
```

The `malloc`/`memset`/`free` triple collapses into a single `vec!` macro call.
The destructor is implicit: `Vec` deallocates when it goes out of scope.

## Pointer Arithmetic to Slices

**C**

```c
double dot_product(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

**Rust**

```rust
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

Raw pointers with a separate length parameter become slices (`&[f64]`), which
carry their length and enforce bounds checking. The iterator chain replaces the
index-based loop, eliminating off-by-one errors.

## Buffer Overflow Elimination

**C (vulnerable)**

```c
void copy_data(double* dst, const double* src, size_t n) {
    // No bounds check -- caller must ensure dst has capacity
    memcpy(dst, src, n * sizeof(double));
}
```

**Rust (safe)**

```rust
fn copy_data(dst: &mut [f64], src: &[f64]) {
    // Panics at runtime if src.len() > dst.len()
    dst[..src.len()].copy_from_slice(src);
}
```

The Rust version validates the destination capacity at runtime. In release
builds with `--release`, bounds checks on slice access are optimized away when
the compiler can prove safety statically.

## Realloc to Vec::resize

**C**

```c
double* grow_buffer(double* buf, size_t old_n, size_t new_n) {
    double* new_buf = (double*)realloc(buf, new_n * sizeof(double));
    if (!new_buf) { free(buf); return NULL; }
    memset(new_buf + old_n, 0, (new_n - old_n) * sizeof(double));
    return new_buf;
}
```

**Rust**

```rust
fn grow_buffer(buf: &mut Vec<f64>, new_n: usize) {
    buf.resize(new_n, 0.0);
}
```

`Vec::resize` handles reallocation, copying, and zero-initialization in a
single call. There is no possibility of use-after-free because the old
allocation is managed internally.

## Struct with Owned Data

**C**

```c
typedef struct {
    double* data;
    size_t rows;
    size_t cols;
} Matrix;

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix* m = malloc(sizeof(Matrix));
    m->data = calloc(rows * cols, sizeof(double));
    m->rows = rows;
    m->cols = cols;
    return m;
}

void matrix_free(Matrix* m) {
    free(m->data);
    free(m);
}
```

**Rust**

```rust
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
}
// Drop is automatic -- no matrix_free needed
```

## Key Takeaways

- `malloc`/`free` pairs become `Vec<T>` with automatic deallocation.
- Raw pointer parameters with length become slices (`&[T]` or `&mut [T]`).
- Buffer overflows are caught at compile time or with runtime bounds checks.
- `realloc` patterns simplify to `Vec::resize`.
- Struct destructors (`free` chains) are replaced by Rust's automatic `Drop`.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
