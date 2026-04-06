# Decy: C/C++ to Rust

Decy transpiles C and C++ source code into safe, idiomatic Rust. Its core challenge is inferring Rust ownership semantics from C pointer patterns and replacing manual memory management with RAII.

## Overview

| Attribute | Value |
|-----------|-------|
| **Direction** | C/C++ to Rust |
| **Install** | `cargo install decy` |
| **Input** | `.c`, `.cpp`, `.h`, `.hpp` files |
| **Output** | Safe Rust with ownership and lifetime annotations |

## Ownership Inference from Pointer Analysis

C uses raw pointers for everything: ownership, borrowing, output parameters, and arrays. Decy analyzes pointer usage patterns to infer the correct Rust ownership model.

| C Pattern | Decy Inference | Rust Output |
|-----------|----------------|-------------|
| `const T*` read only | Shared reference | `&T` |
| `T*` written through | Mutable reference | `&mut T` |
| `T*` from `malloc`, returned | Owned value | `Box<T>` or `T` |
| `T*` freed in same scope | Scoped owner | `let val: T` (stack) |
| `T**` output parameter | Return value | `-> T` |
| `T*` array + length | Slice | `&[T]` or `&mut [T]` |

## Memory Management Translation

Decy replaces `malloc`/`free` pairs with Rust RAII, eliminating use-after-free and double-free at compile time.

```c
Buffer* buf = (Buffer*)malloc(sizeof(Buffer));
buf->data = (char*)malloc(size);
free(buf->data);
free(buf);
```

```rust
// RAII: dropped automatically when buf goes out of scope
let buf = Buffer { data: vec![0u8; size], len: size };
```

Common translations: `char*` + `strlen()` becomes `String`, `strdup(s)` becomes `s.to_string()`, `strcmp(a,b)==0` becomes `a == b`, and `snprintf` becomes `format!(...)`.

## FFI Boundary Generation

For gradual migration, Decy generates `extern "C"` wrappers so existing C code can call the new Rust functions. This allows teams to migrate one file at a time, linking Rust objects into the existing C build system.

```rust
#[no_mangle]
pub extern "C" fn process_buffer(data: *const u8, len: usize) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    process_buffer_safe(slice).unwrap_or(-1)
}
```

Pass `--ffi` to `decy transpile` to generate these wrappers alongside the safe Rust implementation.

## Common C Patterns and Rust Equivalents

| C Pattern | Rust Equivalent |
|-----------|-----------------|
| `for (int i = 0; i < n; i++)` | `for i in 0..n` |
| `switch / case` | `match` |
| `typedef struct` | `struct` |
| `union` | `enum` with variants |
| `goto cleanup` | `?` operator or `Drop` trait |
| `#define MAX(a,b)` | `std::cmp::max(a, b)` |
| `NULL` check | `Option<T>` |
| `errno` codes | `Result<T, E>` |

## CLI Usage

```bash
# Transpile a single C file
decy transpile --input parser.c --output parser.rs

# Transpile with FFI wrappers for gradual migration
decy transpile --input lib.c --output lib.rs --ffi

# Transpile a C project directory
decy transpile --input ./c_project --output ./rust_project

# Via Batuta orchestration
batuta transpile --input ./c_project --output ./rust_project
```

## Limitations

- **Inline assembly**: Not transpiled; must be replaced manually or wrapped in `unsafe`
- **Complex macros**: Preprocessor macros with side effects require manual review
- **Void pointers**: `void*` used as generic storage needs manual type annotation
- **Bit fields**: Struct bit fields are converted to explicit mask operations

---

**Navigate:** [Table of Contents](../SUMMARY.md)
