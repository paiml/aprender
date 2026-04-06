# Gradual Migration

A full rewrite is risky. Batuta supports incremental migration where one
component is converted at a time while the rest of the system continues
running in its original language. FFI bridges and feature flags manage the
transition.

## Incremental Approach

The image toolkit migration proceeds in three releases:

```
Release 1: Shell → Rust CLI
  - Original Python and C code unchanged
  - Rust CLI calls Python/C via subprocess (same as before)

Release 2: C library → Rust crate
  - Python code calls Rust via FFI (cdylib) instead of C
  - Rust CLI now calls Rust kernel directly

Release 3: Python → Rust
  - All components are Rust
  - FFI bridges removed
  - Single static binary
```

Each release is independently testable and deployable. If Release 2 introduces
a regression, the team can revert to the C library without affecting the CLI.

## FFI Bridges During Transition

During Release 2, the Python code still needs to call the kernel. Decy generates
a C-compatible shared library from the Rust code:

```rust
// src/kernel/ffi.rs -- temporary bridge for Python
#[no_mangle]
pub extern "C" fn kernel_convolve(
    input: *const f32,
    width: u32,
    height: u32,
    kernel: *const f32,
    kernel_size: u32,
    output: *mut f32,
) -> i32 {
    let input = unsafe {
        std::slice::from_raw_parts(input, (width * height) as usize)
    };
    let kernel = unsafe {
        std::slice::from_raw_parts(kernel, (kernel_size * kernel_size) as usize)
    };
    let output = unsafe {
        std::slice::from_raw_parts_mut(output, (width * height) as usize)
    };

    match crate::kernel::convolve_into(input, width as usize, height as usize,
                                        kernel, output) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}
```

The Python code switches from loading `libkernel.so` (C) to `libkernel_rs.so`
(Rust) with no changes to the Python source:

```python
# Python: same ctypes interface, different .so file
import ctypes
lib = ctypes.CDLL("./libkernel_rs.so")  # Was: libkernel.so
```

## Feature Flags for Old/New Implementations

During the transition, both implementations can coexist behind feature flags:

```toml
# Cargo.toml
[features]
default = ["rust-kernel"]
rust-kernel = []         # New Rust implementation
c-kernel = []            # Original C via FFI
```

```rust
#[cfg(feature = "rust-kernel")]
pub fn convolve(image: &Image, kernel: &[f32]) -> Image {
    // Pure Rust implementation
    rust_convolve(image, kernel)
}

#[cfg(feature = "c-kernel")]
pub fn convolve(image: &Image, kernel: &[f32]) -> Image {
    // FFI call to original C library
    unsafe { c_convolve(image, kernel) }
}
```

This allows A/B testing between the old and new implementations in production.
Benchmarks run both paths to verify performance parity before the C code is
removed.

## Migration Checklist Per Component

For each component being migrated:

1. **Transpile**: Run the appropriate transpiler (depyler, decy, bashrs).
2. **Bridge**: Generate FFI bridge if other components still depend on it.
3. **Test**: Run the component's original test suite against the Rust version.
4. **Benchmark**: Compare latency and throughput against the original.
5. **Deploy**: Release the Rust component behind a feature flag.
6. **Validate**: Monitor production metrics for one release cycle.
7. **Remove**: Delete the FFI bridge and original source code.

## Rollback Strategy

Each step is reversible:

- **Feature flags** let you switch back to the C implementation in a config
  change without redeployment.
- **Shared library ABI** compatibility means Python consumers can revert to
  the original `.so` by changing a single path.
- **Git tags** mark each release boundary for clean rollback if needed.

## Key Takeaways

- Migrate one component at a time, from leaves to roots in the dependency graph.
- FFI bridges maintain compatibility with unconverted components during the
  transition period.
- Feature flags allow both old and new implementations to coexist for A/B
  testing and safe rollback.
- Each migration step is independently testable, deployable, and reversible.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
