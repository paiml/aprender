# Module Boundaries

When a mixed-language project is transpiled, the original language boundaries
become natural Rust module boundaries. Batuta preserves the logical separation
while replacing cross-language interfaces with direct Rust calls.

## Language Boundaries Become Modules

In the image toolkit example, the three source directories map to three Rust
modules:

```
image_toolkit/            image_toolkit_rs/src/
  processing/ (Python) →    processing/mod.rs
  libkernel/  (C)      →    kernel/mod.rs
  scripts/    (Shell)   →    cli/mod.rs
```

Each module maintains its internal structure. Functions that were public in the
original language remain `pub` in Rust. Internal helpers become `pub(crate)` or
private.

## Shared Types Across Former Boundaries

Before migration, the Python code passed image data to C via a file path:

```python
# Python: write to temp file, call C library
import subprocess
np.save("/tmp/input.npy", image_array)
subprocess.run(["./libkernel", "convolve", "/tmp/input.npy", "/tmp/output.npy"])
result = np.load("/tmp/output.npy")
```

After migration, both modules share a common type:

```rust
// src/types.rs -- shared across all modules
pub struct Image {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}
```

```rust
// src/kernel/convolve.rs
pub fn convolve(image: &Image, kernel: &[f32]) -> Image {
    // Direct memory access, no file I/O
    // ...
}
```

```rust
// src/processing/filters.rs
use crate::kernel::convolve;
use crate::types::Image;

pub fn gaussian_blur(image: &Image, sigma: f32) -> Image {
    let kernel = build_gaussian_kernel(sigma);
    convolve(image, &kernel)
}
```

The file-based serialization layer is eliminated entirely. Data passes by
reference between modules with zero copy overhead.

## Unified Error Handling

Each original language had its own error style:

- **Python**: exceptions (`ValueError`, `FileNotFoundError`)
- **C**: integer return codes (`-1`, `ENOMEM`)
- **Shell**: exit codes (`1`, `2`)

After migration, all modules share a common error type:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ToolkitError {
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: usize, height: usize },

    #[error("Kernel size must be odd, got {size}")]
    InvalidKernelSize { size: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image format error: {0}")]
    Format(String),
}
```

Functions across all modules return `Result<T, ToolkitError>`, making error
propagation uniform. A filter function in the processing module can propagate
a kernel error from the kernel module without wrapping or re-throwing.

## Dependency Graph

Batuta generates a dependency graph showing how the unified modules relate:

```
cli (was: Shell scripts)
  └── processing (was: Python)
        └── kernel (was: C library)
              └── trueno (SIMD primitives)
```

The graph enforces that dependencies flow in one direction. Circular
dependencies between former language components are flagged during the unify
step and must be resolved before the build succeeds.

## Workspace Layout

For larger projects, Batuta can generate a Cargo workspace instead of a single
crate:

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["kernel", "processing", "cli"]
```

Each member is an independent crate with its own tests, but they share a common
`types` crate for cross-module data structures. This layout supports parallel
compilation and selective testing.

## Key Takeaways

- Language boundaries map directly to Rust module boundaries, preserving the
  original project's logical structure.
- Cross-language interfaces (files, subprocess, FFI) become direct function
  calls with shared types.
- A common error enum replaces the three different error conventions (Python
  exceptions, C return codes, Shell exit codes).
- Dependency direction is enforced by the module hierarchy: CLI depends on
  processing, which depends on kernel.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
