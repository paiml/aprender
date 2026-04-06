# Integration Testing

Validating a mixed-language migration requires testing at multiple levels:
unit tests for individual functions, integration tests for module interactions,
and end-to-end tests that confirm the full system behaves identically to the
original.

## Cross-Component Test Strategy

The three testing levels map to different Cargo test targets:

```
tests/
  unit/           # cargo test --lib
    kernel.rs     # Individual convolution functions
    filters.rs    # Individual filter functions
    cli.rs        # Argument parsing
  integration/    # cargo test --test integration
    pipeline.rs   # Kernel + filters working together
    io.rs         # File loading + processing + saving
  e2e/            # cargo test --test e2e
    golden.rs     # Full CLI invocation, output comparison
```

Unit tests verify that each transpiled function matches its original behavior
in isolation. Integration tests verify that modules interact correctly through
shared types. End-to-end tests run the CLI binary and compare output files
byte-for-byte with reference outputs.

## End-to-End Validation

Batuta's `validate` command automates the comparison:

```bash
batuta validate ./image_toolkit_rs --reference ./image_toolkit
```

Under the hood, this:

1. Runs the original test suites (pytest, CUnit, shell) against the original
   code and captures outputs.
2. Runs the Rust test suite against the Rust code and captures outputs.
3. Compares outputs pairwise with configurable tolerance.
4. Reports any numerical divergence, missing outputs, or extra outputs.

For floating-point comparisons, the default tolerance is 1e-6 (relative). This
can be adjusted in `batuta.toml`:

```toml
[validation]
float_tolerance = 1e-6
comparison_mode = "relative"  # or "absolute", "ulp"
```

## Golden File Tests

Golden file tests capture known-good outputs and compare against them on every
run:

```rust
#[test]
fn test_gaussian_blur_golden() {
    let input = Image::load("tests/fixtures/input.png").unwrap();
    let output = gaussian_blur(&input, 2.0);

    let expected = Image::load("tests/fixtures/gaussian_blur_expected.png").unwrap();

    assert_images_equal(&output, &expected, 1e-6);
}
```

Golden files are generated once from the original Python implementation and
committed to the repository. They serve as the ground truth throughout the
migration.

## Regression Suites

To prevent regressions as components are migrated one at a time, Batuta
generates a regression suite that runs against every component boundary:

```rust
#[test]
fn regression_python_c_boundary() {
    // Verifies that the Rust kernel produces the same output
    // as the original C kernel for the Python test cases
    let test_cases = load_python_test_vectors("tests/fixtures/python_vectors.json");

    for case in test_cases {
        let result = convolve(&case.input, &case.kernel);
        assert_vec_approx_eq(&result.data, &case.expected, 1e-6);
    }
}
```

These boundary tests are particularly important during the gradual migration
period when some components are Rust and others are still in their original
language.

## Syscall Tracing for I/O Validation

For components that perform file or network I/O, Batuta uses `renacer` (the
syscall tracer) to verify that the Rust version makes equivalent system calls:

```bash
batuta validate ./image_toolkit_rs --reference ./image_toolkit --trace-syscalls
```

This catches subtle differences such as:

- Different file open flags (O_CREAT vs O_TRUNC)
- Missing fsync calls
- Changed buffer sizes in read/write calls
- Network connections to unexpected endpoints

## Test Coverage Tracking

Batuta tracks coverage across the migration to ensure no test gaps are
introduced:

```bash
make coverage
```

The coverage target should remain at or above the combined coverage of the
original test suites. Batuta reports coverage per module so that drops in a
specific area can be traced to the corresponding migration step.

## Continuous Integration

A typical CI pipeline for a mixed-language migration:

```yaml
test:
  steps:
    - cargo test --lib                     # Unit tests
    - cargo test --test integration        # Integration tests
    - cargo test --test e2e                # End-to-end tests
    - batuta validate . --reference ../ref # Cross-language comparison
    - make coverage                        # Coverage gate (>= 95%)
```

All five gates must pass before a migration PR is merged.

## Key Takeaways

- Test at three levels: unit (per-function), integration (cross-module), and
  end-to-end (full CLI with golden files).
- Golden files generated from the original implementation serve as ground truth
  throughout the migration.
- Boundary regression tests catch incompatibilities between migrated and
  unmigrated components.
- Syscall tracing validates I/O equivalence beyond just output correctness.
- Coverage tracking per module ensures that test quality does not regress as
  components are converted.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
