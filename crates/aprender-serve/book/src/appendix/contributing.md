# Contributing to This Book

Thank you for your interest in improving the Realizar documentation! This book follows the same EXTREME TDD principles that guide the project itself.

## Core Principles

> **"Every code example must be test-backed. Zero tolerance for hallucinated examples."**

All book content follows these principles:

1. **Test-Backed Code** - Every Rust code example must be validated by actual tests
2. **Production-Proven** - Examples come from the real codebase
3. **Reproducible** - Readers can run the same tests and see the same results
4. **Anti-Hallucination** - If it's not tested, it doesn't go in the book

## Code Example Requirements

### ✅ Valid Code Examples

**1. Code from Actual Source Files**

```rust
// From src/tensor.rs:58-64 (test-backed)
pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Tensor<T>> {
    let size: usize = shape.iter().product();
    if data.len() != size {
        return Err(RealizarError::InvalidShape { /* ... */ });
    }
    Ok(Tensor { shape, data })
}
```

Reference the file and line numbers in comments when possible.

**2. Complete Tested Examples**

```rust
// From examples/inference.rs (fully tested example)
use realizar::{Tensor, Transformer, ModelConfig};

fn main() -> Result<()> {
    let config = ModelConfig::default();
    let model = Transformer::new(config);
    // ...
}
```

**3. Property-Based Test Examples**

```rust
// From tests/property_tensor.rs (demonstrates TDD principles)
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_tensor_shape_matches(
        shape in prop::collection::vec(1usize..10, 1..4)
    ) {
        // Property: shape always matches input
    }
}
```

### ❌ Invalid Code Examples

**Do NOT include:**

- Hypothetical code that doesn't exist in the codebase
- Pseudocode without corresponding real implementation
- "Future API" examples unless clearly marked and in roadmap
- Code snippets without test coverage
- Examples that haven't been compiled and tested

## Validation Process

All code examples are validated automatically:

```bash
# Validate book code is test-backed
make book-validate

# Full quality gates (includes book validation)
make quality-gates
```

The validation script (`scripts/validate-book-code.sh`) checks:

1. **Code Block Extraction** - Finds all \`\`\`rust blocks
2. **Identifier Matching** - Searches for functions/structs in actual code
3. **Test Coverage** - Warns about potentially untested code

## Writing Guidelines

### Chapter Structure

Each chapter should follow this structure:

1. **Concept Introduction** - What are we building?
2. **TDD Approach** - How we test-drove the implementation
3. **RED Phase** - Show the failing test(s)
4. **GREEN Phase** - Show the minimal implementation
5. **REFACTOR Phase** - Show the final production code
6. **Results** - Show test output, benchmarks, metrics

### Code Comments

Always include:

```rust
// From: src/quantize.rs:45-67
// Test: tests/unit_quantize.rs:123-145
// Coverage: 95% (line 52 edge case in property test)
pub fn q4_0(weights: &[f32]) -> (Vec<u8>, Vec<f32>) {
    // Implementation
}
```

### Linking to Tests

When showing production code, link to the tests:

> This implementation is validated by:
> - Unit tests: `tests/unit_quantize.rs:123-145`
> - Property tests: `tests/property_quantize.rs:89-112`
> - Mutation score: 100% (18/18 mutants killed)

## Adding New Chapters

1. **Create the chapter file** in `book/src/`
2. **Add to SUMMARY.md** in the appropriate section
3. **Write content following TDD structure**
4. **Reference actual tested code only**
5. **Validate**: `make book-validate`
6. **Build**: `make book-build`
7. **Submit PR** with tested code examples

## Common Pitfalls

### ❌ Pitfall: Illustrative Examples

```rust
// DON'T: Hypothetical example that doesn't exist
pub fn load_model(path: &str) -> Model {
    // This looks reasonable but isn't in our codebase
}
```

### ✅ Solution: Use Actual Code or Mark as Roadmap

```rust
// ✓ DO: Reference actual code
// From: examples/inference.rs:45-52
pub fn create_demo_model() -> Transformer {
    // This is in our examples/ directory and tested
}
```

Or for future features:

```markdown
> **Phase 2 Feature** (Not yet implemented)
>
> In Phase 2, we plan to add model loading from disk:
> ```text
> (Illustrative pseudocode - not in codebase)
> ```
```

### ❌ Pitfall: Outdated Code

```rust
// DON'T: Old API that's been refactored
let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
//                   ^^^ This API changed to from_vec()
```

### ✅ Solution: Always Reference Current Code

```rust
// ✓ DO: Use current API (from src/tensor.rs)
let tensor = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
```

## Building Locally

```bash
# Install mdbook (one-time setup)
cargo install mdbook

# Build the book
make book-build

# Serve with live reload (for writing)
make book-serve

# Validate code examples
make book-validate
```

## Submitting Changes

1. **Fork** the repository
2. **Create branch** from `main` (per CLAUDE.md: all work on main)
3. **Write** content following TDD structure
4. **Validate**: `make book-validate` passes
5. **Build**: `make book-build` succeeds
6. **Test**: `make quality-gates` all pass
7. **Commit** with descriptive message
8. **Push** to your fork
9. **Create PR** with summary of changes

## Quality Standards

All contributions must meet:

- ✅ Code examples are test-backed
- ✅ Book builds without errors
- ✅ Validation script passes
- ✅ No hallucinated examples
- ✅ Clear TDD narrative
- ✅ Proper attribution (file:line references)

## Getting Help

- **GitHub Issues**: Report documentation bugs
- **Discussions**: Ask questions about contributing
- **Examples**: Look at existing chapters for patterns

## Acknowledgments

This book follows the same anti-hallucination principles as:
- **aprender** - EXTREME TDD methodology guide
- **The Rust Book** - Technical documentation standards
- **mdBook** - Documentation as code

---

**Thank you for helping make Realizar's documentation test-backed and hallucination-free!** 📚✅
