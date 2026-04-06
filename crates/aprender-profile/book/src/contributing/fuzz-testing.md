# Fuzz Testing

Renacer uses **cargo-fuzz** for coverage-guided fuzz testing to discover edge cases and security vulnerabilities in parser code and other critical components.

## Overview

Fuzz testing generates random inputs to test code robustness. Unlike unit tests with specific inputs, fuzzing explores the input space automatically to find crashes, panics, and unexpected behavior.

**When to Use Fuzz Testing:**
- Parser implementations (filter expressions, syscall names)
- Input validation code
- Binary format parsers (DWARF, ELF)
- Serialization/deserialization code

## Infrastructure (Sprint 29)

### Setup

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# List available fuzz targets
cargo fuzz list

# Run a specific fuzz target
cargo fuzz run filter_parser
```

### Fuzz Targets

#### Filter Parser (`fuzz/fuzz_targets/filter_parser.rs`)

Tests the `SyscallFilter::from_expr()` parser with arbitrary byte sequences:

```rust,ignore
#![no_main]

use libfuzzer_sys::fuzz_target;
use renacer::filter::SyscallFilter;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to UTF-8 string (lossy conversion)
    if let Ok(input) = std::str::from_utf8(data) {
        // Attempt to parse the filter expression
        // This should not panic regardless of input
        let _ = SyscallFilter::from_expr(input);
    }
});
```

**What it Tests:**
- Invalid regex patterns
- Malformed class names
- Edge cases in negation operator
- Unusual character combinations
- Empty strings, null bytes, unicode

### Running Fuzz Tests

```bash
# Run filter_parser fuzz target
make fuzz

# Run with specific duration
cargo fuzz run filter_parser -- -max_total_time=300  # 5 minutes

# Run with specific number of runs
cargo fuzz run filter_parser -- -runs=1000000

# Minimize a crashing input
cargo fuzz cmin filter_parser
```

### Analyzing Crashes

If fuzzing finds a crash:

```bash
# Crashes are saved to fuzz/artifacts/filter_parser/
ls fuzz/artifacts/filter_parser/

# Reproduce a crash
cargo fuzz run filter_parser fuzz/artifacts/filter_parser/crash-abc123

# View the input that caused the crash
hexdump -C fuzz/artifacts/filter_parser/crash-abc123
```

## Integration with EXTREME TDD

Fuzz testing is part of **Tier 3** testing workflow:

```bash
# Tier 3: Slow tests (<5 minutes)
make test-tier3
```

This runs:
1. Fuzz tests (short run)
2. Mutation tests
3. Long-running integration tests

## Best Practices

### 1. Test Invariants, Not Specific Behavior

Good fuzz target:
```rust,ignore
fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        // Should never panic
        let _ = parse_something(input);
    }
});
```

Bad fuzz target:
```rust,ignore
fuzz_target!(|data: &[u8]| {
    // Too specific - won't discover interesting inputs
    assert_eq!(parse_number(data), expected_value);
});
```

### 2. Use `arbitrary` Crate for Structured Fuzzing

For complex inputs:

```toml
[dependencies]
arbitrary = { version = "1.3", features = ["derive"], optional = true }
```

```rust,ignore
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct FilterExpr {
    trace_spec: String,
    negations: Vec<String>,
}

fuzz_target!(|input: FilterExpr| {
    // Fuzz with structured data
    let _ = SyscallFilter::from_parts(&input.trace_spec, &input.negations);
});
```

### 3. Continuous Fuzzing

Run fuzzing on CI/CD for extended periods:

```yaml
# .github/workflows/fuzz.yml
- name: Fuzz for 1 hour
  run: cargo fuzz run filter_parser -- -max_total_time=3600
```

### 4. Corpus Management

Save interesting inputs:

```bash
# Export corpus for long-term fuzzing
cp -r fuzz/corpus/filter_parser fuzz/corpus-backup/

# Merge multiple corpora
cargo fuzz cmin filter_parser corpus1 corpus2 corpus3
```

## Coverage-Guided Fuzzing

Cargo-fuzz uses **libFuzzer** which provides:
- **Coverage Feedback**: Prioritizes inputs that increase code coverage
- **Mutation Strategies**: Generates new inputs by mutating previous ones
- **Crash Detection**: Automatically saves crashing inputs

### Viewing Coverage

```bash
# Generate coverage report
cargo fuzz coverage filter_parser
llvm-cov show target/*/release/filter_parser -format=html -instr-profile=fuzz/coverage/filter_parser/coverage.profdata > coverage.html
```

## Example: Finding Edge Cases

Fuzz testing discovered these edge cases in Renacer:

1. **Empty Regex Pattern**: `/(?:)/` (valid but unusual)
2. **Unicode in Class Names**: `trace=file�invalid`
3. **Nested Negations**: `trace=!!open` (double negation)
4. **Malformed UTF-8**: Filter parser handles invalid UTF-8 gracefully

## Troubleshooting

### Slow Fuzzing

```bash
# Use more cores
cargo fuzz run filter_parser -- -workers=8

# Reduce memory limit
cargo fuzz run filter_parser -- -rss_limit_mb=2048
```

### Out of Memory

```bash
# Limit input size
cargo fuzz run filter_parser -- -max_len=1024
```

### No New Coverage

```bash
# Try different mutation strategies
cargo fuzz run filter_parser -- -mutate_depth=5

# Seed with interesting inputs
echo "trace=file,!close" > fuzz/corpus/filter_parser/seed1
```

## Future Fuzz Targets

Planned for upcoming sprints:
- `syscall_name_parser` - Test syscall name resolution
- `dwarf_line_parser` - Test DWARF debug info parsing
- `json_serializer` - Test JSON output serialization
- `transpiler_map_parser` - Test source map parsing

## Resources

- [cargo-fuzz Documentation](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer Tutorial](https://llvm.org/docs/LibFuzzer.html)
- [Rust Fuzzing Authority](https://github.com/rust-fuzz)
