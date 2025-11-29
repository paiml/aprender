# Case Study: CITL Automated Program Repair

Using the Compiler-in-the-Loop Learning module for automated Rust code repair.

## Overview

The `aprender::citl` module provides a complete system for:
- Parsing compiler diagnostics
- Encoding errors into embeddings for pattern matching
- Suggesting and applying fixes
- Tracking metrics for continuous improvement
- **SIMD-accelerated similarity search via trueno**

## Basic Usage

```rust
use aprender::citl::{CITL, CITLBuilder, CompilerMode};

// Create CITL instance with Rust compiler
let citl = CITLBuilder::new()
    .with_compiler(CompilerMode::Rustc)
    .max_iterations(5)
    .confidence_threshold(0.7)
    .build()
    .expect("Failed to create CITL instance");

// Source code with a type error
let source = r#"
fn main() {
    let x: i32 = "hello";
}
"#;

// Get fix suggestions
if let Some(suggestion) = citl.suggest_fix(source, source) {
    println!("Suggested fix: {}", suggestion.description);
    println!("Confidence: {:.1}%", suggestion.confidence * 100.0);
}
```

## Iterative Fix Loop

The `fix_all` method attempts to fix all errors iteratively:

```rust
use aprender::citl::{CITL, CITLBuilder, CompilerMode, FixResult};

let citl = CITLBuilder::new()
    .with_compiler(CompilerMode::Rustc)
    .max_iterations(10)
    .build()
    .expect("CITL build failed");

let buggy_code = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result: String = add(1, 2);
    println!("{}", result);
}
"#;

match citl.fix_all(buggy_code) {
    FixResult::Success { fixed_code, iterations, fixes_applied } => {
        println!("Fixed in {} iterations!", iterations);
        println!("Applied {} fixes", fixes_applied.len());
        println!("Fixed code:\n{}", fixed_code);
    }
    FixResult::Failure { last_code, remaining_errors, .. } => {
        println!("Could not fully fix. {} errors remain.", remaining_errors);
    }
}
```

## Cargo Mode for Dependencies

When code requires external crates, use Cargo mode:

```rust
use aprender::citl::{CITL, CITLBuilder, CompilerMode};

let citl = CITLBuilder::new()
    .with_compiler(CompilerMode::Cargo)  // Uses cargo check
    .build()
    .expect("CITL build failed");

let code_with_deps = r#"
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Config {
    name: String,
    value: i32,
}

fn main() {
    let config = Config { name: "test".into(), value: 42 };
    println!("{}", serde_json::to_string(&config).unwrap());
}
"#;

// Cargo mode resolves dependencies automatically
if let Some(fix) = citl.suggest_fix(code_with_deps, code_with_deps) {
    println!("Fix: {}", fix.description);
}
```

## Pattern Library

The pattern library stores learned error-fix mappings:

```rust
use aprender::citl::{PatternLibrary, ErrorFixPattern, FixTemplate};

let mut library = PatternLibrary::new();

// Add a custom pattern
let pattern = ErrorFixPattern {
    error_code: "E0308".to_string(),
    error_message_pattern: "expected `i32`, found `String`".to_string(),
    context_pattern: "let.*:.*i32.*=".to_string(),
    fix_template: FixTemplate::type_conversion("i32", ".parse().unwrap()"),
    success_count: 0,
    failure_count: 0,
};

library.add_pattern(pattern);

// Save patterns for persistence
library.save("patterns.citl").expect("Save failed");

// Load patterns later
let loaded = PatternLibrary::load("patterns.citl").expect("Load failed");
```

## Built-in Fix Templates

The module includes 21 fix templates for common errors:

### E0308 - Type Mismatch
- `type_annotation` - Add explicit type annotation
- `type_conversion` - Add conversion method (.into(), .to_string())
- `reference_conversion` - Convert between & and owned types

### E0382 - Use of Moved Value
- `borrow_instead_of_move` - Change to borrow
- `rc_wrap` - Wrap in Rc for shared ownership
- `arc_wrap` - Wrap in Arc for thread-safe sharing

### E0277 - Trait Bound Not Satisfied
- `derive_debug` - Add #[derive(Debug)]
- `derive_clone_trait` - Add #[derive(Clone)]
- `impl_display` - Implement Display trait
- `impl_from` - Implement From trait

### E0515 - Cannot Return Reference
- `return_owned` - Return owned value instead
- `return_cloned` - Clone and return
- `use_cow` - Use Cow<'a, T> for flexibility

## Metrics Tracking

Track performance with the built-in metrics system:

```rust
use aprender::citl::{MetricsTracker, MetricsSummary};
use std::time::Duration;

let mut metrics = MetricsTracker::new();

// Record fix attempts
metrics.record_fix_attempt(true, "E0308");
metrics.record_fix_attempt(true, "E0308");
metrics.record_fix_attempt(false, "E0382");

// Record pattern usage
metrics.record_pattern_use(0, true);  // Pattern 0 succeeded
metrics.record_pattern_use(1, false); // Pattern 1 failed

// Record compilation times
metrics.record_compilation_time(Duration::from_millis(150));
metrics.record_compilation_time(Duration::from_millis(200));

// Record convergence (iterations to fix)
metrics.record_convergence(2, true);  // Fixed in 2 iterations
metrics.record_convergence(5, false); // Failed after 5 iterations

// Get summary
let summary = metrics.summary();
println!("{}", summary.to_report());
```

Output:
```
=== CITL Metrics Summary ===

Fix Attempts: 3 (success rate: 66.7%)
Compilations: 2 (avg time: 175.0ms)
Convergence: 50.0% (avg 3.5 iterations)

Most Common Errors:
  E0308: 2
  E0382: 1

Session Duration: 1.2s
```

## Error Embedding

The encoder converts errors into embeddings for similarity matching:

```rust
use aprender::citl::ErrorEncoder;

let encoder = ErrorEncoder::new();

// Encode a diagnostic
let diagnostic = "error[E0308]: mismatched types, expected i32 found String";
let embedding = encoder.encode(diagnostic, "let x: i32 = get_string();");

// Embeddings can be compared for similarity
// Similar errors produce similar embeddings
```

## Integration Test Example

```rust
#[test]
fn test_citl_fixes_type_mismatch() {
    let citl = CITLBuilder::new()
        .with_compiler(CompilerMode::Rustc)
        .max_iterations(3)
        .build()
        .unwrap();

    let source = r#"
fn main() {
    let x: i32 = "42";
}
"#;

    let result = citl.fix_all(source);
    assert!(matches!(result, FixResult::Success { .. }));
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CITL Module                             │
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────┐      │
│   │ Compiler  │───►│  Parser   │───►│  Error Encoder    │      │
│   │ Interface │    │ (JSON)    │    │  (Embeddings)     │      │
│   └───────────┘    └───────────┘    └─────────┬─────────┘      │
│                                               │                 │
│                                               ▼                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────┐      │
│   │  Apply    │◄───│  Pattern  │◄───│  Pattern Library  │      │
│   │   Fix     │    │  Matcher  │    │  (21 Templates)   │      │
│   └───────────┘    └─────┬─────┘    └───────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                    trueno                            │      │
│   │         SIMD Vector Operations (CPU/GPU)             │      │
│   │    dot() • norm_l2() • sub() • normalize()           │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              Metrics Tracker                         │      │
│   │  (Success Rate, Compilation Time, Convergence)       │      │
│   └─────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Neural Encoder (Multi-Language)

For cross-language transpilation (Python→Rust, Julia→Rust, etc.), use the neural encoder:

```rust
use aprender::citl::{NeuralErrorEncoder, NeuralEncoderConfig, ContrastiveLoss};

// Create encoder with configuration
let config = NeuralEncoderConfig::small();  // 128-dim embeddings
let encoder = NeuralErrorEncoder::with_config(config);

// Encode errors from different languages
let rust_emb = encoder.encode(
    "E0308: mismatched types, expected i32 found &str",
    "let x: i32 = \"hello\";",
    "rust",
);

let python_emb = encoder.encode(
    "TypeError: expected int, got str",
    "x: int = \"hello\"",
    "python",
);

// Similar type errors cluster together in embedding space
```

### Training with Contrastive Loss

```rust
let mut encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::default());
encoder.train();  // Enable training mode

// Encode batch of anchors and positives
let anchors = &[
    ("E0308: type mismatch", "let x: i32 = s;", "rust"),
    ("E0382: moved value", "let y = x; let z = x;", "rust"),
];
let positives = &[
    ("E0308: expected i32", "let a: i32 = b;", "rust"),
    ("E0382: borrow after move", "let p = q; let r = q;", "rust"),
];

let anchor_emb = encoder.encode_batch(anchors);
let positive_emb = encoder.encode_batch(positives);

// InfoNCE contrastive loss
let loss_fn = ContrastiveLoss::with_temperature(0.07);
let loss = loss_fn.forward(&anchor_emb, &positive_emb, None);
```

### Configuration Options

| Config | Embed Dim | Layers | Encode Time |
|--------|-----------|--------|-------------|
| `minimal()` | 64 | 1 | 132 µs |
| `small()` | 128 | 2 | 919 µs |
| `default()` | 256 | 2 | ~2 ms |

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Tokenizer   │────►│  Embedding  │────►│ Transformer │────►│ L2 Norm     │
│ (8K vocab)  │     │ + Position  │     │ (N layers)  │     │ (SIMD)      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

Supported languages: `rust`, `python`, `julia`, `typescript`, `go`, `java`, `cpp`

## Key Types

| Type | Purpose |
|------|---------|
| `CITL` | Main orchestrator for fix operations |
| `CITLBuilder` | Builder pattern for configuration |
| `CompilerMode` | Rustc, Cargo, or CargoCheck |
| `PatternLibrary` | Stores error-fix patterns |
| `FixTemplate` | Describes how to apply a fix |
| `ErrorEncoder` | Hand-crafted feature embeddings |
| `NeuralErrorEncoder` | Transformer-based embeddings (GPU) |
| `ContrastiveLoss` | InfoNCE loss for training |
| `MetricsTracker` | Performance tracking |
| `FixResult` | Success/Failure with details |

## Performance Characteristics

CITL uses **trueno** for SIMD-accelerated vector operations:

| Operation | Time | Throughput |
|-----------|------|------------|
| Cosine similarity (256-dim) | 122 ns | 2.1 Gelem/s |
| Cosine similarity (1024-dim) | 375 ns | 2.7 Gelem/s |
| L2 distance (256-dim) | 147 ns | 1.7 Gelem/s |
| Pattern search (100 patterns) | 9.3 µs | 10.7 Melem/s |
| Batch similarity (500 comparisons) | 40 µs | 12.4 Melem/s |

**Complexity:**
- **Pattern matching**: O(n) where n = number of patterns
- **Embedding generation**: O(m) where m = diagnostic length
- **Fix application**: O(1) string replacement
- **Persistence**: Binary format with CITL magic header

**GPU Acceleration:**

Enable GPU via trueno's wgpu backend:

```bash
cargo build --features gpu
```

### Running Benchmarks

```bash
cargo bench --bench citl
```

Benchmark groups:
- `citl_cosine_similarity` - Core SIMD similarity
- `citl_l2_distance` - Euclidean distance
- `citl_pattern_search` - Library search scaling
- `citl_error_encoding` - Full encoding pipeline
- `citl_batch_similarity` - Batch comparison throughput
- `citl_neural_encoder` - Transformer encoding
- `citl_neural_config` - Config comparison

## Build-Time Performance Assertions

Beyond correctness, CITL systems enforce **performance contracts** at build time using the renacer.toml DSL.

### renacer.toml Configuration

```toml
[package]
name = "my-transpiled-cli"
version = "0.1.0"

[performance]
# Fail build if startup exceeds 50ms
startup_time_ms = 50

# Fail if binary exceeds 5MB
binary_size_mb = 5

# Memory usage assertions
[performance.memory]
peak_rss_mb = 100
heap_allocations_max = 10000

# Syscall budget per operation
[performance.syscalls]
file_read = 50
file_write = 25
network_connect = 5

# Regression detection
[performance.regression]
baseline = "baseline.json"
max_regression_percent = 5.0
```

### Build-Time Validation

```bash
# Run performance assertions during build
cargo build --release

# renacer validates assertions automatically
[PASS] startup_time: 23ms (limit: 50ms)
[PASS] binary_size: 2.1MB (limit: 5MB)
[PASS] peak_rss: 24MB (limit: 100MB)
[PASS] syscalls/file_read: 12 (limit: 50)
[FAIL] syscalls/network_connect: 8 (limit: 5)

error: Performance assertion failed
  --> renacer.toml:18:1
   |
18 | network_connect = 5
   | ^^^^^^^^^^^^^^^^^^^ actual: 8, limit: 5
   |
   = help: Consider batching network operations or using connection pooling
```

### Real-World Performance Improvements

The reprorusted-python-cli project demonstrates dramatic improvements achieved through CITL transpilation with performance assertions:

```
┌─────────────────────────────────────────────────────────────────┐
│           REPRORUSTED-PYTHON-CLI BENCHMARK RESULTS              │
│                                                                 │
│   Operation          Python      Rust        Improvement        │
│   ────────────────   ──────      ────        ───────────        │
│   CSV parse (10MB)   2.3s        0.08s       28.7× faster       │
│   JSON serialize     890ms       31ms        28.7× faster       │
│   Regex matching     1.2s        0.11s       10.9× faster       │
│   HTTP requests      4.5s        0.42s       10.7× faster       │
│                                                                 │
│   Resource Usage:                                               │
│   Total syscalls     185,432     10,073      18.4× fewer        │
│   Memory allocs      45,231      2,891       15.6× fewer        │
│   Peak memory        127.4MB     23.8MB      5.4× smaller       │
│                                                                 │
│   Binary Size:       N/A         2.1MB       (static linked)    │
│   Startup Time:      ~500ms      23ms        21.7× faster       │
└─────────────────────────────────────────────────────────────────┘
```

### Syscall Budget Enforcement

The DSL supports fine-grained syscall budgets:

```toml
[performance.syscalls]
# I/O operations
read = 100
write = 50
open = 20
close = 20

# Memory operations
mmap = 10
munmap = 10
brk = 5

# Process operations
clone = 2
execve = 1
fork = 0  # Forbidden

# Network operations
socket = 5
connect = 5
sendto = 100
recvfrom = 100
```

### Integration with CI/CD

```yaml
# .github/workflows/performance.yml
name: Performance Gates

on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build with assertions
        run: cargo build --release

      - name: Run renacer validation
        run: |
          renacer validate --config renacer.toml
          renacer compare --baseline baseline.json --report pr-perf.md

      - name: Upload performance report
        uses: actions/upload-artifact@v4
        with:
          name: performance-report
          path: pr-perf.md

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('pr-perf.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### Profiling Integration

Use renacer with profiling tools for detailed analysis:

```bash
# Generate syscall trace
renacer profile --trace syscalls ./target/release/my-cli

# Analyze allocation patterns
renacer profile --trace allocations ./target/release/my-cli

# Compare against baseline
renacer diff baseline.trace current.trace --format markdown
```

Output:
```markdown
## Syscall Comparison

| Syscall | Baseline | Current | Delta |
|---------|----------|---------|-------|
| read    | 45       | 12      | -73%  |
| write   | 23       | 8       | -65%  |
| mmap    | 156      | 4       | -97%  |
| **Total** | **1,203** | **89** | **-93%** |
```

## See Also

- [Compiler-in-the-Loop Learning Theory](../ml-fundamentals/compiler-in-the-loop.md)
- [Building Custom Error Classifiers](./custom-error-classifier.md)
