# Transpiler Integration

Renacer supports source mapping for transpiled code, allowing you to trace binaries back to their original high-level source (Python, C, TypeScript, etc.) instead of just the generated Rust code.

## Overview

When you transpile code (e.g., Python → Rust via Depyler, C → Rust via Decy), Renacer can:
- Map syscalls back to original source files (`.py`, `.c`, `.ts`)
- Show original function names instead of generated ones
- Display original line numbers from your source code
- Track transpiler optimization decisions

## Supported Transpilers

### Depyler (Python → Rust)

```bash
# Transpile Python to Rust with source map
depyler transpile app.py --output app.rs --source-map app.map.json

# Compile with debug info
rustc app.rs -g -o app

# Trace with source mapping
renacer --transpiler-map app.map.json -- ./app
```

**Output:**
```text
read(3, buf, 1024) = 42    [app.py:15 in read_config]  ← Original Python!
```

### Decy (C → Rust)

```bash
# Transpile C to Rust with source map
decy convert main.c --output main.rs --source-map main.map.json

# Compile with debug info
rustc main.rs -g -o main

# Trace with source mapping
renacer --transpiler-map main.map.json -- ./main
```

**Output:**
```text
write(1, "Hello", 5) = 5   [main.c:42 in printf_wrapper]  ← Original C!
```

### Generic Transpiler Support

Renacer supports any transpiler that generates source maps in this format:

```json
{
  "version": 1,
  "source_language": "python",
  "target_language": "rust",
  "mappings": [
    {
      "generated_file": "app.rs",
      "generated_line": 150,
      "original_file": "app.py",
      "original_line": 15,
      "original_function": "read_config",
      "transpiler_decision": {
        "optimization": "inline_small_function",
        "reasoning": "Function body <10 lines"
      }
    }
  ]
}
```

## Source Map Format

### Required Fields

- `version`: Source map format version (currently `1`)
- `source_language`: Original language (e.g., "python", "c", "typescript")
- `target_language`: Target language (typically "rust")
- `mappings`: Array of line mappings

### Mapping Entry

Each mapping entry contains:

```json
{
  "generated_file": "output.rs",      // Generated Rust file
  "generated_line": 100,              // Line in generated code
  "original_file": "input.py",        // Original source file
  "original_line": 25,                // Line in original source
  "original_function": "my_function", // Original function name (optional)
  "transpiler_decision": {            // Optimization metadata (optional)
    "optimization": "vectorize_loop",
    "reasoning": "Simple iteration pattern detected"
  }
}
```

### Optional Fields

- `original_function`: Function name in original source
- `transpiler_decision`: Metadata about transpiler optimizations
  - `optimization`: Name of optimization applied
  - `reasoning`: Human-readable explanation

## Basic Usage

### 1. Simple Source Mapping

```bash
renacer --transpiler-map source.map.json -- ./app
```

Shows original source locations:
```text
openat(AT_FDCWD, "/config.json", O_RDONLY) = 3
  [config.py:10 in load_settings]

read(3, buf, 1024) = 512
  [config.py:11 in load_settings]
```

### 2. Combined with DWARF

```bash
renacer --source --transpiler-map source.map.json -- ./app
```

Renacer prefers transpiler mappings over DWARF when available:
1. Check transpiler map first
2. Fall back to DWARF debug info if no mapping found
3. Fall back to no source info if neither available

### 3. With Function Profiling

```bash
renacer --function-time --transpiler-map source.map.json -- ./app
```

**Output:**
```text
Function Profiling Summary:
========================
Top 10 Hot Paths (by total time):
  1. load_settings [config.py:10]      - 45.2% (1.2s, 67 syscalls)
  2. process_data [main.py:25]         - 32.1% (850ms, 45 syscalls)
  3. write_output [output.py:100]      - 15.3% (400ms, 23 syscalls)
```

Original function names from your source code!

### 4. With OTLP Export

```bash
renacer \
  --transpiler-map source.map.json \
  --otlp-endpoint http://localhost:4317 \
  -- ./app
```

Spans include transpiler attributes:
```json
{
  "source.file": "config.py",
  "source.line": 10,
  "source.function": "load_settings",
  "transpiler.source_language": "python",
  "transpiler.decision": "inline_small_function"
}
```

## Advanced Features

### Tracking Transpiler Decisions

Source maps can include optimization metadata:

```json
{
  "generated_line": 200,
  "original_line": 50,
  "transpiler_decision": {
    "optimization": "simd_vectorization",
    "reasoning": "Loop with constant stride, vectorizable"
  }
}
```

View in traces:
```bash
renacer --transpiler-map source.map.json --show-transpiler-decisions -- ./app
```

**Output:**
```text
read(3, buf, 8192) = 8192
  [data.py:50 in process_batch]
  💡 Transpiler: simd_vectorization (Loop with constant stride, vectorizable)
```

### Multi-Language Projects

Support multiple transpiled modules:

```bash
# Combine source maps
renacer \
  --transpiler-map module1.map.json \
  --transpiler-map module2.map.json \
  --transpiler-map module3.map.json \
  -- ./app
```

Renacer automatically:
- Merges all mappings
- Detects conflicts (warns if same generated line maps to multiple sources)
- Routes each syscall to correct source map

### Ruchy Runtime Integration

Renacer integrates with Ruchy Runtime for transpiler decision tracking:

```bash
# Trace with Ruchy runtime context
renacer \
  --transpiler-map output.map.json \
  --ruchy-trace-decisions \
  --otlp-endpoint http://localhost:4317 \
  -- ./ruchy-transpiled-app
```

This links:
- Syscalls → Original source lines
- Source lines → Transpiler decisions
- Decisions → Runtime performance
- Performance → OTLP observability backend

### Trueno SIMD Block Tracing

When tracing Trueno-accelerated code:

```bash
renacer \
  --transpiler-map trueno.map.json \
  --trace-simd-blocks \
  -- ./trueno-app
```

Renacer emits special spans for SIMD compute blocks:
```text
Span: simd_block
  source.file: stats.py
  source.line: 100
  source.function: calculate_percentiles
  simd.instruction_set: AVX2
  simd.vector_width: 256
  compute.block_id: trueno_block_42
```

## Generating Source Maps

### From Depyler

```bash
depyler transpile input.py \
  --output output.rs \
  --source-map output.map.json \
  --track-decisions
```

### From Decy

```bash
decy convert input.c \
  --output output.rs \
  --source-map output.map.json \
  --preserve-line-mapping
```

### Custom Transpiler

If building your own transpiler, implement source map generation:

```rust,ignore
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SourceMap {
    version: u32,
    source_language: String,
    target_language: String,
    mappings: Vec<Mapping>,
}

#[derive(Serialize, Deserialize)]
struct Mapping {
    generated_file: String,
    generated_line: u32,
    original_file: String,
    original_line: u32,
    original_function: Option<String>,
    transpiler_decision: Option<Decision>,
}

#[derive(Serialize, Deserialize)]
struct Decision {
    optimization: String,
    reasoning: String,
}

// Generate mappings during transpilation
fn transpile_with_mapping() {
    let mut mappings = Vec::new();

    // For each line transformation
    mappings.push(Mapping {
        generated_file: "output.rs".to_string(),
        generated_line: 100,
        original_file: "input.py".to_string(),
        original_line: 25,
        original_function: Some("process_data".to_string()),
        transpiler_decision: Some(Decision {
            optimization: "loop_unrolling".to_string(),
            reasoning: "Fixed iteration count detected".to_string(),
        }),
    });

    let source_map = SourceMap {
        version: 1,
        source_language: "python".to_string(),
        target_language: "rust".to_string(),
        mappings,
    };

    // Write to file
    std::fs::write(
        "output.map.json",
        serde_json::to_string_pretty(&source_map).unwrap()
    ).unwrap();
}
```

## Validation

Renacer validates source maps on load:

### Version Check

```text
Error: Unsupported source map version: 2 (expected: 1)
```

**Solution:** Update Renacer or regenerate source map with version 1.

### Required Fields

```text
Error: Missing required field: source_language
```

**Solution:** Ensure source map includes all required fields.

### Line Number Bounds

```text
Warning: Mapping references line 1000 in output.rs (file only has 500 lines)
```

**Solution:** Regenerate source map after modifying generated code.

## Best Practices

### 1. Always Generate with Debug Symbols

```bash
# ✅ Good: Debug symbols + source map
rustc output.rs -g -o app
renacer --transpiler-map output.map.json -- ./app

# ❌ Bad: Source map without debug symbols (limited utility)
rustc output.rs -o app
renacer --transpiler-map output.map.json -- ./app
```

### 2. Keep Source Maps Up-to-Date

```bash
# Regenerate after each transpilation
depyler transpile app.py --output app.rs --source-map app.map.json
```

### 3. Include Function Names

```json
{
  "original_function": "load_config",  // ✅ Helpful for profiling
  "original_function": null             // ❌ Less useful
}
```

### 4. Track Important Decisions

```json
{
  "transpiler_decision": {
    "optimization": "vectorize_loop",      // ✅ Useful for debugging
    "reasoning": "Performance boost +40%"
  }
}
```

### 5. Combine with OTLP for Observability

```bash
# Full stack observability with original source
renacer \
  --source \
  --transpiler-map app.map.json \
  --otlp-endpoint http://localhost:4317 \
  -- ./app
```

## Troubleshooting

### Source Map Not Found

```text
Error: Failed to read source map: No such file or directory
```

**Solution:**
```bash
# Verify file exists
ls -la output.map.json

# Use absolute path if needed
renacer --transpiler-map /absolute/path/to/output.map.json -- ./app
```

### Mappings Not Applied

```text
Warning: No mapping found for output.rs:150
```

**Causes:**
1. Source map incomplete (missing lines)
2. Source map out of date (code changed after generation)
3. Generated code modified after transpilation

**Solution:** Regenerate source map.

### Conflicting Mappings

```text
Warning: Multiple mappings for output.rs:100 (using first)
```

**Solution:** Check for duplicate entries in source map:
```bash
# Validate source map
jq '.mappings[] | select(.generated_line == 100)' output.map.json
```

### Original File Not Found

```text
Warning: Original file not found: input.py
```

**Solution:** Ensure original source files are accessible:
```bash
# Use absolute paths in source map
# Or ensure files are in working directory
```

## Performance Impact

Transpiler mapping overhead:
- **Loading source map:** One-time cost at startup (<10ms for 10K mappings)
- **Lookup per syscall:** <1μs (hash map lookup)
- **Total overhead:** <0.1% (negligible)

## Example: Python → Rust Workflow

Complete example with Depyler:

```bash
# 1. Write Python code
cat > app.py << 'EOF'
def read_config(path):
    with open(path) as f:
        return f.read()

def main():
    config = read_config("/etc/app.conf")
    print(f"Config: {config}")

if __name__ == "__main__":
    main()
EOF

# 2. Transpile to Rust with source map
depyler transpile app.py \
  --output app.rs \
  --source-map app.map.json \
  --track-decisions

# 3. Compile with debug symbols
rustc app.rs -g -o app

# 4. Trace with full observability
renacer \
  --source \
  --function-time \
  --transpiler-map app.map.json \
  --otlp-endpoint http://localhost:4317 \
  -- ./app

# Output shows Python source!
openat(AT_FDCWD, "/etc/app.conf", O_RDONLY) = 3
  [app.py:2 in read_config]

read(3, buf, 8192) = 156
  [app.py:3 in read_config]
  💡 Transpiler: buffered_io_optimization
```

## Next Steps

- [OpenTelemetry Integration](./opentelemetry.md) - Export transpiler metadata
- [Distributed Tracing](./distributed-tracing.md) - Trace across services
- [Function Profiling](./function-profiling.md) - Profile with original names
