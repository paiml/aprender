# Compile Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §17
**Status**: Active
**CLI**: `apr compile`
**Implementation**: `crates/apr-cli/src/commands/compile.rs`

---

## 1. Overview

Compile a model into a standalone executable binary with embedded weights.
The output binary runs inference without requiring the apr CLI or any
runtime dependencies.

## 2. CLI Interface

```
apr compile <FILE> \
  --output <PATH> \
  [--target <TRIPLE>] \
  [--quantize <int8|int4|fp16>] \
  [--release] [--strip] [--lto] \
  [--list-targets]
```

## 3. Supported Targets

| Target | Description |
|--------|-------------|
| `x86_64-unknown-linux-gnu` | Linux x86_64 (default) |
| `x86_64-unknown-linux-musl` | Linux static binary |
| `aarch64-unknown-linux-gnu` | Linux ARM64 |
| `aarch64-apple-darwin` | macOS Apple Silicon |
| `wasm32-wasi` | WASM (via wasi) |

## 4. Features

- **Quantization during compilation**: Reduce embedded weight size
- **Release mode**: Optimized with LTO
- **Strip**: Remove debug symbols
- **Cross-compilation**: Via rustup targets

---

## Provable Contracts

### Contract: `compile-v1.yaml`

```yaml
metadata:
  description: "Model compilation to standalone binary"
  depends_on:
    - "apr-format-v2"

equations:
  binary_embedding:
    formula: "binary = runtime_code + embedded_model_data"
    invariants:
      - "Embedded model produces same inference as original"
      - "Binary is self-contained (no external files needed)"
      - "Model data integrity preserved (checksum verified)"

  cross_compilation:
    formula: "compile(model, target) produces valid ELF/Mach-O/WASM"
    invariants:
      - "Output format matches target triple"
      - "SIMD dispatch matches target architecture"

proof_obligations:
  - type: equivalence
    property: "Inference parity"
    formal: "compiled_binary(prompt) == apr_run(model, prompt)"
  - type: invariant
    property: "Self-contained binary"
    formal: "binary runs without model file on PATH"
  - type: invariant
    property: "Target format"
    formal: "x86_64 target → ELF, aarch64-apple → Mach-O, wasm32 → WASM"

falsification_tests:
  - id: FALSIFY-COMPILE-001
    rule: "Inference parity"
    prediction: "compiled binary produces same tokens as apr run"
    if_fails: "Model embedding corrupts weights"
  - id: FALSIFY-COMPILE-002
    rule: "Self-contained"
    prediction: "binary runs on clean machine without model file"
    if_fails: "Runtime tries to load external model"
  - id: FALSIFY-COMPILE-003
    rule: "Invalid target rejection"
    prediction: "unsupported target triple → clear error"
    if_fails: "Missing target validation"
  - id: FALSIFY-COMPILE-004
    rule: "Quantize + compile"
    prediction: "compile with --quantize produces smaller binary"
    if_fails: "Quantization not applied before embedding"
```

### Binding Requirements

```yaml
  - contract: compile-v1.yaml
    equation: binary_embedding
    module_path: "aprender::compile"
    function: compile_model
    status: implemented
```
