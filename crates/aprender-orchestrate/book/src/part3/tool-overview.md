# Tool Overview

Batuta does not transpile code itself. It orchestrates a curated ecosystem of external tools, each purpose-built for a specific language or task. Tools are organized into three categories: **transpilers** that convert source languages to Rust, **foundation libraries** that provide compute and ML primitives, and **support tools** that handle analysis, testing, and tracing.

## Tool Categories

### Transpilers

Transpilers convert source code from one language to idiomatic Rust. Batuta selects the appropriate transpiler based on the detected source language.

| Tool | Direction | Install | Status |
|------|-----------|---------|--------|
| **Depyler** | Python to Rust | `cargo install depyler` | Production |
| **Decy** | C/C++ to Rust | `cargo install decy` | Production |
| **Bashrs** | Rust to Shell | `cargo install bashrs` | Production |

### Foundation Libraries

Foundation libraries are Rust crates used as dependencies in generated code. They replace source-language libraries with SIMD/GPU-accelerated Rust equivalents.

| Library | Purpose | crates.io |
|---------|---------|-----------|
| **Trueno** | SIMD/GPU compute primitives (AVX2, AVX-512, NEON, wgpu) | `trueno` |
| **Aprender** | ML algorithms, APR v2 model format | `aprender` |
| **Realizar** | Inference runtime with quantized kernels | `realizar` |
| **Repartir** | Distributed compute (CPU, GPU, remote) | `repartir` |
| **Trueno-zram** | SIMD-accelerated compression (LZ4, ZSTD) | `trueno-zram-core` |
| **Whisper.apr** | Pure Rust speech recognition | `whisper-apr` |

### Support Tools

Support tools assist with quality analysis, runtime validation, and scripting.

| Tool | Purpose | Install |
|------|---------|---------|
| **PMAT** | Static analysis and TDG scoring | `cargo install pmat` |
| **Renacer** | Syscall tracing for semantic validation | `cargo install renacer` |
| **Ruchy** | Rust scripting for automation | `cargo install ruchy` |

## Tool Detection

Batuta discovers tools automatically at startup using PATH-based detection. The `ToolRegistry` struct in `src/tools.rs` drives this process:

```rust
// Batuta scans PATH for each known tool
let registry = ToolRegistry::detect();

// Check what is available
for tool in registry.available_tools() {
    println!("Found: {}", tool);
}
```

Detection follows three steps:

1. **PATH lookup** -- `which::which(name)` locates the binary
2. **Version probe** -- runs `tool --version` and parses the output
3. **Registry population** -- stores name, path, version, and availability flag

If a tool is missing, Batuta provides installation instructions:

```bash
$ batuta analyze --input project/
Warning: Depyler not found. Install with: cargo install depyler
```

## Language-to-Tool Mapping

When Batuta encounters source files, it maps the detected language to the appropriate transpiler:

| Source Language | Transpiler | Generated Dependencies |
|----------------|------------|----------------------|
| Python | Depyler | trueno, aprender, realizar |
| C / C++ | Decy | (pure Rust output) |
| Shell | Bashrs | (POSIX shell output) |
| Rust | (no transpilation) | -- |

Languages without a matching transpiler are reported but not processed. Batuta never guesses -- if the right tool is not installed, the pipeline stops with a clear error (Jidoka principle).

## Checking Tool Status

```bash
# List all detected tools
batuta analyze --tools

# Install all stack tools at once
cargo install depyler decy bashrs pmat renacer ruchy
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
