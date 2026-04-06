# Tool Selection

Batuta orchestrates external transpiler tools rather than implementing transpilation itself. The `ToolRegistry` detects which tools are available on the system and selects the appropriate one for each source language.

## Tool Detection

On startup, `ToolRegistry::detect()` probes the system PATH for each known tool using the `which` crate:

```rust
fn detect_tool(name: &str) -> Option<ToolInfo> {
    let path = which::which(name).ok()?;
    let version = get_tool_version(name);
    Some(ToolInfo {
        name: name.to_string(),
        version,
        path: path.to_string_lossy().to_string(),
        available: true,
    })
}
```

Version detection runs `<tool> --version` and extracts the version string from the last whitespace-delimited token in the first line of output.

## Registry Contents

The full registry checks for nine tools:

| Tool | Purpose | Install Command |
|------|---------|----------------|
| `depyler` | Python to Rust | `cargo install depyler` |
| `decy` | C/C++ to Rust | `cargo install decy` |
| `bashrs` | Shell to Rust | `cargo install bashrs` |
| `ruchy` | Rust scripting | `cargo install ruchy` |
| `pmat` | Quality analysis | `cargo install pmat` |
| `trueno` | SIMD/GPU compute | Cargo.toml dependency |
| `aprender` | ML library | Cargo.toml dependency |
| `realizar` | Inference runtime | Cargo.toml dependency |
| `renacer` | Syscall tracing | `cargo install renacer` |

## Fallback Strategies

When a required transpiler is missing, Batuta provides actionable installation instructions:

```bash
$ batuta transpile

Error: No transpiler available for Python
Install Depyler: cargo install depyler
```

The `get_installation_instructions()` method generates per-tool instructions. CLI tools use `cargo install`, while library crates reference Cargo.toml additions.

## Version Compatibility

Each transpiler version is recorded in the `ToolInfo` struct. Batuta logs the detected version at the start of transpilation for reproducibility. Future versions will enforce minimum version requirements to prevent compatibility issues.

## Checking Available Tools

```bash
$ batuta tools

Detected Tools
--------------
Depyler (Python -> Rust)     v2.1.0  /usr/local/bin/depyler
Bashrs (Shell -> Rust)       v1.3.0  /usr/local/bin/bashrs
PMAT (Quality analysis)      v1.8.0  /usr/local/bin/pmat
Renacer (Syscall tracing)    v0.9.0  /usr/local/bin/renacer

Missing:
  Decy (C/C++ -> Rust)       cargo install decy
  Ruchy (Rust scripting)     cargo install ruchy
```

## Tool Invocation

All tool invocation goes through the `run_tool()` function in `src/tools.rs`, which captures stdout and stderr, checks exit codes, and wraps failures in structured `anyhow` errors with the tool name and exit code.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
