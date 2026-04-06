# GitHub Issue: bashrs Integration with cargo xtask Pattern

**Repository**: https://github.com/paiml/bashrs
**Title**: Support cargo xtask-like Integration for Natural Workflow

---

## Summary

Enable bashrs to work naturally within cargo's xtask pattern, allowing developers to transpile Rust → Shell scripts as part of their standard development workflow without requiring `cargo install bashrs`.

## Problem Statement

**Current State**:
- bashrs requires global installation: `cargo install bashrs`
- Must be invoked as separate binary: `bashrs transpile src/script.rs`
- Not integrated with project workspace
- Separate from project's development tooling
- Adds external dependency for contributors

**Desired State**:
- bashrs integrated as workspace member (like xtask)
- Invoked via project commands: `cargo xtask transpile-hooks`
- No global installation required
- Single `cargo` command for all tooling
- Contributors get bashrs automatically with `cargo build`

## Use Case: Git Hooks in Rust

**Concrete Example** (from trueno project):

We want to write git pre-commit hooks in Rust and transpile them to shell:

```rust
// xtask/src/hooks/pre_commit.rs
use bashrs::prelude::*;

#[bash_function]
fn check_simd() -> BashResult {
    println!("🔍 Running SIMD validation...");

    let output = cmd!("cargo xtask check-simd")?;

    if output.status.success() {
        Ok(())
    } else {
        eprintln!("❌ SIMD validation failed");
        Err(BashError::ExitCode(1))
    }
}

#[bash_main]
fn main() -> BashResult {
    check_simd()?;
    Ok(())
}
```

**Desired Workflow**:
```bash
# Transpile hooks (integrated with xtask)
cargo xtask transpile-hooks

# Or automatic transpilation during build
cargo build  # also transpiles if hooks/*.rs changed

# Install hooks (transpiled shell script)
cargo xtask install-hooks
```

## Proposed Solution

### Option 1: bashrs as Workspace Dependency

**Structure**:
```
my-project/
├── Cargo.toml          # workspace
├── src/                # main library
├── xtask/
│   ├── Cargo.toml     # [dependencies] bashrs = "6.35"
│   ├── src/
│   │   ├── main.rs
│   │   ├── check_simd.rs
│   │   └── hooks/     # Rust code with bashrs annotations
│   │       ├── pre_commit.rs
│   │       └── post_merge.rs
│   └── build.rs       # transpiles hooks/*.rs → .git/hooks/*
```

**xtask/build.rs**:
```rust
use bashrs::Transpiler;

fn main() {
    let transpiler = Transpiler::new();

    // Transpile pre-commit hook
    transpiler
        .input("src/hooks/pre_commit.rs")
        .output(".git/hooks/pre-commit")
        .permissions(0o755)
        .transpile()
        .expect("Failed to transpile pre-commit hook");
}
```

**Benefits**:
- No global installation required
- Automatic transpilation during build
- Version-locked with project (Cargo.lock)
- Works in CI/CD without extra setup
- Contributors get bashrs automatically

### Option 2: Cargo Subcommand (like cargo-xtask)

Create `cargo-bashrs` subcommand that integrates with workspace:

```bash
# Install (one-time for development)
cargo install cargo-bashrs

# Use in project
cargo bashrs transpile xtask/src/hooks/pre_commit.rs --output .git/hooks/pre-commit
cargo bashrs watch xtask/src/hooks/**/*.rs --output-dir .git/hooks/
```

**Configuration** in Cargo.toml:
```toml
[package.metadata.bashrs]
source_dir = "xtask/src/hooks"
output_dir = ".git/hooks"
watch = true
permissions = 0o755
```

### Option 3: Hybrid Approach (Recommended)

Combine both:

1. **bashrs as library dependency** in xtask
2. **xtask command** for transpilation
3. **build.rs** for automatic transpilation

**Example**:

```rust
// xtask/src/transpile_hooks.rs
use bashrs::Transpiler;
use anyhow::Result;

pub fn run() -> Result<()> {
    println!("🔄 Transpiling git hooks...");

    let hooks = [
        ("src/hooks/pre_commit.rs", ".git/hooks/pre-commit"),
        ("src/hooks/post_merge.rs", ".git/hooks/post-merge"),
    ];

    for (input, output) in hooks {
        Transpiler::new()
            .input(input)
            .output(output)
            .permissions(0o755)
            .transpile()?;

        println!("✅ Transpiled: {}", output);
    }

    Ok(())
}
```

**Usage**:
```bash
# Manual transpilation
cargo xtask transpile-hooks

# Automatic during build (via build.rs)
cargo build

# Install hooks (uses transpiled scripts)
cargo xtask install-hooks
```

## API Requirements

To support this pattern, bashrs needs:

### 1. Library API (not just binary)

```rust
// Current: bashrs is primarily a CLI tool
// Needed: Programmatic API

pub struct Transpiler {
    input: PathBuf,
    output: PathBuf,
    permissions: Option<u32>,
    options: TranspileOptions,
}

impl Transpiler {
    pub fn new() -> Self;
    pub fn input(&mut self, path: impl AsRef<Path>) -> &mut Self;
    pub fn output(&mut self, path: impl AsRef<Path>) -> &mut Self;
    pub fn permissions(&mut self, mode: u32) -> &mut Self;
    pub fn transpile(&mut self) -> Result<TranspileResult>;
}
```

### 2. Build Script Integration

```rust
// bashrs/src/build.rs
pub mod build {
    pub fn transpile_hooks() -> Result<()> {
        // Auto-discover hooks/*.rs
        // Transpile to configured output
        // Set executable permissions
    }
}
```

### 3. File Watcher API (for development)

```rust
pub fn watch(
    source_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<Watcher> {
    // Watch source files
    // Auto-transpile on changes
    // Hot-reload for development
}
```

## Benefits

### For bashrs

1. **Increased Adoption**: Natural integration with Rust projects
2. **Better UX**: No global installation, works out-of-box
3. **CI/CD Friendly**: No extra setup steps
4. **Version Control**: bashrs version locked with project

### For Projects

1. **Type Safety**: Write hooks in Rust with compile-time checks
2. **Testing**: Unit test hooks before transpilation
3. **Maintainability**: Refactor hooks with Rust tooling
4. **Consistency**: Single language for all tooling

### For Contributors

1. **Zero Setup**: `cargo build` gets everything
2. **Familiar Workflow**: Just `cargo` commands
3. **IDE Support**: Full Rust tooling for hooks
4. **Documentation**: rustdoc for hook logic

## Real-World Example: trueno Project

**Current Implementation** (workaround):

```rust
// xtask/src/install_hooks.rs
const PRE_COMMIT_HOOK: &str = r#"#!/bin/bash
# Manually embedded shell script
cargo run --quiet --package xtask -- check-simd
"#;

pub fn run() -> Result<()> {
    fs::write(".git/hooks/pre-commit", PRE_COMMIT_HOOK)?;
    // ... set permissions ...
}
```

**With bashrs Integration**:

```rust
// xtask/src/hooks/pre_commit.rs
use bashrs::prelude::*;

#[bash_function]
fn run_simd_check() -> BashResult {
    echo!("🔍 Running SIMD validation...");

    let result = cargo_run!("--quiet --package xtask -- check-simd")?;

    if !result.success() {
        echo!("❌ SIMD validation failed");
        exit!(1);
    }

    Ok(())
}

#[bash_main]
fn main() -> BashResult {
    run_simd_check()?;
    Ok(())
}
```

```rust
// xtask/src/install_hooks.rs
use bashrs::Transpiler;

pub fn run() -> Result<()> {
    // Transpile Rust → Shell
    Transpiler::new()
        .input("src/hooks/pre_commit.rs")
        .output(".git/hooks/pre-commit")
        .permissions(0o755)
        .transpile()?;

    println!("✅ Installed pre-commit hook (transpiled from Rust)");
    Ok(())
}
```

## Implementation Roadmap

### Phase 1: Library API (Required)
- [ ] Extract CLI logic to library crate
- [ ] Expose `Transpiler` builder API
- [ ] Support programmatic invocation
- [ ] Add comprehensive error types

### Phase 2: Build Script Integration
- [ ] Create `bashrs-build` crate for build.rs usage
- [ ] Auto-discovery of `*.rs` files to transpile
- [ ] Incremental compilation (only changed files)
- [ ] Integration with cargo's rerun-if-changed

### Phase 3: xtask Pattern Support
- [ ] Documentation for xtask integration
- [ ] Example projects showing patterns
- [ ] Template repository with bashrs + xtask
- [ ] CI/CD examples

### Phase 4: Enhanced Features
- [ ] File watcher for development
- [ ] Source maps for debugging
- [ ] Verification mode (check transpiled output)
- [ ] Diff mode (compare Rust vs Shell)

## Success Criteria

- [ ] Can add bashrs to xtask without global install
- [ ] `cargo build` transpiles hooks automatically
- [ ] No manual `bashrs` CLI invocation needed
- [ ] Works in CI/CD without extra setup
- [ ] Contributors get full functionality with `cargo build`

## References

- **xtask pattern**: https://github.com/matklad/cargo-xtask
- **bashrs repo**: https://github.com/paiml/bashrs
- **trueno usage**: https://github.com/paiml/trueno (uses bashrs for hooks)
- **Cargo build scripts**: https://doc.rust-lang.org/cargo/reference/build-scripts.html

## Questions for bashrs Maintainers

1. Is library API planned for bashrs?
2. Any concerns with build.rs integration?
3. Preferred API design for programmatic use?
4. Timeline for library support?
5. Can we contribute this feature?

---

**Priority**: High
**Complexity**: Medium
**Impact**: Enables natural cargo integration
**Affects**: All projects using bashrs for scripting

**Labels**: enhancement, api, integration, xtask
