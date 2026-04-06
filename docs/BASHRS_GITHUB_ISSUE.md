# Support cargo xtask-like Integration for Natural Workflow

## Summary

Enable bashrs to work naturally within cargo's xtask pattern, allowing developers to transpile Rust → Shell scripts as part of their standard development workflow without requiring global `cargo install bashrs`.

## Problem

**Current State**:
```bash
# Global installation required
cargo install bashrs

# Separate binary invocation
bashrs transpile src/script.rs --output script.sh

# Not integrated with project workflow
# External dependency for contributors
```

**Pain Points**:
- Requires all contributors to install bashrs globally
- Not version-locked with project
- Separate from project's development tooling
- Extra setup in CI/CD
- Doesn't leverage cargo's workspace features

## Desired State

```bash
# Contributors just clone and build
git clone https://github.com/org/project
cargo build  # bashrs included, hooks transpiled automatically

# Integrated with xtask
cargo xtask transpile-hooks

# No global installation needed
# Version-locked via Cargo.lock
# Works immediately in CI/CD
```

## Use Case: Git Hooks in Rust

**Current Workaround** (what we're doing now):
```rust
// Manually embed shell scripts in Rust strings
const PRE_COMMIT_HOOK: &str = r#"#!/bin/bash
cargo run --quiet --package xtask -- check-simd
if [ $? -ne 0 ]; then
    echo "❌ Check failed"
    exit 1
fi
"#;
```

**With bashrs Library Integration**:
```rust
// Write hooks in Rust with type safety
use bashrs::prelude::*;

#[bash_function]
fn check_simd() -> BashResult {
    let result = cargo_run!("xtask check-simd")?;
    if !result.success() {
        eprintln!("❌ SIMD validation failed");
        exit!(1);
    }
    Ok(())
}

#[bash_main]
fn main() -> BashResult {
    check_simd()?;
    Ok(())
}
```

Then transpile programmatically:
```rust
// xtask/src/transpile_hooks.rs
use bashrs::Transpiler;

pub fn run() -> Result<()> {
    Transpiler::new()
        .input("src/hooks/pre_commit.rs")
        .output(".git/hooks/pre-commit")
        .permissions(0o755)
        .transpile()?;
    Ok(())
}
```

## Proposed API

### Core Library API

```rust
// bashrs/src/lib.rs
pub struct Transpiler {
    input: PathBuf,
    output: PathBuf,
    permissions: Option<u32>,
    options: TranspileOptions,
}

impl Transpiler {
    pub fn new() -> Self { /* ... */ }

    pub fn input(mut self, path: impl AsRef<Path>) -> Self {
        self.input = path.as_ref().to_path_buf();
        self
    }

    pub fn output(mut self, path: impl AsRef<Path>) -> Self {
        self.output = path.as_ref().to_path_buf();
        self
    }

    pub fn permissions(mut self, mode: u32) -> Self {
        self.permissions = Some(mode);
        self
    }

    pub fn transpile(&self) -> Result<TranspileResult> {
        // Existing transpile logic
    }
}

pub struct TranspileResult {
    pub input: PathBuf,
    pub output: PathBuf,
    pub warnings: Vec<Warning>,
    pub stats: Stats,
}
```

### Build Script Integration

```rust
// bashrs-build/src/lib.rs (new crate)
pub fn transpile_hooks() -> Result<()> {
    // Auto-discover src/hooks/*.rs
    // Transpile to .git/hooks/*
    // Set executable permissions
    // Register with cargo rerun-if-changed
}
```

**Usage in project**:
```rust
// xtask/build.rs
fn main() {
    bashrs_build::transpile_hooks()
        .expect("Failed to transpile hooks");
}
```

### Workspace Integration Pattern

```toml
# Cargo.toml (workspace)
[workspace]
members = [".", "xtask"]

# xtask/Cargo.toml
[dependencies]
bashrs = "6.35"  # As library, not installed globally
anyhow = "1.0"
```

```rust
// xtask/src/main.rs
mod transpile_hooks;

fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("transpile-hooks") => transpile_hooks::run(),
        // ... other commands
    }
}
```

## Benefits

### For bashrs
- ✅ Increased adoption (easier to integrate)
- ✅ Better developer experience
- ✅ CI/CD friendly
- ✅ Fits Rust ecosystem patterns

### For Projects
- ✅ Type-safe shell scripts
- ✅ Unit testable before transpilation
- ✅ IDE support (Rust tooling)
- ✅ Refactorable with cargo fix

### For Contributors
- ✅ Zero setup (cargo build gets everything)
- ✅ Version-locked dependencies
- ✅ Works offline (no install step)
- ✅ Familiar cargo workflow

## Real Example: trueno Project

We're currently using bashrs in trueno for git hooks and would greatly benefit from this integration:

**Current Structure**:
```
trueno/
├── xtask/
│   ├── Cargo.toml         # has bashrs = "6.35"
│   └── src/
│       ├── install_hooks.rs  # Embeds shell strings 😞
│       └── check_simd.rs
```

**Desired Structure**:
```
trueno/
├── xtask/
│   ├── Cargo.toml         # bashrs = "6.35"
│   ├── build.rs           # Auto-transpile on build
│   └── src/
│       ├── install_hooks.rs  # Uses bashrs::Transpiler 🎉
│       ├── check_simd.rs
│       └── hooks/
│           ├── pre_commit.rs   # Rust with #[bash_main]
│           └── post_merge.rs   # Rust with #[bash_main]
```

## Implementation Phases

### Phase 1: Library API (MVP)
- Extract CLI logic into library
- Expose `Transpiler` builder API
- Keep CLI as thin wrapper over library
- Add library examples to docs

**Effort**: ~2-3 days
**Impact**: Enables programmatic use

### Phase 2: Build Script Support
- Create `bashrs-build` helper crate
- Auto-discovery of files to transpile
- Cargo rerun-if-changed integration
- Incremental compilation

**Effort**: ~1-2 days
**Impact**: Automatic transpilation during cargo build

### Phase 3: Documentation & Examples
- xtask integration guide
- Example repository
- Migration guide from global install
- CI/CD setup examples

**Effort**: ~1 day
**Impact**: Easier adoption

## Questions

1. **Is library API planned?** Or is bashrs intended as CLI-only?
2. **API design preferences?** Builder pattern? Functional? Both?
3. **Breaking changes concerns?** Can we add library API without breaking CLI?
4. **Timeline?** Is this feature planned for a specific version?
5. **Contributions welcome?** Can we submit PRs for this?

## Success Criteria

- [ ] Can add `bashrs = "6.35"` to `[dependencies]` and use programmatically
- [ ] `cargo build` can trigger transpilation via build.rs
- [ ] No global `cargo install bashrs` needed
- [ ] Works in fresh git clone without any setup
- [ ] CI/CD needs no special bashrs installation step

## Alternatives Considered

### 1. Keep Current CLI-Only Approach
**Pros**: No changes needed
**Cons**: Poor integration, manual setup required

### 2. Create Separate cargo-bashrs Subcommand
**Pros**: Familiar pattern (like cargo-watch)
**Cons**: Still requires installation, not workspace-integrated

### 3. Recommended: Library + CLI
**Pros**: Best of both worlds, flexible usage
**Cons**: Requires refactoring existing code

## References

- **xtask pattern**: https://github.com/matklad/cargo-xtask
- **trueno project**: https://github.com/paiml/trueno (current bashrs usage)
- **Cargo build scripts**: https://doc.rust-lang.org/cargo/reference/build-scripts.html
- **Builder pattern in Rust**: https://rust-lang.github.io/api-guidelines/type-safety.html

---

**Note**: We're actively using bashrs in trueno and would be happy to contribute this feature if the maintainers are open to it!
