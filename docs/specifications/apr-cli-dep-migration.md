# APR-CLI Dep Migration: Old Names → aprender-*

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0
**Date**: 2026-04-08
**Status**: PROPOSAL
**Contract**: `contracts/apr-cli-dep-migration-v1.yaml`
**Blocks**: `cargo install aprender` from crates.io on clean machine

---

## Problem

`cargo install aprender` fails on crates.io because `apr-cli v0.29.0`
depends on old crate names (`batuta = "0.7"`, `realizar = "0.8"`, etc.)
which resolve to the OLD separate-repo crates.io packages. These old
packages depend on `aprender >= 0.27` which creates version conflicts
with `aprender v0.29.0`.

The local workspace compiles because `[lib] name` aliases make old Rust
identifiers work. But crates.io doesn't see the workspace — it resolves
deps independently.

## Root Cause

`crates/apr-cli/Cargo.toml` has:
```toml
batuta = "0.7"           # resolves to old batuta 0.7.3 on crates.io
realizar = "0.8"         # resolves to old realizar 0.8.6
trueno = "0.17"          # resolves to old trueno 0.17.5
entrenar = "0.7"         # resolves to old entrenar 0.7.13
```

Should be:
```toml
aprender-orchestrate = { version = "0.29", package = "aprender-orchestrate" }
aprender-serve = { version = "0.29", package = "aprender-serve" }
aprender-compute = { version = "0.29", package = "aprender-compute" }
aprender-train = { version = "0.29", package = "aprender-train" }
```

But source code uses `use batuta::`, `use realizar::`, etc. — changing
the dep key changes the Rust identifier.

## Solution

For each old dep name in apr-cli:

1. Change Cargo.toml dep key to old name with `package = "new-name"`:
   ```toml
   batuta = { version = "0.29", package = "aprender-orchestrate" }
   ```
   This keeps `use batuta::` working (dep KEY = Rust identifier)
   while resolving to `aprender-orchestrate` on crates.io.

2. No source code changes needed — `use batuta::` still works because
   the dep key is `batuta` and the [lib] name in aprender-orchestrate
   is `batuta`.

## Migration Table

| Old Dep | New Package | Dep Key (keeps Rust ident) |
|---------|-------------|---------------------------|
| `batuta = "0.7"` | `aprender-orchestrate` | `batuta = { version = "0.29", package = "aprender-orchestrate" }` |
| `realizar = "0.8"` | `aprender-serve` | `realizar = { version = "0.29", package = "aprender-serve" }` |
| `trueno = "0.17"` | `aprender-compute` | `trueno = { version = "0.29", package = "aprender-compute" }` |
| `entrenar = "0.7"` | `aprender-train` | `entrenar = { version = "0.29", package = "aprender-train" }` |
| `trueno-gpu = "0.4"` | `aprender-gpu` | `trueno-gpu = { version = "0.29", package = "aprender-gpu" }` |
| `trueno-quant = "0.1"` | `aprender-quant` | `trueno-quant = { version = "0.29", package = "aprender-quant" }` |
| `trueno-viz = "0.2"` | `aprender-viz` | `trueno-viz = { version = "0.29", package = "aprender-viz" }` |
| `batuta-common = "0.1"` | `aprender-common` | `batuta-common = { version = "0.29", package = "aprender-common" }` |
| `renacer = "0.9"` | `aprender-profile` | `renacer = { version = "0.29", package = "aprender-profile" }` |
| `alimentar = "0.2"` | `aprender-data` | `alimentar = { version = "0.29", package = "aprender-data" }` |

## Implementation

1. Write contract (DONE: `apr-cli-dep-migration-v1.yaml`)
2. Update `crates/apr-cli/Cargo.toml` — change version-only deps to `package = "new"`
3. Verify `cargo check -p apr-cli` still compiles
4. Republish `apr-cli` to crates.io
5. Republish root `aprender` facade
6. Verify `cargo install aprender` from crates.io on clean machine

## After Migration

`cargo install aprender` will resolve:
```
aprender v0.29 → apr-cli v0.29 → aprender-orchestrate v0.29
                                → aprender-serve v0.29
                                → aprender-compute v0.29
                                → aprender-train v0.29
```

No old separate-repo crates in the dependency chain.
