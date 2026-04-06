# Release Builds

Release builds produce optimized binaries for production deployment. Phase 5 applies Cargo profile settings tuned during Phase 3 optimization.

## Optimization Profiles

Phase 3 writes `[profile.release]` settings to the output project's `Cargo.toml`. Three profiles are available:

| Profile | `opt-level` | LTO | `codegen-units` | Strip | Use Case |
|---------|------------|-----|----------------|-------|----------|
| Fast | 2 | off | 16 | No | Quick iteration, CI |
| Balanced | 3 | thin | 4 | No | Default production |
| Aggressive | 3 | full | 1 | symbols | Maximum performance |

## Cargo.toml Configuration

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = "symbols"
panic = "abort"
```

### What Each Setting Does

**`opt-level = 3`** -- Maximum optimization. Enables auto-vectorization, loop unrolling, and function inlining beyond the default level 2.

**`lto = "fat"`** -- Link-Time Optimization across all crates. Allows the linker to optimize across crate boundaries, eliminating dead code and enabling cross-crate inlining. Increases build time significantly.

**`codegen-units = 1`** -- Forces single-threaded code generation. This allows LLVM to see the entire crate at once, enabling better optimization at the cost of slower compilation.

**`strip = "symbols"`** -- Removes debug symbols from the final binary, reducing size by 50-80%.

**`panic = "abort"`** -- Generates abort on panic instead of unwinding. Reduces binary size and improves performance by eliminating unwind tables.

## Profile-Guided Optimization (PGO)

For maximum performance, PGO uses a profiling run to guide optimization:

```bash
# Step 1: Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
    cargo build --release

# Step 2: Run representative workload
./target/release/app < benchmark-input.txt

# Step 3: Rebuild with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
    cargo build --release
```

PGO typically provides an additional 5-15% speedup over standard release builds by optimizing branch prediction and code layout.

## Size Optimization

For deployment-constrained environments (embedded, WASM):

```toml
[profile.release]
opt-level = "z"      # Optimize for size
lto = true
codegen-units = 1
strip = true
panic = "abort"
```

## CLI Usage

```bash
# Standard release build
batuta build --release

# With aggressive optimization
batuta build --release --profile aggressive

# Check binary size
ls -lh rust-output/target/release/app
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
