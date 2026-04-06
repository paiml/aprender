# Transpilation Failures

Transpilation failures occur in Phase 2 when source code cannot be converted to Rust. The three main categories are missing tools, unsupported features, and dependency resolution failures.

## Missing Tool Detection

```bash
# Check all transpilers
batuta analyze --check-tools
```

| Language | Transpiler | Install Command |
|----------|-----------|-----------------|
| Python | depyler | `cargo install depyler` |
| C/C++ | decy | `cargo install decy` |
| Shell | bashrs | `cargo install bashrs` |

## Unsupported Language Features

### Python

| Feature | Status | Workaround |
|---------|--------|------------|
| `eval()` / `exec()` | Unsupported | Refactor to static code |
| `getattr` (dynamic) | Partial | Use enum dispatch |
| Multiple inheritance | Unsupported | Trait composition |
| `*args, **kwargs` | Partial | Explicit params or builder |
| `async/await` | Supported | Maps to tokio async |

### C

| Feature | Status | Workaround |
|---------|--------|------------|
| `goto` | Unsupported | Refactor to loops/match |
| Pointer arithmetic | Partial | Slice indexing |
| Variadic functions | Partial | Macro or builder |
| `setjmp`/`longjmp` | Unsupported | `Result` error handling |

## Dependency Resolution Failures

Batuta maps source dependencies to Rust crate equivalents:

| Python Package | Rust Crate | Notes |
|---------------|------------|-------|
| numpy | trueno | Stack native |
| scikit-learn | aprender | Stack native |
| torch | realizar | Inference only |
| pandas | polars / alimentar | alimentar for Arrow |
| requests | reqwest | Async HTTP |
| flask | axum | Async web framework |

### When Mapping Fails

Batuta halts with a Jidoka stop. Options:

1. **Add manual mapping** in `batuta.toml`
2. **Wrap via FFI** (keep the original library)
3. **Implement directly** in Rust

```toml
[dependencies.mapping]
obscure_lib = { crate = "my-rust-alternative", version = "0.1" }
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
