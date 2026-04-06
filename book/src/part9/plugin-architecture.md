# Plugin Architecture (Future)

This chapter describes the planned plugin system for extending Batuta with custom transpilers, optimization passes, and validation hooks. This feature is under development.

## Motivation

A plugin system would enable:

- Custom transpilers for additional languages (Go, Java, TypeScript)
- Domain-specific optimization passes
- Custom validation hooks (e.g., regulatory compliance)
- Alternative backend selectors for specialized hardware

## Planned Plugin API

Plugins will implement a trait-based interface:

```rust
pub trait TranspilerPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn supported_languages(&self) -> &[Language];
    fn transpile(&self, input: &SourceFile) -> Result<RustOutput, TranspileError>;
}

pub trait ValidationPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, original: &SourceFile, transpiled: &RustOutput)
        -> Result<ValidationReport>;
}
```

## Hook Points in the Pipeline

```
Phase 1: Analysis     -> post_analysis hook
Phase 2: Transpile    -> pre_transpile, transpile, post_transpile hooks
Phase 3: Optimization -> pre_optimize, optimize, post_optimize hooks
Phase 4: Validation   -> validate hook
Phase 5: Build        -> post_build hook
```

## Plugin Configuration

```toml
# batuta.toml
[plugins]
search_paths = ["~/.batuta/plugins", "./plugins"]

[[plugins.transpiler]]
name = "go-transpiler"
path = "libgo_transpiler.so"

[[plugins.validation]]
name = "compliance-checker"
path = "libcompliance.so"
config = { standard = "SOX" }
```

## Discovery Order

1. Built-in transpilers (depyler, decy, bashrs) always available
2. Plugins declared in `batuta.toml`
3. Shared libraries in `search_paths` matching `lib*_plugin.so`

## Security Considerations

| Measure | Purpose |
|---------|---------|
| SHA-256 checksums in config | Verify plugin integrity |
| API version checking | Prevent incompatible plugins |
| Explicit opt-in | No automatic discovery by default |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
