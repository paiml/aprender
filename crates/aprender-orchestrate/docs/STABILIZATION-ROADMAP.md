# Batuta 1.0 Stabilization Roadmap

> **Goal**: Transform batuta from 0.6.0 (development) to 1.0.0 (stable release)

## Executive Summary

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Version | 0.6.0 | 1.0.0 | 4 releases |
| `unwrap()` calls | 1,085 | 0 (public paths) | -1,085 |
| Unit tests | 2,531 | 2,531+ | ✅ |
| Public API items | ~286 | ~50 (curated) | -236 |
| crates.io | 0.5.0 | 1.0.0 | Behind |

## Phase 1: Error Handling (P0-Critical)

Replace all `unwrap()` calls with proper error handling. Each unwrap is a potential panic that crashes user workflows.

### STABLE-001: CLI Module
- **Priority**: P0-Critical (user-facing)
- **Current**: 5 unwrap() calls
- **Target**: 0
- **Files**: `src/cli/*.rs`
- **Target Date**: 2026-02-15

### STABLE-002: Stack Module
- **Priority**: P0-Critical (stack management)
- **Current**: 205 unwrap() calls
- **Target**: 0
- **Files**: `src/stack/*.rs` (types.rs is 977 lines - also needs splitting)
- **Target Date**: 2026-02-28

### STABLE-003: Oracle Module
- **Priority**: P1-High (recommendations/RAG)
- **Current**: 135 unwrap() calls
- **Target**: 0
- **Files**: `src/oracle/*.rs`, `src/oracle/rag/*.rs`
- **Target Date**: 2026-03-15

### STABLE-004: Serve Module
- **Priority**: P0-Critical (production serving)
- **Current**: 29 unwrap() calls
- **Target**: 0
- **Files**: `src/serve/*.rs`
- **Target Date**: 2026-02-28

### STABLE-005: Converter Modules
- **Priority**: P1-High (core transpilation)
- **Current**: ~396 unwrap() calls
- **Target**: 0
- **Files**: `src/numpy_converter.rs`, `src/sklearn_converter.rs`, `src/pytorch_converter.rs`, `src/backend.rs`
- **Target Date**: 2026-03-31

**Phase 1 Total**: 770 unwrap() calls to eliminate

## Phase 2: API Stability (P1-High)

### STABLE-006: API Surface Audit
- **Priority**: P1-High
- **Current**: ~286 public items
- **Target**: ~50 curated items in prelude
- **Tasks**:
  - Create `src/prelude.rs` with stable re-exports
  - Change internal modules to `pub(crate)`
  - Add `#[doc(hidden)]` for semi-public items
  - Document all remaining public items
- **Target Date**: 2026-04-15
- **Blocked By**: STABLE-001 through STABLE-005

### STABLE-007: Dependency Pinning
- **Priority**: P2-Medium
- **Current**: Semver ranges (`"0.14"`, `"1.0"`)
- **Target**: Exact versions (`"=0.14.0"`)
- **Stack Dependencies**:
  ```toml
  trueno = "=0.14.0"
  renacer = "=0.9.8"
  repartir = "=2.0.0"
  trueno-graph = "=0.1.11"
  trueno-db = "=0.3.11"
  ```
- **Target Date**: 2026-04-30

## Phase 3: Documentation (P2-Medium)

### STABLE-008: Documentation Completeness
- **Priority**: P2-Medium
- **Tasks**:
  - All public items have `///` doc comments
  - `cargo doc` builds with zero warnings
  - Book covers all major features
  - CHANGELOG.md maintained
  - Migration guide (0.x → 1.0)
- **Target Date**: 2026-05-15
- **Blocked By**: STABLE-006

## Phase 4: Release (P2-Medium)

### STABLE-009: 1.0 Release Process
- **Priority**: P2-Medium
- **Target Date**: 2026-06-01
- **Blocked By**: All previous tickets

**Pre-Release Checklist**:
- [ ] All STABLE-00x tickets completed
- [ ] Zero unwrap() in public code paths
- [ ] API surface locked and documented
- [ ] All deps pinned
- [ ] CHANGELOG complete
- [ ] Version bumped to 1.0.0
- [ ] Release candidate (0.9.0-rc.1) published
- [ ] Community testing period (2 weeks)
- [ ] Final security audit

**Release Steps**:
1. Tag `v1.0.0` in git
2. `cargo publish`
3. Update docs site
4. Announce on GitHub releases
5. Update stack components to use batuta 1.0

## Timeline

```
2026-02     2026-03     2026-04     2026-05     2026-06
    │           │           │           │           │
    ├── CLI ────┤           │           │           │
    ├── STACK ──┤           │           │           │
    ├── SERVE ──┤           │           │           │
    │           ├── ORACLE ─┤           │           │
    │           ├── CONVERT─┤           │           │
    │           │           ├── API ────┤           │
    │           │           ├── DEPS ───┤           │
    │           │           │           ├── DOCS ──┤
    │           │           │           │          ├─ 1.0.0
    ▼           ▼           ▼           ▼          ▼
  Phase 1    Phase 1    Phase 2    Phase 3    Phase 4
```

## Error Handling Patterns

### Before (panic-prone)
```rust
let value = something.unwrap();
let parsed = text.parse::<i32>().unwrap();
let item = collection.get(0).unwrap();
```

### After (graceful)
```rust
// With context for debugging
let value = something.context("operation failed")?;

// With default for optional values
let value = something.unwrap_or_default();

// With custom error
let parsed = text.parse::<i32>()
    .map_err(|e| anyhow!("invalid number '{}': {}", text, e))?;

// With Option handling
let item = collection.get(0)
    .ok_or_else(|| anyhow!("collection is empty"))?;
```

## Success Criteria for 1.0

| Criterion | Requirement |
|-----------|-------------|
| Error Handling | Zero unwrap() in user-facing code paths |
| Test Coverage | ≥95% line coverage |
| Documentation | All public items documented |
| API Stability | Prelude contains only stable, versioned API |
| Dependencies | All pinned, zero audit warnings |
| Performance | No regressions from 0.5.0 |

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [The Cargo Book: Publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)
